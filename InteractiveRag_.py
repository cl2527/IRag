import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List, Dict, Tuple, Set
import re

# -----------------------------
# 0. Qwen2.5-7B LLM Client
# -----------------------------
class QwenClient:
    """Local LLM client using Qwen2.5-7B model"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.device = device
        print(f"Model loaded on {device}")
    
    def chat_completion(self, messages, max_tokens=1024, temperature=0.7):
        """Generate chat completion compatible with OpenAI API format"""
        # Format messages for Qwen
        prompt = self._format_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Return in OpenAI-like format
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': response_text
                })()
            })()]
        })()
    
    def _format_messages(self, messages):
        """Format messages for Qwen chat template"""
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Add assistant prompt
        formatted += "<|im_start|>assistant\n"
        return formatted


# -----------------------------
# 1. Corpus Interaction Engine
# -----------------------------
class CorpusInteractionEngine:
    """
    Interactive engine with fine-grained information control.
    Supports multiple retrieval strategies and context manipulation.
    """
    
    def __init__(self, documents: List[Dict]):
        """
        Args:
            documents: List of dicts with 'id', 'text', 'entities' keys
        """
        self.documents = documents
        self.doc_id_map = {doc['id']: doc for doc in documents}
        
        # Setup semantic search (dense retrieval)
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = self.semantic_model.encode(
            [doc['text'] for doc in documents], 
            convert_to_numpy=True
        )
        
        # Setup exact search (sparse retrieval with FTS)
        self._init_fts_index()
        
        # Fusion weights (default: balanced)
        self.weight_semantic = 0.5
        self.weight_exact = 0.5
        
        # Context shaping state
        self.included_docs: Set[str] = set()
        self.excluded_docs: Set[str] = set()
        self.retrieval_scale = 5  # default k
    
    def _init_fts_index(self):
        """Initialize SQLite FTS5 for exact keyword search"""
        self.conn = sqlite3.connect(':memory:')
        cursor = self.conn.cursor()
        
        # Create FTS5 virtual table
        cursor.execute('''
            CREATE VIRTUAL TABLE documents_fts 
            USING fts5(doc_id, text)
        ''')
        
        # Insert documents
        for doc in self.documents:
            cursor.execute(
                'INSERT INTO documents_fts (doc_id, text) VALUES (?, ?)',
                (doc['id'], doc['text'])
            )
        
        self.conn.commit()
    
    # -------------------------
    # Multi-Faceted Retrieval
    # -------------------------
    
    def semantic_search(self, query: str, k: int = None) -> List[Dict]:
        """
        Dense retrieval using embedding similarity.
        Returns documents semantically related to query.
        """
        k = k or self.retrieval_scale
        
        # Encode query
        query_emb = self.semantic_model.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = np.dot(self.doc_embeddings, query_emb.T).squeeze()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            if self._is_doc_allowed(doc['id']):
                results.append({
                    'doc_id': doc['id'],
                    'text': doc['text'],
                    'semantic_score': float(similarities[idx]),
                    'exact_score': 0.0
                })
        
        return results[:k]
    
    def exact_search(self, keywords: str, k: int = None) -> List[Dict]:
        """
        Sparse retrieval using exact keyword matching with BM25-like ranking.
        Ideal for finding specific terms, names, or phrases.
        """
        k = k or self.retrieval_scale
        
        # FTS5 query
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT doc_id, rank 
            FROM documents_fts 
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (keywords, k * 2))  # Get more for filtering
        
        results = []
        for doc_id, rank in cursor.fetchall():
            if self._is_doc_allowed(doc_id):
                doc = self.doc_id_map[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'text': doc['text'],
                    'semantic_score': 0.0,
                    'exact_score': abs(float(rank))  # FTS5 rank is negative
                })
                
                if len(results) >= k:
                    break
        
        return results
    
    def weighted_fusion(self, query: str, keywords: str = None, 
                       ws: float = None, we: float = None, k: int = None) -> List[Dict]:
        """
        Fused retrieval combining semantic and exact search with weights.
        
        Args:
            query: Semantic query
            keywords: Keywords for exact search (uses query if None)
            ws: Semantic weight (uses self.weight_semantic if None)
            we: Exact weight (uses self.weight_exact if None)
            k: Number of results
        """
        k = k or self.retrieval_scale
        ws = ws if ws is not None else self.weight_semantic
        we = we if we is not None else self.weight_exact
        keywords = keywords or query
        
        # Get results from both strategies
        semantic_results = self.semantic_search(query, k=k*2)
        exact_results = self.exact_search(keywords, k=k*2)
        
        # Normalize scores
        semantic_scores = {r['doc_id']: r['semantic_score'] for r in semantic_results}
        exact_scores = {r['doc_id']: r['exact_score'] for r in exact_results}
        
        # Normalize to [0, 1]
        if semantic_scores:
            max_sem = max(semantic_scores.values())
            semantic_scores = {k: v/max_sem if max_sem > 0 else 0 
                             for k, v in semantic_scores.items()}
        
        if exact_scores:
            max_exact = max(exact_scores.values())
            exact_scores = {k: v/max_exact if max_exact > 0 else 0 
                          for k, v in exact_scores.items()}
        
        # Combine scores
        all_doc_ids = set(semantic_scores.keys()) | set(exact_scores.keys())
        fused_scores = {}
        
        for doc_id in all_doc_ids:
            if self._is_doc_allowed(doc_id):
                sem_score = semantic_scores.get(doc_id, 0.0)
                ex_score = exact_scores.get(doc_id, 0.0)
                fused_scores[doc_id] = ws * sem_score + we * ex_score
        
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:k]:
            doc = self.doc_id_map[doc_id]
            results.append({
                'doc_id': doc_id,
                'text': doc['text'],
                'semantic_score': semantic_scores.get(doc_id, 0.0),
                'exact_score': exact_scores.get(doc_id, 0.0),
                'fused_score': score
            })
        
        return results
    
    # -------------------------
    # Anchored Matching
    # -------------------------
    
    def entity_match(self, entity: str, k: int = None) -> List[Dict]:
        """
        Retrieves information segments strongly associated with specified entity.
        Ensures results are centered around a key subject.
        """
        k = k or self.retrieval_scale
        
        results = []
        for doc in self.documents:
            if not self._is_doc_allowed(doc['id']):
                continue
            
            # Check if entity exists in document
            text_lower = doc['text'].lower()
            entity_lower = entity.lower()
            
            if entity_lower in text_lower:
                # Count entity mentions
                mention_count = text_lower.count(entity_lower)
                
                # Check if entity in metadata
                entity_in_metadata = False
                if 'entities' in doc and entity_lower in [e.lower() for e in doc['entities']]:
                    entity_in_metadata = True
                
                # Compute relevance score
                relevance = mention_count * (2 if entity_in_metadata else 1)
                
                results.append({
                    'doc_id': doc['id'],
                    'text': doc['text'],
                    'entity': entity,
                    'mention_count': mention_count,
                    'relevance_score': relevance
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:k]
    
    # -------------------------
    # Context Shaping
    # -------------------------
    
    def include_docs(self, doc_ids: List[str]):
        """
        Guarantees inclusion of specified documents in subsequent retrieval.
        Ensures critical information is not missed.
        """
        self.included_docs.update(doc_ids)
    
    def exclude_docs(self, doc_ids: List[str]):
        """
        Filters out irrelevant documents from subsequent searches.
        Prevents noisy distractions.
        """
        self.excluded_docs.update(doc_ids)
    
    def adjust_scale(self, n: int):
        """
        Adaptively adjusts the scale of retrieved information.
        Matches different complexity of sub-problems.
        """
        self.retrieval_scale = max(1, n)
    
    def set_fusion_weights(self, ws: float, we: float):
        """Set fusion weights for semantic and exact search"""
        total = ws + we
        self.weight_semantic = ws / total
        self.weight_exact = we / total
    
    def reset_context(self):
        """Reset context shaping state"""
        self.included_docs.clear()
        self.excluded_docs.clear()
        self.retrieval_scale = 5
        self.weight_semantic = 0.5
        self.weight_exact = 0.5
    
    def _is_doc_allowed(self, doc_id: str) -> bool:
        """Check if document passes context shaping filters"""
        if doc_id in self.excluded_docs:
            return False
        return True
    
    def get_included_docs(self) -> List[Dict]:
        """Get documents that must be included"""
        results = []
        for doc_id in self.included_docs:
            if doc_id in self.doc_id_map:
                doc = self.doc_id_map[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'text': doc['text'],
                    'included': True
                })
        return results
    
    def execute_actions(self, actions: List[Dict]) -> Dict:
        """
        Execute multiple interaction primitives and aggregate results.
        
        Args:
            actions: List of action dicts with 'action' and 'params' keys
            
        Returns:
            Consolidated response with aggregated content and metadata
        """
        all_results = []
        metadata = {
            'actions_executed': [],
            'total_docs': 0
        }
        
        for action in actions:
            action_name = action['action']
            params = action.get('params', {})
            
            try:
                if action_name == 'semantic_search':
                    results = self.semantic_search(**params)
                elif action_name == 'exact_search':
                    results = self.exact_search(**params)
                elif action_name == 'weighted_fusion':
                    results = self.weighted_fusion(**params)
                elif action_name == 'entity_match':
                    results = self.entity_match(**params)
                elif action_name == 'include_docs':
                    self.include_docs(params['doc_ids'])
                    results = self.get_included_docs()
                elif action_name == 'exclude_docs':
                    self.exclude_docs(params['doc_ids'])
                    results = []
                elif action_name == 'adjust_scale':
                    self.adjust_scale(params['n'])
                    results = []
                else:
                    results = []
                
                all_results.extend(results)
                metadata['actions_executed'].append(action_name)
                
            except Exception as e:
                metadata['actions_executed'].append(f"{action_name} (failed: {str(e)})")
        
        # Deduplicate results by doc_id
        seen_ids = set()
        unique_results = []
        for result in all_results:
            doc_id = result['doc_id']
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        metadata['total_docs'] = len(unique_results)
        
        return {
            'results': unique_results,
            'metadata': metadata
        }


# -----------------------------
# 2. Reasoning-Enhanced Workflow
# -----------------------------
class GlobalPlanner:
    """
    Analyzes user query and decomposes it into step-by-step execution plan.
    Provides high-level strategic roadmap.
    """
    
    def __init__(self, llm_client):
        self.client = llm_client
    
    def create_plan(self, query: str) -> Dict:
        """
        Generate a high-level plan for answering the query.
        
        Returns:
            Dict with 'analysis' and 'steps' keys
        """
        prompt = f"""Analyze the following query and create a step-by-step execution plan for answering it.

Query: {query}

Provide:
1. A brief analysis of the query complexity and key information needs
2. A numbered list of execution steps (3-5 steps)

Format your response as JSON:
{{
    "analysis": "Brief analysis of the query...",
    "steps": [
        "Step 1: ...",
        "Step 2: ...",
        ...
    ]
}}
"""
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        
        try:
            plan = json.loads(response.choices[0].message.content)
        except:
            # Fallback parsing
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            plan = json.loads(content[start:end])
        
        return plan


class AdaptiveReasoner:
    """
    Cognitive core that analyzes state and issues directives.
    Can issue 'Proceed' or 'Reflect & Refine' directives.
    """
    
    def __init__(self, llm_client):
        self.client = llm_client
    
    def analyze_and_direct(self, state: Dict) -> Dict:
        """
        Analyze current state and issue directive.
        
        Args:
            state: Dict with 'query', 'plan', 'current_step', 'history', 'retrieved_info'
            
        Returns:
            Dict with 'directive' ('proceed' or 'refine'), 'analysis', and 'strategy'
        """
        prompt = f"""You are an adaptive reasoner analyzing the progress of an information retrieval task.

Query: {state['query']}

Plan: {json.dumps(state['plan']['steps'], indent=2)}

Current Step: {state['current_step']}

Action History:
{self._format_history(state['history'])}

Retrieved Information:
{self._format_retrieved_info(state.get('retrieved_info', []))}

Analyze the current state and decide:
1. Is the retrieved information sufficient for the current step?
2. Should we PROCEED to next step or REFINE the retrieval strategy?
3. If refining, what specific changes should be made?

Respond in JSON format:
{{
    "directive": "proceed" or "refine",
    "analysis": "Your analysis of the current state...",
    "strategy": "If refining, describe the new strategy. If proceeding, describe next action."
}}
"""
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        
        try:
            directive = json.loads(response.choices[0].message.content)
        except:
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            directive = json.loads(content[start:end])
        
        return directive
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format action history for prompt"""
        text = ""
        for i, entry in enumerate(history[-5:]):  # Last 5 entries
            text += f"Step {i+1}: {entry.get('action', 'N/A')}\n"
            text += f"  Result: {entry.get('result_summary', 'N/A')}\n"
        return text or "No history yet."
    
    def _format_retrieved_info(self, info: List[Dict]) -> str:
        """Format retrieved information for prompt"""
        if not info:
            return "No information retrieved yet."
        
        text = ""
        for i, doc in enumerate(info[:3]):  # Show first 3 docs
            text += f"Doc {i+1} (ID: {doc.get('doc_id', 'N/A')}): {doc.get('text', '')[:100]}...\n"
        
        if len(info) > 3:
            text += f"... and {len(info) - 3} more documents\n"
        
        return text


class Executor:
    """
    Translates strategy into concrete, structured actions.
    Generates precise function calls for interaction primitives.
    """
    
    def __init__(self, llm_client):
        self.client = llm_client
    
    def generate_actions(self, directive: Dict, state: Dict) -> List[Dict]:
        """
        Generate concrete actions based on reasoner's directive.
        
        Returns:
            List of action dicts with 'action' and 'params' keys
        """
        prompt = f"""You are an executor that translates strategy into concrete actions.

Available Actions:
- semantic_search(query): Dense retrieval using embeddings
- exact_search(keywords): Sparse retrieval with exact keywords
- weighted_fusion(query, keywords, ws, we): Fused retrieval with weights
- entity_match(entity): Find docs related to specific entity
- include_docs(doc_ids): Force include certain documents
- exclude_docs(doc_ids): Filter out certain documents
- adjust_scale(n): Set number of documents to retrieve

Directive: {directive['directive']}
Strategy: {directive['strategy']}

Current Query: {state['query']}
Current Step: {state['current_step']}

Generate 1-3 concrete actions to execute. Respond in JSON format:
{{
    "actions": [
        {{"action": "action_name", "params": {{"param1": "value1"}}}},
        ...
    ]
}}
"""
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except:
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            result = json.loads(content[start:end])
        
        return result.get('actions', [])
    
    def generate_final_answer(self, query: str, retrieved_info: List[Dict]) -> str:
        """Generate final answer based on retrieved information"""
        context = "\n\n".join([
            f"[Doc {i+1}]: {doc['text']}" 
            for i, doc in enumerate(retrieved_info[:5])
        ])
        
        prompt = f"""Based on the following retrieved information, answer the query.

Query: {query}

Retrieved Information:
{context}

Provide a clear, concise answer based on the information above.
"""
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        
        return response.choices[0].message.content


# -----------------------------
# 3. Interactive RAG Agent
# -----------------------------
class InteractiveRAGAgent:
    """
    Complete Interactive RAG system with reasoning-enhanced workflow.
    Integrates Corpus Interaction Engine with modular reasoning components.
    """
    
    def __init__(self, documents: List[Dict], llm_client=None, device="cuda"):
        self.engine = CorpusInteractionEngine(documents)
        self.client = llm_client or QwenClient(device=device)
        
        # Workflow modules
        self.planner = GlobalPlanner(self.client)
        self.reasoner = AdaptiveReasoner(self.client)
        self.executor = Executor(self.client)
    
    def query(self, query: str, max_steps: int = 6) -> Dict:
        """
        Process query using reasoning-enhanced workflow.
        
        Returns:
            Dict with 'answer', 'plan', 'history', 'retrieved_docs'
        """
        # Reset engine state
        self.engine.reset_context()
        
        # Step 1: Global Planning
        print("=" * 50)
        print("Step 1: Global Planning")
        print("=" * 50)
        plan = self.planner.create_plan(query)
        print(f"Analysis: {plan['analysis']}")
        print(f"Steps: {json.dumps(plan['steps'], indent=2)}")
        
        # Initialize state
        state = {
            'query': query,
            'plan': plan,
            'current_step': 0,
            'history': [],
            'retrieved_info': []
        }
        
        # Step 2: Iterative Reasoning and Execution
        for step in range(max_steps):
            print(f"\n{'='*50}")
            print(f"Step {step + 1}: Reasoning & Execution")
            print(f"{'='*50}")
            
            # Adaptive reasoning
            directive = self.reasoner.analyze_and_direct(state)
            print(f"Directive: {directive['directive'].upper()}")
            print(f"Analysis: {directive['analysis']}")
            print(f"Strategy: {directive['strategy']}")
            
            # Check if we should proceed to answer
            if directive['directive'] == 'proceed' and step > 0:
                if 'answer' in directive['strategy'].lower() or step >= len(plan['steps']):
                    break
            
            # Generate and execute actions
            actions = self.executor.generate_actions(directive, state)
            print(f"Actions: {json.dumps(actions, indent=2)}")
            
            response = self.engine.execute_actions(actions)
            
            # Update state
            state['retrieved_info'].extend(response['results'])
            state['history'].append({
                'step': step + 1,
                'directive': directive['directive'],
                'actions': actions,
                'result_summary': f"Retrieved {len(response['results'])} documents"
            })
            state['current_step'] += 1
            
            print(f"Retrieved: {len(response['results'])} documents")
        
        # Step 3: Generate Final Answer
        print(f"\n{'='*50}")
        print("Step 3: Generating Final Answer")
        print(f"{'='*50}")
        
        answer = self.executor.generate_final_answer(query, state['retrieved_info'])
        print(f"Answer: {answer}")
        
        return {
            'answer': answer,
            'plan': plan,
            'history': state['history'],
            'retrieved_docs': state['retrieved_info']
        }


# -----------------------------
# 4. Example Usage
# -----------------------------
if __name__ == "__main__":
    # Sample documents
    documents = [
        {
            'id': 'doc1',
            'text': 'Paris is the capital and largest city of France.',
            'entities': ['Paris', 'France']
        },
        {
            'id': 'doc2',
            'text': 'The Eiffel Tower is located in Paris, France.',
            'entities': ['Eiffel Tower', 'Paris', 'France']
        },
        {
            'id': 'doc3',
            'text': 'France is a country in Western Europe.',
            'entities': ['France', 'Europe']
        },
        {
            'id': 'doc4',
            'text': 'Berlin is the capital of Germany.',
            'entities': ['Berlin', 'Germany']
        },
        {
            'id': 'doc5',
            'text': 'The French Revolution began in 1789 in Paris.',
            'entities': ['French Revolution', 'Paris']
        }
    ]
    
    # Initialize agent with Qwen2.5-7B
    # Use device="cpu" if no GPU available
    agent = InteractiveRAGAgent(documents, device="cuda")
    
    # Query
    result = agent.query("What is the capital of France and what famous landmark is there?")
    
    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(f"Answer: {result['answer']}")
    print(f"\nTotal documents retrieved: {len(result['retrieved_docs'])}")
    print(f"Steps taken: {len(result['history'])}")
