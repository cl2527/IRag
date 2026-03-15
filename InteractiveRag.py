import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
	import chromadb
except Exception:
	chromadb = None


# ============================================================
# 0) LLM interface
# ============================================================

class LLMBackend(Protocol):
	def generate(self, prompt: str) -> str:
		...


class OpenAIBackend:
	"""Minimal backend wrapper for OpenAI chat-completions."""

	def __init__(self, model: str = "qwen3-8b"):
		from openai import OpenAI

		self.client = OpenAI()
		self.model = model

	def generate(self, prompt: str) -> str:
		resp = self.client.chat.completions.create(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			temperature=0.2,
		)
		return resp.choices[0].message.content or ""


class QwenLocalBackend:
	"""
	Local backend for Qwen models.
	Default follows paper setting: Qwen3-8B.
	"""

	def __init__(
		self,
		model_name: str = "Qwen/Qwen3-8B",
		device: Optional[str] = None,
		max_new_tokens: int = 512,
		temperature: float = 0.2,
	):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.max_new_tokens = max_new_tokens
		self.temperature = temperature

		self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
			device_map="auto" if self.device == "cuda" else None,
			trust_remote_code=True,
		)
		if self.device == "cpu":
			self.model = self.model.to(self.device)

	def generate(self, prompt: str) -> str:
		messages = [{"role": "user", "content": prompt}]
		text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=self.max_new_tokens,
				temperature=self.temperature,
				do_sample=self.temperature > 0,
				top_p=0.9,
				pad_token_id=self.tokenizer.eos_token_id,
			)

		new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
		return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 1) Corpus Interaction Engine
# ============================================================

@dataclass
class RetrievedDoc:
	doc_id: str
	text: str
	semantic_score: float = 0.0
	exact_score: float = 0.0
	fused_score: float = 0.0


class CorpusInteractionEngine:
	"""
	Implements action space A_CI:
	- semantic_search(query)
	- exact_search(keywords)
	- weighted_fusion(ws, we)
	- entity_match(entity)
	- include_docs(doc_ids)
	- exclude_docs(doc_ids)
	- adjust_scale(n)

	Uses:
	- e5-base-v2 embeddings + ChromaDB for semantic retrieval
	- SQLite FTS5 BM25 for exact/entity matching

	Deterministic concurrent-action pipeline:
	1) Build fused candidate list from semantic+exact (top-20 each, min-max normalized)
	2) Apply include/exclude filters
	3) Select top chunks by budget
	4) Append up to 3 unique entity-specific snippets
	"""

	def __init__(self, documents: List[Dict[str, Any]], embedding_model: str = "intfloat/e5-base-v2"):
		self.documents = documents
		self.doc_by_id = {d["id"]: d for d in documents}

		self.include_set = set()
		self.exclude_set = set()
		self.scale = 3
		self.ws = 0.5
		self.we = 0.5
		self.fusion_pool_size = 20

		# Dense index (e5-base-v2)
		self.embedder = SentenceTransformer(embedding_model)
		self.doc_texts = [d["text"] for d in documents]
		e5_docs = [f"passage: {t}" for t in self.doc_texts]
		self.doc_emb = self.embedder.encode(e5_docs, convert_to_numpy=True, normalize_embeddings=True)

		# ChromaDB backing store (lightweight; fallback to in-memory scoring if unavailable)
		self.chroma_client = None
		self.chroma_collection = None
		if chromadb is not None:
			self.chroma_client = chromadb.Client()
			self.chroma_collection = self.chroma_client.create_collection(name="interact_rag_chunks")
			self.chroma_collection.add(
				ids=[d["id"] for d in documents],
				documents=self.doc_texts,
				embeddings=self.doc_emb.tolist(),
			)

		# Sparse index (FTS)
		self.conn = sqlite3.connect(":memory:")
		self._init_fts()

	def _init_fts(self):
		cur = self.conn.cursor()
		cur.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(doc_id, text)")
		for d in self.documents:
			cur.execute("INSERT INTO docs_fts(doc_id, text) VALUES (?, ?)", (d["id"], d["text"]))
		self.conn.commit()

	def reset(self):
		self.include_set.clear()
		self.exclude_set.clear()
		self.scale = 3
		self.ws = 0.5
		self.we = 0.5

	def _allowed(self, doc_id: str) -> bool:
		return doc_id not in self.exclude_set

	@staticmethod
	def _minmax_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
		if not score_map:
			return {}
		vals = list(score_map.values())
		vmin, vmax = min(vals), max(vals)
		if abs(vmax - vmin) < 1e-12:
			return {k: 1.0 for k in score_map}
		return {k: (v - vmin) / (vmax - vmin) for k, v in score_map.items()}

	@staticmethod
	def _normalize_words(text: str) -> List[str]:
		return re.findall(r"[a-z0-9]+", (text or "").lower())

	@staticmethod
	def _split_sentences(text: str) -> List[str]:
		parts = re.split(r"(?<=[.!?])\s+", text.strip())
		return [p.strip() for p in parts if p.strip()]

	# ---------- Multi-faceted retrieval ----------
	def semantic_search(self, query: str, k: Optional[int] = None) -> List[RetrievedDoc]:
		k = k or self.fusion_pool_size
		q_vec = self.embedder.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)[0]

		# Prefer ChromaDB query path
		if self.chroma_collection is not None:
			res = self.chroma_collection.query(query_embeddings=[q_vec.tolist()], n_results=min(k, len(self.documents)))
			ids = res.get("ids", [[]])[0]
			distances = res.get("distances", [[]])[0]
			out = []
			for doc_id, dist in zip(ids, distances):
				doc = self.doc_by_id[doc_id]
				# convert cosine distance-like output to score where higher is better
				sem_score = float(1.0 - dist)
				out.append(RetrievedDoc(doc_id=doc_id, text=doc["text"], semantic_score=sem_score))
			return out

		# Fallback
		sims = self.doc_emb @ q_vec
		idx = np.argsort(-sims)[:k]
		return [
			RetrievedDoc(doc_id=self.documents[i]["id"], text=self.documents[i]["text"], semantic_score=float(sims[i]))
			for i in idx
		]

	def exact_search(self, keywords: str, k: Optional[int] = None) -> List[RetrievedDoc]:
		k = k or self.fusion_pool_size
		cur = self.conn.cursor()
		cur.execute(
			"""
			SELECT doc_id, bm25(docs_fts) AS score
			FROM docs_fts
			WHERE docs_fts MATCH ?
			ORDER BY score ASC
			LIMIT ?
			""",
			(keywords, min(k, len(self.documents))),
		)

		out: List[RetrievedDoc] = []
		for doc_id, score in cur.fetchall():
			doc = self.doc_by_id[doc_id]
			out.append(
				RetrievedDoc(
					doc_id=doc_id,
					text=doc["text"],
					exact_score=float(-score),
				)
			)
		return out

	def weighted_fusion(self, ws: float, we: float):
		total = max(ws + we, 1e-8)
		self.ws = ws / total
		self.we = we / total

	# ---------- Anchored matching ----------
	def entity_match(self, entity: str, sub_query: str = "", k: int = 3) -> List[Dict[str, Any]]:
		"""
		Retrieve entity-containing snippets and append 3 most relevant short sentences.
		"""
		token = entity.strip().replace('"', "")
		if not token:
			return []

		# FTS phrase retrieval for docs containing entity terms.
		cur = self.conn.cursor()
		phrase_query = f'"{token}"'
		cur.execute(
			"""
			SELECT doc_id, bm25(docs_fts) AS score
			FROM docs_fts
			WHERE docs_fts MATCH ?
			ORDER BY score ASC
			LIMIT 30
			""",
			(phrase_query,),
		)

		query_words = set(self._normalize_words(sub_query))
		entity_words = set(self._normalize_words(token))
		cand: List[Tuple[float, Dict[str, Any]]] = []

		for doc_id, bm25_score in cur.fetchall():
			if not self._allowed(doc_id):
				continue
			text = self.doc_by_id[doc_id]["text"]
			for sent in self._split_sentences(text):
				sw = set(self._normalize_words(sent))
				if not entity_words.issubset(sw) and token.lower() not in sent.lower():
					continue
				overlap = len(sw & query_words) if query_words else 0
				score = overlap + float(-bm25_score)
				cand.append(
					(
						score,
						{
							"doc_id": doc_id,
							"snippet": sent,
							"entity": token,
							"snippet_score": float(score),
						},
					)
				)

		cand.sort(key=lambda x: x[0], reverse=True)
		unique_snippets = []
		seen = set()
		for _, item in cand:
			key = (item["doc_id"], item["snippet"].lower())
			if key in seen:
				continue
			seen.add(key)
			unique_snippets.append(item)
			if len(unique_snippets) >= max(1, k):
				break

		return unique_snippets

	# ---------- Context shaping ----------
	def include_docs(self, doc_ids: List[str]):
		self.include_set.update([d for d in doc_ids if d in self.doc_by_id])

	def exclude_docs(self, doc_ids: List[str]):
		self.exclude_set.update(doc_ids)

	def adjust_scale(self, n: int):
		self.scale = max(1, int(n))

	# ---------- Consolidated execution ----------
	def execute_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""
		Executes concurrent actions in one iteration and returns ONE consolidated context.
		"""
		sem_results: List[RetrievedDoc] = []
		ex_results: List[RetrievedDoc] = []
		entity_requests: List[Dict[str, str]] = []
		executed: List[str] = []

		# Run state-changing actions first where needed
		for a in actions:
			name = a.get("name", "")
			args = a.get("args", {})
			if name == "weighted_fusion":
				self.weighted_fusion(float(args.get("ws", self.ws)), float(args.get("we", self.we)))
				executed.append(name)
			elif name == "include_docs":
				self.include_docs(args.get("doc_ids", []))
				executed.append(name)
			elif name == "exclude_docs":
				self.exclude_docs(args.get("doc_ids", []))
				executed.append(name)
			elif name == "adjust_scale":
				self.adjust_scale(int(args.get("n", self.scale)))
				executed.append(name)

		# Retrieval actions
		for a in actions:
			name = a.get("name", "")
			args = a.get("args", {})
			if name == "semantic_search":
				sem_results.extend(self.semantic_search(args.get("query", ""), self.fusion_pool_size))
				executed.append(name)
			elif name == "exact_search":
				ex_results.extend(self.exact_search(args.get("keywords", ""), self.fusion_pool_size))
				executed.append(name)
			elif name == "entity_match":
				entity_requests.append(
					{
						"entity": args.get("entity", ""),
						"sub_query": args.get("sub_query", args.get("query", "")),
					}
				)
				executed.append(name)

		# 1) Fuse semantic+exact pools (top-20 each, min-max normalized).
		sem_map: Dict[str, float] = {}
		ex_map: Dict[str, float] = {}
		for d in sem_results[: self.fusion_pool_size]:
			sem_map[d.doc_id] = max(sem_map.get(d.doc_id, -1e9), d.semantic_score)
		for d in ex_results[: self.fusion_pool_size]:
			ex_map[d.doc_id] = max(ex_map.get(d.doc_id, -1e9), d.exact_score)

		n_sem = self._minmax_normalize(sem_map)
		n_ex = self._minmax_normalize(ex_map)

		# Candidate list from fused retrieval
		merged: Dict[str, RetrievedDoc] = {}

		def upsert(doc: RetrievedDoc):
			if doc.doc_id not in merged:
				merged[doc.doc_id] = RetrievedDoc(doc_id=doc.doc_id, text=doc.text)
			m = merged[doc.doc_id]
			m.semantic_score = max(m.semantic_score, doc.semantic_score)
			m.exact_score = max(m.exact_score, doc.exact_score)

		for d in sem_results:
			upsert(d)
		for d in ex_results:
			upsert(d)

		for doc_id, d in merged.items():
			d.semantic_score = n_sem.get(doc_id, 0.0)
			d.exact_score = n_ex.get(doc_id, 0.0)
			d.fused_score = self.ws * d.semantic_score + self.we * d.exact_score

		# 2) Apply include/exclude filters (retain include, remove exclude)
		filtered = [d for d in merged.values() if self._allowed(d.doc_id)]

		included_docs = []
		for doc_id in sorted(self.include_set):
			if self._allowed(doc_id) and doc_id in self.doc_by_id:
				d = self.doc_by_id[doc_id]
				included_docs.append(
					RetrievedDoc(doc_id=doc_id, text=d["text"], semantic_score=0.0, exact_score=0.0, fused_score=1.1)
				)

		# Merge include docs without duplicates
		by_id = {d.doc_id: d for d in filtered}
		for d in included_docs:
			if d.doc_id in by_id:
				by_id[d.doc_id].fused_score = max(by_id[d.doc_id].fused_score, d.fused_score)
			else:
				by_id[d.doc_id] = d

		# 3) Top-ranked chunks by budget
		ranked = sorted(by_id.values(), key=lambda x: x.fused_score, reverse=True)[: self.scale]

		# 4) Append unique entity-specific short snippets (up to 3 total)
		snippets: List[Dict[str, Any]] = []
		for req in entity_requests:
			entity = req.get("entity", "")
			sub_q = req.get("sub_query", "")
			if entity:
				snippets.extend(self.entity_match(entity=entity, sub_query=sub_q, k=3))
		# unique and cap to 3
		seen_sn = set()
		uniq_snippets = []
		for s in snippets:
			key = (s["doc_id"], s["snippet"].lower())
			if key in seen_sn:
				continue
			seen_sn.add(key)
			uniq_snippets.append(s)
			if len(uniq_snippets) >= 3:
				break

		response = {
			"retrieved": [
				{
					"doc_id": d.doc_id,
					"text": d.text,
					"semantic_score": d.semantic_score,
					"exact_score": d.exact_score,
					"fused_score": d.fused_score,
				}
				for d in ranked
			],
			"entity_snippets": uniq_snippets,
			"metadata": {
				"actions_executed": executed,
				"fusion_pool_size": self.fusion_pool_size,
				"scale": self.scale,
				"ws": self.ws,
				"we": self.we,
				"included_docs": sorted(list(self.include_set)),
				"excluded_docs": sorted(list(self.exclude_set)),
			},
		}
		return response


# ============================================================
# 2) Reasoning-enhanced workflow modules
# ============================================================

def safe_json_parse(text: str) -> Dict[str, Any]:
	text = text.strip()
	try:
		return json.loads(text)
	except Exception:
		s = text.find("{")
		e = text.rfind("}")
		if s >= 0 and e >= s:
			return json.loads(text[s : e + 1])
		raise


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
	"""
	Parse repeated tool-call blocks.
	Supports both:
	- <tool_call>{...}</tool_call>
	- <tool call>{...}</tool call>
	JSON schema: {"name":"semantic_search", "args":{...}}
	"""
	blocks = re.findall(r"<tool[_ ]call>(.*?)</tool[_ ]call>", text, flags=re.DOTALL)
	calls = []
	for b in blocks:
		try:
			calls.append(safe_json_parse(b))
		except Exception:
			continue
	return calls


class GlobalPlanner:
	def __init__(self, llm: LLMBackend):
		self.llm = llm

	def run(self, query: str) -> Dict[str, Any]:
		prompt = f"""
You are an expert research assistant focused on high-level planning.
There is a search tool available to fetch information, but DO NOT execute it.
Your goal is only to produce a research plan.

Planning process:
1) Thoroughly analyze the user question. Identify key entities, concepts, and constraints.
2) If the question is straightforward, provide one comprehensive search direction.
3) If complex, break it into clear sub-tasks and desired outcomes for each step.
4) Mention possible parallelizable sub-tasks when appropriate.
5) Do not answer the question directly in this stage.
6) Do not rely on uncommon internal knowledge.

User query: {query}

Return STRICT JSON only:
{{
	"analysis": "Concise natural-language analysis with fluent connectors.",
	"steps": [
		"Step 1: ...",
		"Step 2: ..."
	]
}}
""".strip()
		return safe_json_parse(self.llm.generate(prompt))


class AdaptiveReasoner:
	def __init__(self, llm: LLMBackend):
		self.llm = llm

	def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
		prompt = f"""
You are an expert research strategist.
Analyze latest search state and decide the next best step.
Only generate the next-step plan; do not execute tools and do not directly answer unless task is fully complete.

Instruction:
- Briefly summarize key findings from previous retrieval.
- State what is still missing.
- Choose one path:
	A) proceed  -> current sub-task is progressing; suggest the next logical search.
	B) conclude -> overall task solved; research can end.
	C) refine   -> previous search is weak/noisy/incomplete; propose improved strategy.
- If a sub-task has failed repeatedly, allow moving to the next sub-task.
- Keep language concise and clear.

Current plan step index: {state['step_idx']}
Plan steps: {json.dumps(state['plan'].get('steps', []), ensure_ascii=False)}
Recent history: {json.dumps(state['history'][-4:], ensure_ascii=False)}
Retrieved docs count: {len(state['retrieved'])}

Return STRICT JSON only:
{{
	"directive": "proceed" | "refine" | "conclude",
	"thought": "concise natural-language analysis",
	"strategy": "specific next-step strategy"
}}
""".strip()
		return safe_json_parse(self.llm.generate(prompt))


class Executor:
	def __init__(self, llm: LLMBackend):
		self.llm = llm

	def run(self, query: str, reasoner_output: Dict[str, Any], state: Dict[str, Any]) -> str:
		"""
		Returns text containing:
		- <think>...</think>
		- one or multiple <tool_call>{json}</tool_call> (max 2)
		OR final answer block <final_answer>...</final_answer>
		"""
		prompt = f"""
You are a specialized searching execution agent.
Your only action choices are:
1) call search primitives (tool-calls), or
2) provide final answer.

Available primitives:
- semantic_search(query)              # required query parameter for semantic retrieval
- exact_search(keywords)
- weighted_fusion(ws,we)
- entity_match(entity, sub_query)
- include_docs(doc_ids)
- exclude_docs(doc_ids)
- adjust_scale(n)

Output contract:
- If evidence is sufficient, output ONLY <final_answer>...</final_answer>.
- Otherwise output exactly:
  (a) one <think>...</think>
  (b) 1 to 2 tool-call blocks
- Tool-call tags:
  preferred: <tool_call>{{...}}</tool_call>
  compatible: <tool call>{{...}}</tool call>
- Each call JSON schema:
  {{"name":"primitive_name", "args":{{...}}}}
- Keep arguments clear and specific.
- If no explicit semantic query was provided by strategy, formulate one.

Query: {query}
Directive: {reasoner_output.get('directive', 'refine')}
Reasoner analysis: {reasoner_output.get('thought', '')}
Reasoner strategy: {reasoner_output.get('strategy', '')}
Current context size: {len(state['retrieved'])}
""".strip()
		return self.llm.generate(prompt)

	def finalize(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
		context = "\n".join([f"[{d['doc_id']}] {d['text']}" for d in retrieved[:12]])
		prompt = f"""
Answer the query using context only when possible.
Query: {query}

Context:
{context}

Return concise answer.
""".strip()
		return self.llm.generate(prompt).strip()


# ============================================================
# 3) Interact-RAG Orchestrator
# ============================================================

class InteractiveRAGAgent:
	def __init__(self, engine: CorpusInteractionEngine, llm: LLMBackend):
		self.engine = engine
		self.global_planner = GlobalPlanner(llm)
		self.reasoner = AdaptiveReasoner(llm)
		self.executor = Executor(llm)

	@staticmethod
	def _parse_thought(executor_text: str) -> str:
		# Prefer Qwen-style think tag, but keep backward compatibility.
		m = re.search(r"<think>(.*?)</think>", executor_text, flags=re.DOTALL)
		if m:
			return m.group(1).strip()
		m = re.search(r"<thought>(.*?)</thought>", executor_text, flags=re.DOTALL)
		return m.group(1).strip() if m else ""

	@staticmethod
	def _parse_final(executor_text: str) -> Optional[str]:
		m = re.search(r"<final_answer>(.*?)</final_answer>", executor_text, flags=re.DOTALL)
		return m.group(1).strip() if m else None

	@staticmethod
	def _tool_response_tag(payload: Dict[str, Any]) -> str:
		# Canonical tag in this implementation.
		return f"<tool_response>{json.dumps(payload, ensure_ascii=False)}</tool_response>"

	def query(self, user_query: str, max_steps: int = 7) -> Dict[str, Any]:
		self.engine.reset()
		plan = self.global_planner.run(user_query)

		state = {
			"plan": plan,
			"step_idx": 0,
			"history": [],
			"retrieved": [],
		}

		for t in range(max_steps):
			reason = self.reasoner.run(state)
			directive = str(reason.get("directive", "")).lower().strip()
			if directive == "conclude":
				directive = "finish"
			reason["directive"] = directive

			if directive == "finish":
				break

			executor_text = self.executor.run(user_query, reason, state)
			thought = self._parse_thought(executor_text)
			maybe_final = self._parse_final(executor_text)

			if maybe_final:
				return {
					"answer": maybe_final,
					"plan": plan,
					"history": state["history"],
					"retrieved_docs": state["retrieved"],
				}

			actions = extract_tool_calls(executor_text)
			tool_payload = self.engine.execute_actions(actions)

			# Aggregate one consolidated context:
			# unified ranked chunks + concise entity snippets
			combined = list(tool_payload["retrieved"])
			for sn in tool_payload.get("entity_snippets", []):
				combined.append(
					{
						"doc_id": f"{sn['doc_id']}#snippet",
						"text": sn["snippet"],
						"semantic_score": 0.0,
						"exact_score": 0.0,
						"fused_score": sn.get("snippet_score", 0.0),
					}
				)
			state["retrieved"] = combined

			state["history"].append(
				{
					"t": t,
					"reasoner": reason,
					"thought": thought,
					"actions": actions,
					"tool_response_tag": self._tool_response_tag(tool_payload),
				}
			)

			# progress plan loosely
			if reason.get("directive") == "proceed":
				state["step_idx"] += 1

			# finish if last plan step likely done
			if state["step_idx"] >= len(plan.get("steps", [])):
				break

		final_answer = self.executor.finalize(user_query, state["retrieved"])
		return {
			"answer": final_answer,
			"plan": plan,
			"history": state["history"],
			"retrieved_docs": state["retrieved"],
		}


# ============================================================
# 4) Example
# ============================================================

if __name__ == "__main__":
	docs = [
		{"id": "d1", "text": "Paris is the capital of France."},
		{"id": "d2", "text": "The Eiffel Tower is in Paris."},
		{"id": "d3", "text": "Berlin is the capital of Germany."},
		{"id": "d4", "text": "France is in Western Europe."},
	]

	# Paper-aligned defaults:
	# - Retriever: e5-base-v2
	# - Top chunks: 3
	# - LLM: Qwen3-8B
	#
	# Option A: local model
	# llm = QwenLocalBackend(model_name="Qwen/Qwen3-8B")
	#
	# Option B: OpenAI-compatible endpoint serving Qwen3-8B
	llm = OpenAIBackend(model="qwen3-8b")

	engine = CorpusInteractionEngine(docs)
	agent = InteractiveRAGAgent(engine=engine, llm=llm)

	out = agent.query("What is the capital of France and which landmark is there?")
	print("Answer:", out["answer"])
	print("Plan:", json.dumps(out["plan"], indent=2, ensure_ascii=False))
	print("History steps:", len(out["history"]))
