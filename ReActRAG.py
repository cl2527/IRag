import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json

# -----------------------------
# 1. Vector Store (FAISS + real embeddings)
# -----------------------------
class VectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embs = self.model.encode(docs, convert_to_numpy=True)

        dim = self.embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embs)

    def search(self, query, k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        scores, idx = self.index.search(q_emb, k)
        results = [(self.docs[i], float(scores[0][j])) for j, i in enumerate(idx[0])]
        return results


# -----------------------------
# 2. LLM Policy (OpenAI-style)
# -----------------------------=
client = OpenAI()

def llm_policy(history):
    prompt = format_history(history)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    """
        response = {
        "choices": [
            {
                "message": {
                    "content": '{"action": "search", "query": "capital France"}'
                }
            }
        ]
    }
    """
    return parse_action(response.choices[0].message.content) # → '{"action": "search", "query": "capital France"}'


# -----------------------------
# 3. History Formatting + Action Parsing
# -----------------------------
def format_history(history):
    text = ""
    for h in history:
        if h["type"] == "question":
            text += f"Question: {h['content']}\n"
        elif h["type"] == "thought":
            text += f"Thought: {h['content']}\n"
        elif h["type"] == "action":
            text += f"Action: {h['content']}\n"
        elif h["type"] == "info":
            text += f"Observation:\n{h['content']}\n"

    text += """
Decide the next action as JSON.

Use one of:
{"action": "search", "query": "..."}
{"action": "answer", "output": "..."}
"""
    return text


def parse_action(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])


# -----------------------------
# 4. Sequential RAG Agent Loop (Section 2.2)
# -----------------------------
def rag_agent(query, store):
    history = [{"type": "question", "content": query}]

    for step in range(6):
        action = llm_policy(history)

        if action["action"] == "search":
            q = action["query"]
            history.append({"type": "thought", "content": f"I should search for: {q}"})
            
            results = store.search(q)
            info_text = "\n".join([f"- {doc}" for doc, _ in results])
            history.append({"type": "info", "content": info_text})

            print(f"[search] {q}")
            print("[info]\n", info_text)

        elif action["action"] == "answer":
            answer = action["output"]
            history.append({"type": "thought", "content": "I can answer now."})
            history.append({"type": "action", "content": answer})

            print("[answer]", answer)
            return answer

    return "Failed to answer."


# -----------------------------
# 5. Example Usage
# -----------------------------
if __name__ == "__main__":
    docs = [
        "Paris is the capital of France.",
        "France is located in Western Europe.",
        "The Eiffel Tower is in Paris.",
        "Berlin is the capital of Germany."
    ]

    store = VectorStore(docs)

    rag_agent("What is the capital of France?", store)
