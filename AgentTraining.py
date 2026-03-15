import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# 1) Trajectory Sampling from InteractiveRAG
# -----------------------------

def normalize_text(text: str) -> str:
	return " ".join((text or "").strip().lower().split())


@dataclass
class Trajectory:
	question: str
	steps: List[Dict]
	final_answer: str


class InteractiveTrajectorySampler:
	"""
	Collect trajectories from InteractiveRAG agent.

	Expected agent API (from InteractiveRag.py):
	- agent.query(question, max_steps=...)
	  returns dict with keys: answer, history, retrieved_docs, plan
	"""

	def __init__(self, agent, max_steps: int = 6):
		self.agent = agent
		self.max_steps = max_steps

	def _convert_query_result(self, question: str, result: Dict) -> Trajectory:
		steps: List[Dict] = []

		for h in result.get("history", []):
			steps.append(
				{
					"type": "thought",
					"content": f"Directive={h.get('directive', 'unknown')}; {h.get('result_summary', '')}",
				}
			)
			for action in h.get("actions", []):
				steps.append({"type": "action", "content": json.dumps(action, ensure_ascii=False)})

		# Consolidated observations (retrieved content) for masking during SFT/RL.
		if result.get("retrieved_docs"):
			obs_text = "\n".join(
				[f"[{d.get('doc_id', 'unk')}] {d.get('text', '')}" for d in result["retrieved_docs"][:20]]
			)
			steps.append({"type": "observation", "content": obs_text})

		return Trajectory(
			question=question,
			steps=steps,
			final_answer=result.get("answer", ""),
		)

	def sample_one(self, question: str) -> Trajectory:
		result = self.agent.query(question, max_steps=self.max_steps)
		return self._convert_query_result(question, result)

	def sample_group(self, question: str, n: int) -> List[Trajectory]:
		return [self.sample_one(question) for _ in range(n)]


# -----------------------------
# 2) Formatting + Masking Helpers
# -----------------------------

OBS_START = "<tool_response>"
OBS_END = "</tool_response>"


def format_trajectory_for_training(traj: Trajectory) -> str:
	lines = [f"Question: {traj.question}"]
	for s in traj.steps:
		if s["type"] == "thought":
			lines.append(f"Thought: {s['content']}")
		elif s["type"] == "action":
			lines.append(f"<tool_call>{s['content']}</tool_call>")
		elif s["type"] == "observation":
			lines.append(f"{OBS_START}{s['content']}{OBS_END}")
	lines.append(f"Final Answer: {traj.final_answer}")
	return "\n".join(lines)


def build_ids_and_labels(tokenizer, text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Build autoregressive labels, masking retrieved-info tokens inside
	<tool_response> ... </tool_response> with -100.
	"""
	input_ids: List[int] = []
	labels: List[int] = []

	cursor = 0
	while cursor < len(text):
		s = text.find(OBS_START, cursor)
		if s == -1:
			chunk = text[cursor:]
			ids = tokenizer.encode(chunk, add_special_tokens=False)
			input_ids.extend(ids)
			labels.extend(ids)
			break

		# normal text before observation => supervised
		prefix = text[cursor:s]
		ids = tokenizer.encode(prefix, add_special_tokens=False)
		input_ids.extend(ids)
		labels.extend(ids)

		e = text.find(OBS_END, s)
		if e == -1:
			e = len(text)
			obs_chunk = text[s:e]
		else:
			e = e + len(OBS_END)
			obs_chunk = text[s:e]

		obs_ids = tokenizer.encode(obs_chunk, add_special_tokens=False)
		input_ids.extend(obs_ids)
		labels.extend([-100] * len(obs_ids))
		cursor = e

	if tokenizer.eos_token_id is not None:
		input_ids.append(tokenizer.eos_token_id)
		labels.append(tokenizer.eos_token_id)

	return (
		torch.tensor(input_ids, dtype=torch.long, device=device),
		torch.tensor(labels, dtype=torch.long, device=device),
	)


def pad_batch(input_tensors: List[torch.Tensor], label_tensors: List[torch.Tensor], pad_id: int):
	max_len = max(t.size(0) for t in input_tensors)
	bsz = len(input_tensors)

	input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=input_tensors[0].device)
	labels = torch.full((bsz, max_len), -100, dtype=torch.long, device=input_tensors[0].device)
	attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=input_tensors[0].device)

	for i, (inp, lab) in enumerate(zip(input_tensors, label_tensors)):
		L = inp.size(0)
		input_ids[i, :L] = inp
		labels[i, :L] = lab
		attention_mask[i, :L] = 1

	return input_ids, labels, attention_mask


# -----------------------------
# 3) Stage-1 SFT
# -----------------------------

class SFTTrainer:
	def __init__(
		self,
		model_name: str = "Qwen/Qwen2.5-7B-Instruct",
		lr: float = 2e-5,
		device: Optional[str] = None,
	):
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
			trust_remote_code=True,
		).to(self.device)

		self.optimizer = AdamW(self.model.parameters(), lr=lr)

	def collect_successful_trajectories(self, sampler: InteractiveTrajectorySampler, qa_pairs: List[Dict]) -> List[Trajectory]:
		successful = []
		for qa in qa_pairs:
			traj = sampler.sample_one(qa["question"])
			if normalize_text(traj.final_answer) == normalize_text(qa["answer"]):
				successful.append(traj)
		return successful

	def train(self, trajectories: List[Trajectory], epochs: int = 2, batch_size: int = 1):
		if not trajectories:
			print("No successful trajectories found for SFT.")
			return

		self.model.train()
		texts = [format_trajectory_for_training(t) for t in trajectories]

		for ep in range(epochs):
			total_loss = 0.0
			for i in range(0, len(texts), batch_size):
				batch_texts = texts[i : i + batch_size]

				ins, labs = [], []
				for txt in batch_texts:
					inp, lab = build_ids_and_labels(self.tokenizer, txt, self.device)
					ins.append(inp)
					labs.append(lab)

				input_ids, labels, attention_mask = pad_batch(ins, labs, self.tokenizer.pad_token_id)

				out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
				loss = out.loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_loss += float(loss.detach().cpu())

			print(f"[SFT] epoch={ep+1}/{epochs}, loss={total_loss / max(1, (len(texts) // batch_size + (len(texts)%batch_size>0))):.4f}")


# -----------------------------
# 4) Stage-2 RL with GRPO
# -----------------------------

class GRPOTrainer:
	def __init__(
		self,
		model,
		tokenizer,
		lr: float = 5e-6,
		group_size: int = 4,
		clip_eps: float = 0.2,
		beta_kl: float = 0.02,
		device: Optional[str] = None,
	):
		self.model = model
		self.tokenizer = tokenizer
		self.group_size = group_size
		self.clip_eps = clip_eps
		self.beta_kl = beta_kl
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.optimizer = AdamW(self.model.parameters(), lr=lr)

		self.reference_model = copy.deepcopy(self.model).eval()

	def _trajectory_validity(self, traj: Trajectory) -> bool:
		"""Syntactic validity: reasoning + action/tool-call structure."""
		if not traj.steps:
			return False
		try:
			for s in traj.steps:
				if s["type"] not in {"thought", "action", "observation"}:
					return False
				if s["type"] == "action":
					parsed = json.loads(s["content"])
					if "action" not in parsed:
						return False
			return True
		except Exception:
			return False

	def reward(self, traj: Trajectory, ground_truth: str) -> float:
		"""R(τ) = -1 + I{τ_valid} + I{τ_valid} * I{y_ans}"""
		is_valid = self._trajectory_validity(traj)
		is_correct = normalize_text(traj.final_answer) == normalize_text(ground_truth)
		return -1.0 + float(is_valid) + float(is_valid and is_correct)

	@staticmethod
	def advantages_from_rewards(rewards: List[float]) -> np.ndarray:
		r = np.array(rewards, dtype=np.float32)
		return (r - r.mean()) / (r.std() + 1e-8)

	def _token_logprobs(self, model, input_ids, attention_mask):
		out = model(input_ids=input_ids, attention_mask=attention_mask)
		logits = out.logits[:, :-1, :]
		tgt = input_ids[:, 1:]
		logp = F.log_softmax(logits, dim=-1)
		tok_logp = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
		return tok_logp, logits

	def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor):
		denom = mask.sum().clamp_min(1.0)
		return (x * mask).sum() / denom

	def train(self, sampler: InteractiveTrajectorySampler, qa_pairs: List[Dict], epochs: int = 1):
		self.model.train()

		for ep in range(epochs):
			ep_loss = 0.0
			ep_reward = 0.0

			old_policy = copy.deepcopy(self.model).eval()

			for qa in qa_pairs:
				group = sampler.sample_group(qa["question"], self.group_size)
				rewards = [self.reward(t, qa["answer"]) for t in group]
				adv = self.advantages_from_rewards(rewards)

				loss_terms = []
				for idx, traj in enumerate(group):
					text = format_trajectory_for_training(traj)
					inp, lab = build_ids_and_labels(self.tokenizer, text, self.device)
					input_ids, labels, attn = pad_batch([inp], [lab], self.tokenizer.pad_token_id)

					# Mask for RL objective: keep non-observation tokens only.
					valid_mask = (labels[:, 1:] != -100).float()

					new_tok_logp, new_logits = self._token_logprobs(self.model, input_ids, attn)
					with torch.no_grad():
						old_tok_logp, _ = self._token_logprobs(old_policy, input_ids, attn)
						ref_tok_logp, ref_logits = self._token_logprobs(self.reference_model, input_ids, attn)

					# ρ_t = exp(log πθ - log πold)
					ratio_t = torch.exp(new_tok_logp - old_tok_logp)
					unclipped = ratio_t * float(adv[idx])
					clipped = torch.clamp(ratio_t, 1 - self.clip_eps, 1 + self.clip_eps) * float(adv[idx])
					ppo_obj = torch.minimum(unclipped, clipped)
					ppo_obj = self._masked_mean(ppo_obj, valid_mask)

					# KL(πθ || πref)
					new_log_probs = F.log_softmax(new_logits, dim=-1)
					ref_probs = F.softmax(ref_logits, dim=-1)
					kl_per_tok = F.kl_div(new_log_probs, ref_probs, reduction="none").sum(dim=-1)
					kl = self._masked_mean(kl_per_tok, valid_mask)

					# maximize objective => minimize negative
					loss_i = -(ppo_obj - self.beta_kl * kl)
					loss_terms.append(loss_i)

				loss = torch.stack(loss_terms).mean()
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				ep_loss += float(loss.detach().cpu())
				ep_reward += float(np.mean(rewards))

			n = max(1, len(qa_pairs))
			print(f"[GRPO] epoch={ep+1}/{epochs}, loss={ep_loss / n:.4f}, avg_reward={ep_reward / n:.4f}")


# -----------------------------
# 5) Full End-to-End Trainer
# -----------------------------

class EndToEndAgentTrainer:
	"""
	Two-stage training:
	1) SFT from successful trajectories
	2) GRPO policy refinement
	"""

	def __init__(self, agent, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: Optional[str] = None):
		self.agent = agent
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.sampler = InteractiveTrajectorySampler(agent)
		self.sft = SFTTrainer(model_name=model_name, device=self.device)

	def train(self, qa_pairs: List[Dict], sft_epochs: int = 2, rl_epochs: int = 1, group_size: int = 4):
		print("=" * 60)
		print("Stage 1: Trace Sampling + SFT")
		print("=" * 60)
		successful = self.sft.collect_successful_trajectories(self.sampler, qa_pairs)
		print(f"successful trajectories: {len(successful)} / {len(qa_pairs)}")
		self.sft.train(successful, epochs=sft_epochs, batch_size=1)

		print("\n" + "=" * 60)
		print("Stage 2: GRPO Reinforcement Learning")
		print("=" * 60)
		grpo = GRPOTrainer(
			model=self.sft.model,
			tokenizer=self.sft.tokenizer,
			group_size=group_size,
			device=self.device,
		)
		grpo.train(self.sampler, qa_pairs, epochs=rl_epochs)

		return self.sft.model, self.sft.tokenizer


# -----------------------------
# 6) Minimal Usage Example
# -----------------------------

if __name__ == "__main__":
	# Expected usage with your InteractiveRAGAgent from InteractiveRag.py
	# from InteractiveRag import InteractiveRAGAgent
	# agent = InteractiveRAGAgent(documents, device="cuda")
	#
	# qa_pairs = [
	#     {"question": "What is the capital of France?", "answer": "Paris"},
	#     {"question": "What city has the Eiffel Tower?", "answer": "Paris"},
	# ]
	#
	# trainer = EndToEndAgentTrainer(agent, model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda")
	# model, tokenizer = trainer.train(qa_pairs, sft_epochs=1, rl_epochs=1, group_size=4)
	pass
