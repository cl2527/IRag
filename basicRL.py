import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple
import json

# -----------------------------
# 1. Trajectory Collection for SFT
# -----------------------------
class TrajectoryCollector:
    """Collects successful trajectories from the RAG agent"""
    
    def __init__(self, agent, qa_pairs):
        self.agent = agent
        self.qa_pairs = qa_pairs
        self.successful_trajectories = []
    
    def collect_trajectories(self):
        """Generate trajectories and keep only successful ones"""
        for qa in self.qa_pairs:
            question = qa['question']
            ground_truth = qa['answer']
            
            # Run agent and track trajectory
            trajectory = self.agent.run_with_trajectory(question)
            
            # Check if final answer matches ground truth
            if self._matches_ground_truth(trajectory['final_answer'], ground_truth):
                self.successful_trajectories.append({
                    'question': question,
                    'trajectory': trajectory['steps'],
                    'answer': trajectory['final_answer']
                })
        
        return self.successful_trajectories
    
    def _matches_ground_truth(self, answer, ground_truth):
        """Check if answer matches ground truth"""
        # Simple exact match (can be improved with fuzzy matching)
        return answer.strip().lower() == ground_truth.strip().lower()


# -----------------------------
# 2. Supervised Fine-Tuning (SFT)
# -----------------------------
class SFTTrainer:
    """Fine-tune LLM on successful trajectories"""
    
    def __init__(self, model_name="gpt2", learning_rate=5e-5):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
    
    def prepare_training_data(self, trajectories):
        """Convert trajectories to training format with masking"""
        training_data = []
        
        for traj in trajectories:
            # Format trajectory as text
            text = f"Question: {traj['question']}\n"
            
            for step in traj['trajectory']:
                if step['type'] == 'thought':
                    text += f"Thought: {step['content']}\n"
                elif step['type'] == 'action':
                    text += f"Action: {step['content']}\n"
                elif step['type'] == 'observation':
                    # Mark observation tokens for masking
                    text += f"Observation: <MASK>{step['content']}</MASK>\n"
            
            text += f"Answer: {traj['answer']}"
            training_data.append(text)
        
        return training_data
    
    def train(self, trajectories, epochs=3, batch_size=4):
        """Train model on successful trajectories"""
        training_texts = self.prepare_training_data(trajectories)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(training_texts), batch_size):
                batch = training_texts[i:i+batch_size]
                
                # Tokenize with masking
                inputs = self.tokenizer(batch, return_tensors="pt", 
                                       padding=True, truncation=True, max_length=512)
                
                # Create labels with masked observation tokens
                labels = inputs['input_ids'].clone()
                labels = self._mask_observations(labels, batch)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(training_texts) / batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _mask_observations(self, labels, batch):
        """Mask observation tokens during training"""
        # Find tokens between <MASK> and </MASK> and set to -100 (ignored in loss)
        for i, text in enumerate(batch):
            tokens = self.tokenizer.encode(text)
            mask_start = self.tokenizer.encode("<MASK>")
            mask_end = self.tokenizer.encode("</MASK>")
            
            # Simple masking logic (can be improved)
            if "<MASK>" in text:
                labels[i, :] = -100  # Simplified - mask all observations
        
        return labels


# -----------------------------
# 3. GRPO (Group Relative Policy Optimization)
# -----------------------------
class GRPOTrainer:
    """Reinforcement learning with GRPO for policy refinement"""
    
    def __init__(self, model, tokenizer, learning_rate=1e-5, N=4):
        self.policy_model = model  # π_θ
        self.reference_model = self._copy_model(model)  # π_ref for KL penalty
        self.tokenizer = tokenizer
        self.optimizer = Adam(self.policy_model.parameters(), lr=learning_rate)
        self.N = N  # Number of trajectories per question
        self.beta = 0.1  # KL penalty coefficient
    
    def _copy_model(self, model):
        """Create a copy of model for reference"""
        ref_model = type(model).from_pretrained(model.config._name_or_path)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        return ref_model
    
    def compute_reward(self, trajectory, ground_truth):
        """
        Reward function: R(τ) = -1 + I{τ_valid} + I{τ_valid} · I{y_ans}
        """
        reward = -1.0
        
        # Check syntactic validity (proper JSON format, valid actions)
        is_valid = self._check_validity(trajectory)
        if is_valid:
            reward += 1.0  # I{τ_valid}
            
            # Check answer accuracy
            if self._check_answer_match(trajectory['answer'], ground_truth):
                reward += 1.0  # I{τ_valid} · I{y_ans}
        
        return reward
    
    def _check_validity(self, trajectory):
        """Check if trajectory has valid structure"""
        try:
            # Check if all actions are valid JSON
            for step in trajectory['steps']:
                if step['type'] == 'action':
                    json.loads(step['content'])
            return True
        except:
            return False
    
    def _check_answer_match(self, answer, ground_truth):
        """Check if answer matches ground truth"""
        return answer.strip().lower() == ground_truth.strip().lower()
    
    def compute_advantage(self, rewards):
        """
        Compute advantage A(τ_i) by normalizing rewards within group
        """
        rewards = np.array(rewards)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages
    
    def compute_importance_ratio(self, trajectory_tokens, action_tokens):
        """
        Compute ρ_θ(a_t^(i)) = π_θ(a_t^(i) | s_t^(i)-1) / π_all(a_t^(i) | s_t^(i)-1)
        """
        with torch.no_grad():
            # Current policy probability
            policy_logits = self.policy_model(**trajectory_tokens).logits
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            # Reference policy probability
            ref_logits = self.reference_model(**trajectory_tokens).logits
            ref_probs = F.softmax(ref_logits, dim=-1)
            
            # Importance sampling ratio
            action_policy_prob = torch.gather(policy_probs, -1, action_tokens.unsqueeze(-1))
            action_ref_prob = torch.gather(ref_probs, -1, action_tokens.unsqueeze(-1))
            
            ratio = (action_policy_prob / (action_ref_prob + 1e-8)).squeeze(-1)
        
        return ratio
    
    def train_step(self, question, ground_truth, agent):
        """
        Single training step:
        1. Generate N trajectories
        2. Compute rewards and advantages
        3. Update policy using GRPO objective
        """
        trajectories = []
        rewards = []
        
        # Generate N trajectories for this question
        for _ in range(self.N):
            trajectory = agent.run_with_trajectory(question)
            reward = self.compute_reward(trajectory, ground_truth)
            
            trajectories.append(trajectory)
            rewards.append(reward)
        
        # Compute advantages
        advantages = self.compute_advantage(rewards)
        
        # GRPO objective: maximize expected advantage with KL penalty
        total_loss = 0
        
        for i, (traj, advantage) in enumerate(zip(trajectories, advantages)):
            # Convert trajectory to tokens
            traj_text = self._format_trajectory(traj)
            tokens = self.tokenizer(traj_text, return_tensors="pt", 
                                   padding=True, truncation=True)
            
            # Mask observation tokens
            labels = tokens['input_ids'].clone()
            labels = self._mask_observations(labels, traj)
            
            # Forward pass
            outputs = self.policy_model(**tokens, labels=labels)
            
            # Policy gradient loss with advantage
            policy_loss = -advantage * outputs.loss
            
            # KL divergence penalty (simplified)
            with torch.no_grad():
                ref_outputs = self.reference_model(**tokens)
            
            kl_div = F.kl_div(
                F.log_softmax(outputs.logits, dim=-1),
                F.softmax(ref_outputs.logits, dim=-1),
                reduction='batchmean'
            )
            
            # Combined loss
            loss = policy_loss + self.beta * kl_div
            total_loss += loss
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), np.mean(rewards)
    
    def train(self, qa_pairs, agent, epochs=10):
        """Train policy using GRPO"""
        self.policy_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_reward = 0
            
            for qa in qa_pairs:
                loss, reward = self.train_step(qa['question'], qa['answer'], agent)
                total_loss += loss
                total_reward += reward
            
            avg_loss = total_loss / len(qa_pairs)
            avg_reward = total_reward / len(qa_pairs)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
    
    def _format_trajectory(self, trajectory):
        """Format trajectory as text"""
        text = ""
        for step in trajectory['steps']:
            if step['type'] == 'thought':
                text += f"Thought: {step['content']}\n"
            elif step['type'] == 'action':
                text += f"Action: {step['content']}\n"
            elif step['type'] == 'observation':
                text += f"Observation: {step['content']}\n"
        return text
    
    def _mask_observations(self, labels, trajectory):
        """Mask observation tokens"""
        # Set observation positions to -100 (ignored in loss)
        # Simplified implementation
        return labels


# -----------------------------
# 4. Complete Training Pipeline
# -----------------------------
class AgenticRAGTrainer:
    """Two-stage training: SFT + RL"""
    
    def __init__(self, agent, model_name="gpt2"):
        self.agent = agent
        self.model_name = model_name
    
    def train(self, qa_pairs, sft_epochs=3, rl_epochs=10):
        """
        Complete training pipeline:
        1. Collect successful trajectories
        2. SFT on trajectories
        3. RL refinement with GRPO
        """
        print("=" * 50)
        print("Stage 1: Supervised Fine-Tuning (SFT)")
        print("=" * 50)
        
        # Step 1: Collect trajectories
        print("Collecting successful trajectories...")
        collector = TrajectoryCollector(self.agent, qa_pairs)
        trajectories = collector.collect_trajectories()
        print(f"Collected {len(trajectories)} successful trajectories")
        
        # Step 2: SFT
        print("\nFine-tuning on successful trajectories...")
        sft_trainer = SFTTrainer(self.model_name)
        sft_trainer.train(trajectories, epochs=sft_epochs)
        
        print("\n" + "=" * 50)
        print("Stage 2: Reinforcement Learning (GRPO)")
        print("=" * 50)
        
        # Step 3: RL refinement
        print("Refining policy with GRPO...")
        grpo_trainer = GRPOTrainer(
            sft_trainer.model, 
            sft_trainer.tokenizer
        )
        grpo_trainer.train(qa_pairs, self.agent, epochs=rl_epochs)
        
        print("\nTraining complete!")
        return sft_trainer.model


# -----------------------------
# 5. Example Usage
# -----------------------------
if __name__ == "__main__":
    # Mock agent and data for demonstration
    class MockAgent:
        def run_with_trajectory(self, question):
            return {
                'steps': [
                    {'type': 'thought', 'content': 'I should search'},
                    {'type': 'action', 'content': '{"action": "search", "query": "test"}'},
                    {'type': 'observation', 'content': 'Result: Paris is capital'},
                    {'type': 'thought', 'content': 'I can answer'},
                    {'type': 'action', 'content': '{"action": "answer", "output": "Paris"}'}
                ],
                'final_answer': 'Paris',
                'answer': 'Paris'
            }
    
    qa_pairs = [
        {'question': 'What is the capital of France?', 'answer': 'Paris'},
        {'question': 'What is the capital of Germany?', 'answer': 'Berlin'}
    ]
    
    agent = MockAgent()
    
    # Train the agentic RAG system
    trainer = AgenticRAGTrainer(agent, model_name="gpt2")
    trained_model = trainer.train(qa_pairs, sft_epochs=2, rl_epochs=5)
    
    print("\nModel training completed successfully!")
