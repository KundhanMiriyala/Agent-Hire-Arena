# Agent Hire Arena — Solving Nightmare via Post-Training

This notebook demonstrates how to post-train a small language model to remain robust under deception, pressure, and resource constraints in the **Agent Hire Arena** environment.

Instead of optimizing for helpfulness, we train a model to:
- Resist deceptive signals (e.g., perfect but fake candidates)
- Allocate budget strategically for verification
- Ignore external pressure from adversarial agents

The result: a compact **Llama-3.2-1B** model that succeeds in scenarios where larger models fail — including the *Nightmare* difficulty setting.

---

## Overview

This notebook implements a full Supervised Fine-Tuning (SFT) pipeline for training a robust hiring agent.

Unlike standard SFT setups, the training data is filtered to prioritize high-quality decision trajectories under adversarial conditions. This enables the model to learn behaviors such as verification over blind trust, and reasoning under pressure.

### Pipeline Steps

1. **Environment setup** — installs Unsloth, vLLM, TRL, and supporting libraries on a Colab T4 GPU  
2. **Model loading** — loads `unsloth/Llama-3.2-1B-bnb-4bit` with LoRA adapters (rank 16, ~11M trainable parameters out of 1.25B total, ≈0.9%)  
3. **Data filtering** — retains only high-quality trajectories (default threshold: final_score > 0.70), yielding 248 train / 24 val pairs  
4. **Formatting** — converts `(messages, completion)` into Llama chat template format with EOS; training is applied only on completions (`completion_only_loss=True`)  
5. **SFT training** — 3 epochs using AdamW (8-bit), cosine LR schedule, and mixed precision  
6. **Evaluation** — runs the trained agent in the `HiringEnvironment` across difficulty levels with a safety guard ensuring valid actions  
7. **Export** — saves LoRA adapter + tokenizer and packages them for reuse  

---

## Agent Actions

The model learns to output exactly one structured JSON action per step:

| Action       | Cost       | Description |
|--------------|------------|-------------|
| `interview`  | 10 units   | Reveals a candidate’s interview score |
| `probe`      | 20 units   | Audits coaching risk (requires prior interview) |
| `hire`       | 50 units   | Adds candidate to the team |
| `skip`       | 0 units    | Permanently rejects a candidate |
| `finalize`   | —          | Ends the episode and scores the team |

---

## Training Configuration

| Parameter | Value |
|----------|------|
| Base model | `unsloth/Llama-3.2-1B-bnb-4bit` |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Max sequence length | 2048 |
| Batch size (effective) | 8 (2 × 4 accumulation) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Optimizer | AdamW (8-bit) |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Final training loss | **0.519** |

---

## Results Summary

| Difficulty   | Llama-1B (Post-Trained) |
|--------------|--------------------------|
| Easy         | 0.99                     |
| Medium       | 0.96                     |
| Hard         | 0.93                     |
| Adversarial  | 0.89                     |
| Nightmare    | 0.74                     |

The model maintains strong performance even under extreme adversarial conditions and successfully solves the *Nightmare* setting.

---

## Key Insight

Robust decision-making in adversarial environments does not emerge from model scale alone.

Despite being significantly smaller, the post-trained Llama-1B model learns to:
- Spend resources to verify information  
- Avoid deceptive high-confidence signals  
- Maintain strategy under pressure  

This demonstrates that robustness is a **post-training problem**, not a scaling problem.

---

## Data Format

Training data is stored as a JSONL file (`grpo_agenthire_train.jsonl`) where each entry represents a step within an episode:

```json
{
  "episode_id": "ep_001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": "{\"action\": \"interview\", \"candidate_id\": \"C03\"}",
  "done": false,
  "final_score": null
}