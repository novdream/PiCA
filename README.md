
# PiCA: Pivot-Based Credit Assignment For Search Agentic Reinforcement Learning

This is the **official implementation** of the project **"PiCA: Pivot-Based Credit Assignment For Search Agentic Reinforcement Learning"**.

---

## 💡 Overview
**PiCA** (Pivot-Based Credit Assignment) is a step-reward mechanism for RL-trained LLM search agents. It reformulates search trajectories as a sequential process of cumulative information gain.


### Key Contributions
*   **Pivot-Annotated Dataset**: A large-scale dataset of 60k trajectories (based on MuSiQue) with precise annotations of critical "pivot" steps.
*   **Novel Credit Assignment**: A framework where rewards are determined by information peaks relative to the entire search history, solving the issues of reward sparsity and isolated credit.
*   **State-of-the-Art Results**: Consistent improvements across 7 multi-hop QA benchmarks, achieving a 15.2% performance boost for 3B models and 2.2% for 7B models.
---

## 📂 Project Structure

```text
Search-R1/
├── OpenRLHF/               # Customized OpenRLHF for PiCA reward model training
├── Search-R1/               
│   ├── search_r1/          # search agent tools 
│   └── verl/               # PiCA reward function + policy training
```

---

## Installation

**Requirements:**
* Python 3.10+
* CUDA 12.1+

```bash
conda create -n pica python=3.11
conda activate pica
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```
## 🚀 Quick Start
**Launch a local retrieval server.**
```bash
cd Search-R1
bash retrieval_launch.sh
```
**Train PiCA reward model with qwen2.5-3b-instruct.**
```bash
conda activate pica
cd OpenRLHF
bash examples/scripts/train_prm_mistral.sh
```
**Launch PiCA reward model.**
```bash
cd OpenRLHF
bash example/script/serve_remote_rm.sh
```
**Run RL training (PPO) with qwen2.5-7b-instruct.**
```bash
conda activate pica
cd Search-R1
bash train_ppo.sh
```
---

## 📊 Evaluation

Evaluation scripts for assessing search capabilities and multi-hop reasoning performance are provided in the `Search-R1/` directory.
```bash
bash Search-R1/eval.sh 
```

---

## 🤗 Models & Checkpoints

We provide checkpoints optimized for search tasks:
* **[PiCA-Qwen2.5-3B](https://huggingface.co/NeurIPS2026PiCA/PiCA-Qwen2.5-3B)**: Base policy model optimized for general search.
* **[PiCA-Qwen2.5-7B](https://huggingface.co/NeurIPS2026PiCA/PiCA-Qwen2.5-7B)**: Base policy model optimized for general search.
* **[PiCA-RM-Qwen2.5-3B](https://huggingface.co/NeurIPS2026PiCA/PiCA-RM-Qwen2.5-3B)**: The step-level reward model utilizing Pivot-Based Credit Assignment.

---
