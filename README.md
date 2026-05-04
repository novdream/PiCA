
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
## 📊 Dataset
The PiCA-MuSiQue dataset, consisting of 60k search trajectories with precise pivot step annotations, will be **released upon paper acceptance**.

Below is a sample showing the trajectory structure and pivot labeling:
```
{
  "question": "When did the alma mater of William C. Perry start issuing engineering degrees?",
  "trajectory": [
    {
      "step": 1,
      "observation": None,
      "think": <think>I need to parse the question: "When did the alma mater of William C. Perry start issuing engineering degrees?" First, I need to identify who William C. Perry is. Likely William C. Perry is a notable person, possibly a former US Secretary of Defense (William Perry, served 1994-1997). His full name is William James Perry, but sometimes referred to as William C. Perry? Actually, William Perry's middle name is James, not C. Maybe it's a different William C. Perry. Could be William C. Perry, maybe an academic or engineer. Let's search for William C. Perry to confirm. </think>
      "action": <search>William C. Perry alma mater</search>,
      "is_pivot": true,
    },
    {
      "step": 2,
      "observation": <information>Doc 1 (Title: William C. Perry)... Doc 2 (Title: Perry Belmont)... Doc 3 (Title: William Perry (American football))... </information>,
      "think": <think>The search results show a William C. Perry who graduated from the University of Kansas in 1922. That's likely the person in question. So his alma mater is the University of Kansas. The question asks: "When did the alma mater of William C. Perry start issuing engineering degrees?" So we need to find when the University of Kansas started issuing engineering degrees. We need to search for the history of engineering degrees at the University of Kansas. </think>
      "action": <search>University of Kansas start issuing engineering degrees</search>,
      "is_pivot": true,
      
    }
    {
      "step": 3,
      "observation": <information>Doc 1 (Title: University of Kansas)... Doc 2 (Title: University of Kansas)... Doc 3 (Title: University of Kansas)... </information>,
      "think": None
      "action": <answer>1873</answer>,
      "is_pivot": true,
      
    }
  ],
  "final_answer": "1873"
}
```
## 🤗 Models & Checkpoints

We provide checkpoints optimized for search tasks:
* **[PiCA-Qwen2.5-3B](https://huggingface.co/NeurIPS2026PiCA/PiCA-Qwen2.5-3B)**: Base policy model optimized for general search.
* **[PiCA-Qwen2.5-7B](https://huggingface.co/NeurIPS2026PiCA/PiCA-Qwen2.5-7B)**: Base policy model optimized for general search.
* **[PiCA-RM-Qwen2.5-3B](https://huggingface.co/NeurIPS2026PiCA/PiCA-RM-Qwen2.5-3B)**: The step-level reward model utilizing Pivot-Based Credit Assignment.

---
