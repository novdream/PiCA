
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

## 🚀 Quick Start

### 1. Setup & Environment

**Requirements:**
* Python 3.10+
* CUDA 12.1+

**Installation:**
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/YourUsername/Search-R1.git
cd Search-R1/OpenRLHF
pip install -e .
pip install -r ../requirements.txt
```

### 2. Configuration
- Update the `DATA_PATH` and `MODEL_PATH` in the execution scripts located in `scripts/`.
- Ensure your environment is configured for distributed training via **Ray** and **DeepSpeed**.

### 3. Launch Training
The training process typically involves starting the inference engine and then launching the RL pipeline:
```bash
cd scripts
# Start the vLLM engine for the reward model
bash start_vllm_engine.sh

# Launch the PiCA-based policy optimization
bash run_pica_rl.sh
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
* **Search-R1-Qwen2.5-7B**: Base policy model optimized for general search.
* **Search-R1-Llama3.1-8B**: High-performance reasoning model.
* **PiCA-RM-Qwen-4B**: The step-level reward model utilizing Pivot-Based Credit Assignment.

---
