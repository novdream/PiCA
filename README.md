
# PiCA: Pivot-Based Credit Assignment For Search Agentic Reinforcement Learning

This is the **official implementation** of the project **"PiCA: Pivot-Based Credit Assignment For Search Agentic Reinforcement Learning"**.

---

## 💡 Overview

**PiCA** is a reinforcement learning (RL) framework designed to optimize the search and multi-hop reasoning capabilities of Large Language Model (LLM) agents. The core of the framework is the **Pivot-Based Credit Assignment (PiCA)** mechanism, which addresses the challenges of sparse rewards in complex reasoning trajectories.

### Key Contributions:
*   **PiCA Reward Mechanism**: A novel step-reward approach that identifies and prioritizes "Pivot Steps" with high information gain.
*   **Large-Scale Dataset**: Includes a step-labeled dataset based on **MuSiQue**, consisting of approximately 60,000 samples for training reasoning agents.
*   **Scalability**: Proven effective across multiple model scales and architectures, including the **Qwen** and **Llama** families.
*   **High-Performance Integration**: Fully compatible with distributed training frameworks like **OpenRLHF**, **Ray**, **DeepSpeed**, and **vLLM**.

---

## 📂 Project Structure

```text
Search-R1/
├── OpenRLHF/                # Customized OpenRLHF for agentic RL training
├── Search-R1/
│   ├── data/                # ~60k step-labeled MuSiQue dataset
│   ├── models/              # PiCA reward model and policy definitions
│   └── utils/               # Trajectory parsing and pivot step detection
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
