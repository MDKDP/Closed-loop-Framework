# From Detection to Discovery: A Closed-Loop Approach for Continuous Medical Knowledge Expansion and Depression Detection on Social Media

## 📘 Overview
### This repository provides a modular framework for Knowledge Graph Module, containing **knowledge graph construction**, **pretraining**, **expansion**, and **refinement**.  
### As for the LoRA-based Deepseek fine-tuning module, please refer to "https://hugging-face.cn/docs/peft/task_guides/lora_based_methods"
---

## 🧩 Directory Structure

```
code/
├── MCTS/
│   ├── construct_graph.py       # Knowledge graph construction
│   ├── monte_search.py          # Monte Carlo search algorithm
│   └── __init__.py
│
├── Main/
│   └── kg_refine_and_kg_expand.py   # Unified interface for KG operations (pretrain, expansion, refinement)
│
├── RGHAT/
│   ├── RGHAT.py                 # RGHAT architecture or KG embedding network
│   ├── batch_test.py            # Batch testing utilities
│   ├── load_data_rghat.py       # Data loading functions
│   ├── loader_rghat.py          # Data loading functions
│   ├── parser_rghat.py          # Parameters
│   ├── rghat_main.py            # RGHAT training script
│   └── __init__.py
│
├── LICENSE
└── README.md
```
---

## 📜 License
This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.

---
