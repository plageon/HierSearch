# <div align="center">HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches</div>

<div align="center">
<a href="" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/collections/zstanjj/hiersearch-6889c44cce34aebcdfd73b4a" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Models-27b3b4.svg></a>
<a href="https://www.modelscope.cn/collections/HtmlRAG-c290f7cf673648" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://github.com/plageon/HtmlRAG/blob/main/toolkit/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>

[//]: # (<img alt="PyPI - Version" src="https://img.shields.io/pypi/v/htmlrag">)
<p>
<a href="https://github.com/plageon/HtmlRAG#-quick-start">Quick Start (快速开始)</a>&nbsp ｜ &nbsp<a href="toolkit/README_zh.md">中文文档</a>&nbsp ｜ &nbsp<a href="toolkit/README.md">English Documentation</a>&nbsp
</p>
</div>

![HierSearch](./figures/pipeline0730.png)

## 📖 Table of Contents
- [Introduction](#-introduction)
- [News](#-latest-news)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproduce Results](#-dependencies)

## ✨ Latest News

- [31/07/2025]: The open-source training and inference are released. You can apply HierSearch now.


## 📝 Introduction

1. We explore the deep search framework in multi-knowledge-source scenarios and propose a hierarchical agentic paradigm and train with HRL; 
2. We notice drawbacks of the naive information transmission among deep search agents and developed a knowledge refiner suitable for multi-knowledge-source scenarios; 
3. Our proposed approach for reliable and effective deep search across multiple knowledge sources outperforms existing baselines the flat-RL solution in various domains.

## 📦 Installation

1. Install required packages
```shell
pip intall -e .
```
2. Download required checkpoints
```shell
modelscope download --model zstanjj/HierSearch-Local-Agent --local_dir model/HierSearch-Local-Agent
modelscope download --model zstanjj/HierSearch-Web-Agent --local_dir model/HierSearch-Web-Agent
modelscope download --model zstanjj/HierSearch-Planner-Agent --local_dir model/HierSearch-Planner-Agent
modelscope download --model BAAI/bge-m3 --local_dir model/bge-m3
```

## 🔌 Apply HtmlRAG in your own RAG systems

### 🎯 Quick Start

1. Construct graph
```shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

export OPENIE_LANG="en"
python agentic_rag/construct_graph.py \
    --dataset "DATASET_NAME" \
    --llm_base_url "OPENAI_BASE_URL" \
    --llm_name "gpt-4o-mini" \
    --embedding_name "bge-m3" \
    --force_index_from_scratch "false" \
    --force_openie_from_scratch "false" \
    --openie_mode "online" \
    --save_dir "GRAPH_DIR" \
    --data_dir "DATA_DIR"
```

2. Initialize local search server
```shell
export OPENIE_LANG="en"

python agentic_rag/serve_graph_search.py \
    --num_retriever 1 \
    --port 18009 \
    --dataset_name "DATASET_NAME" \
    --save_dir "GRAGH_DIR" \
    --llm_model "gpt-4o-mini" \
    --embedding_model_name "bge-m3" \
    --corpus_path "DATASET_NAME/DATASET_NAME_corpus.json" \
    --llm_base_url "OPENAI_BASE_URL"
```

3. Initialize Web search server
```shell
python search_utils/web_dedicate_server.py
    --num_retriever 8 \
    --port 15005
```

4. Serve agents
```shell
vllm serve LOCAL_AGENT_PATH \
    --served-model-name HierSearch-Local-Agent \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 80
    
vllm serve WEB_AGENT_PATH \
    --served-model-name HierSearch-Web-Agent \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 80
    
vllm serve PLANNER_AGENT_PATH \
    --served-model-name HierSearch-Planner-Agent \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 80
```

5. Serve deep search agent server
```shell
python agentic_rag/serve_search_agent.py \
    --num_agents 8 \
    --port 16006 \
    --save_note 'your-save-note-for-identification' \
    --local_agent_llm_url "http://LOCAL_AGENT_URL/v1" \
    --web_agent_llm_url "http://WEB_AGENT_URL/v1" \
    --remote_retriever_url "http://127.0.0.1" \
    --remote_web_browse_url "http://127.0.0.1:15005" \
    --remote_web_retriever_url "http://127.0.0.1:15005" \
    --local_agent_llm_model_path "LOCAL_AGENT_PATH" \
    --web_agent_llm_model_path "WEB_AGENT_PATH" \
    --local_agent_serve_model_name "HierSearch-Local-Agent" \
    --web_agent_serve_model_name "HierSearch-Web-Agent" \
    --embedding_model_name "bge-m3" \
    --max_turns 8 \
    --single_sample \
    --question "Who is the sibling of the author of Kapalkundala?" \
    --filter_ratio 0.5
```

6. Run online inference
```shell
python baselines/online_eval.py \
    --method_name "HierSearch" \
    --data_dir "data" \
    --split "test" \
    --remote_llm_url "http://PLANNER_AGENT_PATH/v1" \
    --remote_agent_url "http://127.0.0.1:16006/" \
    --model_path "PLANNER_AGENT_PATH" \
    --serve_model_name "HierSearch-Planner-Agent" \
    --sys_template_name "web_graph_search_agent_template_sys" \
    --max_turns 8
```


### 🚀 Reproduce Our Results
Download dataset
```shell
modelscope download zstanjj/HierSearch-Datasets --local_dir data --repo-type dataset
```

Run deep search
```shell
python baselines/online_eval.py \
    --method_name "HierSearch" \
    --data_dir "DATA_DIR" \
    --split "test" \
    --save_note 'your-save-note-for-identification' \
    --remote_llm_url "http://127.0.0.1:18086/v1" \
    --remote_web_retriever_url "http://10.10.15.46:15005/" \
    --remote_agent_url "http://127.0.0.1:16006/" \
    --model_path "PLANNER_AGENT_PATH" \
    --serve_model_name "HierSearch-Planner-Agent" \
    --sys_template_name "web_graph_search_agent_template_sys" \
    --max_turns 8
```

## 🚀 Training

```shell
export PROMPT_TEMPLATE_NAME="local"
export SUPPORTED_TOOLS="[chunk_search, graph_search, get_adjacent_passages]"
./scripts/train_deep_search.sh


export PROMPT_TEMPLATE_NAME="web"
export SUPPORTED_TOOLS="[web_search, browse_url]"
./scripts/train_deep_search.sh


export PROMPT_TEMPLATE_NAME="planner"
export SUPPORTED_TOOLS="[local_search_agent, web_search_agent, all_search_agent]"
./scripts/train_deep_search.sh
```

## 📜 Citation

```bibtex
@misc{hiersearch2025,
  title={HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches},
  author={Jiejun Tan and Zhicheng Dou and Yan Yu and Jiehan Cheng and Qiang Ju and Jian Xie and Ji-Rong Wen},
  year={2025},
  eprint={25000000},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
