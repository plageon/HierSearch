# <div align="center">HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches</div>

<div align="center">
<a href="https://arxiv.org/abs/2508.08088" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/papers/2508.08088" target="_blank"><img src=https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface></a>
<a href="https://huggingface.co/collections/zstanjj/hiersearch-6889c44cce34aebcdfd73b4a" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Models-27b3b4.svg></a>
<a href="https://www.modelscope.cn/collections/HierSearch-6638ad769ddb44" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://github.com/plageon/HierSearch/blob/main/toolkit/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>

[//]: # (<img alt="PyPI - Version" src="https://img.shields.io/pypi/v/htmlrag">)
<p>
<a href="https://github.com/plageon/HtmlRAG#-quick-start">Quick Start (快速开始)</a>&nbsp ｜ &nbsp<a href="README_zh.md">中文文档</a>&nbsp ｜ &nbsp<a href="README.md">English Documentation</a>&nbsp
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

- [31/07/2025]: - 我们开源了训练和推理代码，欢迎使用HierSearch。


## 📝 简介

1. 我们探索了多知识源场景下的深度搜索框架，提出了分层深度搜索范式，并使用分层强化学习进行训练；
2. 我们注意到深度搜索代理之间信息传递的缺陷，并提出了适用于多知识源场景的知识精炼器；
3. 我们提出的跨多个知识源的可靠有效的深度搜索方法在各个领域中优于现有的基线方法以及非分层的强化学习解决方案。

## 📦 安装

1. 安装必要的包
```shell
conda create -n hiersearch python=3.10
conda activate hiersearch
pip install -e .
cd HippoRAG
pip install -e .
```
2. 下载必要的模型参数
```shell
modelscope download --model zstanjj/HierSearch-Local-Agent --local_dir model/HierSearch-Local-Agent
modelscope download --model zstanjj/HierSearch-Web-Agent --local_dir model/HierSearch-Web-Agent
modelscope download --model zstanjj/HierSearch-Planner-Agent --local_dir model/HierSearch-Planner-Agent
modelscope download --model BAAI/bge-m3 --local_dir model/bge-m3
```

## 🔌 在你的项目中使用HierSearch

### 🎯 Quick Start

1. 建图
    
    我们已经为论文中使用的六个基准数据集构建了图。
    从[ModelScope](https://www.modelscope.cn/collections/HtmlRAG-c290f7cf673648)或[HuggingFace](https://huggingface.co/collections/zstanjj/hiersearch-6889c44cce34aebcdfd73b4a)下载图文件。
    在每个数据集目录中创建一个名为`graph`的目录，并将图文件`openie_results_ner_gpt-4o-mini.json`放入其中。
    您的目录结构应如下所示：

    ```
    data/
    ├── musique/
    │   ├── musique_corpus.json
    │   ├── train.jsonl
    │   ├── test.jsonl
    │   ├── graph/
    │   │   └── openie_results_ner_gpt-4o-mini.json
    ├── omnieval/
    ...
    ```
    如果您需要使用自己的数据集，替换`DATASET_NAME_corpus.json`为您自己的数据集文件，并且从头开始构建图。

    您还需要运行以下脚本来建立索引。
    ```shell
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    
    export OPENIE_LANG="en" # set to "zh" for Chinese datasets (omnieval, etc.)
    python agentic_rag/construct_graph.py \
        --dataset "DATASET_NAME" \
        --llm_base_url "OPENAI_BASE_URL" \
        --llm_name "gpt-4o-mini" \
        --embedding_name "bge-m3" \
        --force_index_from_scratch "false" \
        --force_openie_from_scratch "false" \
        --openie_mode "online" \
        --save_dir "GRAPH_DIR" \
        --data_dir "data/DATASET_NAME"
    ```

2. 初始化本地搜索工具
    ```shell
    export OPENIE_LANG="en" # set to "zh" for Chinese datasets (omnieval, etc.)
    
    python agentic_rag/serve_graph_search.py \
        --num_retriever 1 \
        --port 18009 \
        --dataset_name "DATASET_NAME" \
        --save_dir "data/DATASET_NAME/graph" \
        --llm_model "gpt-4o-mini" \
        --embedding_model_name "bge-m3" \
        --corpus_path "DATASET_NAME/DATASET_NAME_corpus.json" \
        --llm_base_url "OPENAI_BASE_URL"
    ```

3. 初始化联网搜索工具
    ```shell
    python search_utils/web_dedicate_server.py
        --num_retriever 8 \
        --port 15005
    ```
    
4. 启动智能体模型
   ```shell
   # Serve the local agent on a machine with LOCAL_AGENT_URL
   vllm serve LOCAL_AGENT_PATH \
       --served-model-name HierSearch-Local-Agent \
       --gpu-memory-utilization 0.90 \
       --dtype bfloat16 \
       --trust-remote-code \
       --port 80
   ```
   ```shell 
   # Serve the web agent on a machine with WEB_AGENT_URL
   vllm serve WEB_AGENT_PATH \
       --served-model-name HierSearch-Web-Agent \
       --gpu-memory-utilization 0.90 \
       --dtype bfloat16 \
       --trust-remote-code \
       --port 80
   ```
   ```shell 
   # Serve the planner agent on a machine with PLANNER_AGENT_URL
   vllm serve PLANNER_AGENT_PATH \
       --served-model-name HierSearch-Planner-Agent \
       --gpu-memory-utilization 0.90 \
       --dtype bfloat16 \
       --trust-remote-code \
       --port 80
   ```

5. 启动智能体服务
    ```shell
    python agentic_rag/serve_search_agent.py \
        --num_agents 8 \
        --port 16006 \
        --local_agent_llm_url "http://LOCAL_AGENT_URL/v1" \
        --web_agent_llm_url "http://WEB_AGENT_URL/v1" \
        --remote_retriever_url "http://127.0.0.1" \
        --remote_web_browse_url "http://127.0.0.1:15005" \
        --remote_web_retriever_url "http://127.0.0.1:15005" \
        --local_agent_llm_model_path "model/HierSearch-Local-Agent" \
        --web_agent_llm_model_path "model/HierSearch-Web-Agent" \
        --local_agent_serve_model_name "HierSearch-Local-Agent" \
        --web_agent_serve_model_name "HierSearch-Web-Agent" \
        --embedding_model_name "bge-m3" \
        --max_turns 8 \
        --single_sample \
        --question "Who is the sibling of the author of Kapalkundala?" \
        --filter_ratio 0.5
    ```

6. 运行在线推理
    ```shell
    python baselines/online_eval.py \
        --method_name "HierSearch" \
        --data_dir "data" \
        --split "test" \
        --remote_llm_url "http://PLANNER_AGENT_URL/v1" \
        --remote_agent_url "http://127.0.0.1:16006/" \
        --model_path "model/HierSearch-Planner-Agent" \
        --serve_model_name "HierSearch-Planner-Agent" \
        --sys_template_name "planner_agent_template_sys" \
        --max_turns 8
    ```


### 🚀 复现我们的结果

我们在通用领域、医疗领域和金融领域的六个基准上进行了广泛的实验。
结果表明，HierSearch在所有基准上都优于基线和非分层的强化学习解决方案。

![MainResults](./figures/main-results.png)
1. 下载数据集
    ```shell
    modelscope download zstanjj/HierSearch-Datasets --local_dir data --repo-type dataset
    ```

2. 深度搜索推理
    ```shell
    python baselines/online_eval.py \
        --method_name "HierSearch" \
        --data_dir "data" \
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


## 📜 引用

```bibtex
@misc{tan2025hiersearchhierarchicalenterprisedeep,
      title={HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches}, 
      author={Jiejun Tan and Zhicheng Dou and Yan Yu and Jiehan Cheng and Qiang Ju and Jian Xie and Ji-Rong Wen},
      year={2025},
      eprint={2508.08088},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.08088}, 
}
```
