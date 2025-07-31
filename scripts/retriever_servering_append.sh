#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
python agentic_rag/serve_graph_search.py \
    --num_retriever 1 \
    --port 18011 \
    --dataset_name "nq" \
    --save_dir "data/nq/graph/" \
    --llm_model "gpt-4o-mini" \
    --embedding_model_name "model/bge-m3" \
    --corpus_path "nq/nq_corpus.json" \
    --llm_base_url "" &

export CUDA_VISIBLE_DEVICES=3
python agentic_rag/serve_graph_search.py \
    --num_retriever 1 \
    --port 18012 \
    --dataset_name "hotpotqa" \
    --save_dir "hotpotqa/graph/" \
    --llm_model "gpt-4o-mini" \
    --embedding_model_name "model/bge-m3" \
    --corpus_path "hotpotqa/hotpotqa_corpus.json" \
    --llm_base_url "" &

export CUDA_VISIBLE_DEVICES=4
python agentic_rag/serve_graph_search.py \
    --num_retriever 1 \
    --port 18013 \
    --dataset_name "pubmedqa" \
    --save_dir "pubmedqa/graph/" \
    --llm_model "gpt-4o-mini" \
    --embedding_model_name "model/bge-m3" \
    --corpus_path "pubmedqa/pubmedqa_corpus.json" \
    --llm_base_url ""

