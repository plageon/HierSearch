export OPENIE_LANG="en"

python agentic_rag/serve_graph_search.py \
    --num_retriever 1 \
    --port 18009 \
    --dataset_name "musique" \
    --save_dir "data" \
    --llm_model "gpt-4o-mini" \
    --embedding_model_name "model/bge-m3" \
    --corpus_path "data/musique/musique_corpus.json" \
    --llm_base_url "http://127.0.0.1:80"