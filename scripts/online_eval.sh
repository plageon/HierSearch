#!/bin/bash

python baselines/online_eval.py \
    --method_name "" \
    --data_dir "" \
    --split "test" \
    --save_note 'your-save-note-for-identification' \
    --remote_llm_url "http://127.0.0.1:18086/v1" \
    --remote_web_retriever_url "http://10.10.15.46:15005/" \
    --remote_agent_url "http://127.0.0.1:16006/" \
    --model_path "" \
    --serve_model_name "" \
    --sys_template_name "web_graph_search_agent_template_sys" \
    --max_turns 12

#zh_graph_search_agent_template_sys
#zh_web_graph_search_agent_template_sys
#zh_web_only_search_agent_template_sys
#zh_planner_agent_template_sys
