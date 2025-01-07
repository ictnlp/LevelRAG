#!/bin/bash

LEVELRAG_PATH="<path_to_levelrag>"
MODEL_NAME="Qwen2-7B-Instruct"
BASE_URL="http://127.0.0.1:8000/v1"
DATA_PATH=popqa/dev.jsonl
ELASTIC_HOST=http://127.0.0.1:9200/
DENSE_PATH=<path_to_your_dense_retriever_database>
ENCODER_PATH=facebook/contriever-msmarco
BING_KEY="<your_bing_search_subscription_key>"


python -m flexrag.entrypoints.run_assistant \
    user_module=$LEVELRAG_PATH/searchers \
    data_path=$DATA_PATH \
    assistant_type=hybrid \
    hybrid_config.searchers=[keyword,web,dense] \
    hybrid_config.keyword_config.rewrite_query=adaptive \
    hybrid_config.keyword_config.feedback_depth=3 \
    hybrid_config.keyword_config.response_type=short \
    hybrid_config.keyword_config.generator_type=openai \
    hybrid_config.keyword_config.openai_config.model_name=$MODEL_NAME \
    hybrid_config.keyword_config.openai_config.base_url=$BASE_URL \
    hybrid_config.keyword_config.gen_cfg.do_sample=False \
    hybrid_config.keyword_config.host=$ELASTIC_HOST \
    hybrid_config.keyword_config.index_name=wiki_2021 \
    hybrid_config.dense_config.rewrite_query=adaptive \
    hybrid_config.dense_config.response_type=short \
    hybrid_config.dense_config.generator_type=openai \
    hybrid_config.dense_config.openai_config.model_name=$MODEL_NAME \
    hybrid_config.dense_config.openai_config.base_url=$BASE_URL \
    hybrid_config.dense_config.gen_cfg.do_sample=False \
    hybrid_config.dense_config.database_path=$DENSE_PATH \
    hybrid_config.dense_config.index_type=faiss \
    hybrid_config.dense_config.query_encoder_config.encoder_type=hf \
    hybrid_config.dense_config.query_encoder_config.hf_config.model_path=$ENCODER_PATH \
    hybrid_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    hybrid_config.web_config.rewrite_query=False \
    hybrid_config.web_config.response_type=short \
    hybrid_config.web_config.generator_type=openai \
    hybrid_config.web_config.openai_config.model_name=$MODEL_NAME \
    hybrid_config.web_config.openai_config.base_url=$BASE_URL \
    hybrid_config.web_config.gen_cfg.do_sample=False \
    hybrid_config.web_config.subscription_key=$BING_KEY \
    hybrid_config.response_type=short \
    hybrid_config.generator_type=openai \
    hybrid_config.openai_config.model_name=$MODEL_NAME \
    hybrid_config.openai_config.base_url=$BASE_URL \
    hybrid_config.gen_cfg.do_sample=False \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
