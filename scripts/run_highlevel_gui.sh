#!/bin/bash

LEVELRAG_PATH="<path_to_levelrag>"
MODEL_NAME="Qwen2-7B-Instruct"
BASE_URL="http://127.0.0.1:8000/v1"
ELASTIC_HOST=http://127.0.0.1:9200/
DENSE_PATH=<path_to_your_dense_retriever_database>
ENCODER_PATH=facebook/contriever-msmarco
BING_KEY="<your_bing_search_subscription_key>"


python -m flexrag.entrypoints.run_interactive \
    user_module=$LEVELRAG_PATH/searchers \
    assistant_type=highlevel \
    highlevel_config.searchers=[keyword,web,dense] \
    highlevel_config.decompose=True \
    highlevel_config.summarize_for_decompose=True \
    highlevel_config.summarize_for_answer=True \
    highlevel_config.keyword_config.rewrite_query=adaptive \
    highlevel_config.keyword_config.feedback_depth=3 \
    highlevel_config.keyword_config.response_type=short \
    highlevel_config.keyword_config.generator_type=openai \
    highlevel_config.keyword_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.keyword_config.openai_config.base_url=$BASE_URL \
    highlevel_config.keyword_config.gen_cfg.do_sample=False \
    highlevel_config.keyword_config.host=$ELASTIC_HOST \
    highlevel_config.keyword_config.index_name=wiki_2021 \
    highlevel_config.dense_config.rewrite_query=adaptive \
    highlevel_config.dense_config.response_type=short \
    highlevel_config.dense_config.generator_type=openai \
    highlevel_config.dense_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.dense_config.openai_config.base_url=$BASE_URL \
    highlevel_config.dense_config.gen_cfg.do_sample=False \
    highlevel_config.dense_config.database_path=$DENSE_PATH \
    highlevel_config.dense_config.index_type=faiss \
    highlevel_config.dense_config.query_encoder_config.encoder_type=hf \
    highlevel_config.dense_config.query_encoder_config.hf_config.model_path=$ENCODER_PATH \
    highlevel_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    highlevel_config.web_config.search_engine_type=bing \
    highlevel_config.web_config.bing_config.subscription_key=$BING_KEY \
    highlevel_config.web_config.web_reader_type=snippet \
    highlevel_config.web_config.rewrite_query=False \
    highlevel_config.web_config.response_type=short \
    highlevel_config.web_config.generator_type=openai \
    highlevel_config.web_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.web_config.openai_config.base_url=$BASE_URL \
    highlevel_config.web_config.gen_cfg.do_sample=False \
    highlevel_config.response_type=short \
    highlevel_config.generator_type=openai \
    highlevel_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.openai_config.base_url=$BASE_URL \
    highlevel_config.gen_cfg.do_sample=False
