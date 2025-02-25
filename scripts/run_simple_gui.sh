#!/bin/bash

LEVELRAG_PATH="<path_to_levelrag>"
MODEL_NAME="Qwen2-7B-Instruct"
BASE_URL="http://127.0.0.1:8000/v1"


python -m flexrag.entrypoints.run_interactive \
    user_module=$LEVELRAG_PATH/searchers \
    assistant_type=highlevel \
    highlevel_config.searchers=[dense] \
    highlevel_config.decompose=True \
    highlevel_config.summarize_for_decompose=True \
    highlevel_config.summarize_for_answer=True \
    highlevel_config.dense_config.rewrite_query=adaptive \
    highlevel_config.dense_config.response_type=short \
    highlevel_config.dense_config.generator_type=openai \
    highlevel_config.dense_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.dense_config.openai_config.base_url=$BASE_URL \
    highlevel_config.dense_config.gen_cfg.do_sample=False \
    highlevel_config.dense_config.hf_repo='FlexRAG/wiki2021_atlas_contriever' \
    highlevel_config.response_type=short \
    highlevel_config.generator_type=openai \
    highlevel_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.openai_config.base_url=$BASE_URL \
    highlevel_config.gen_cfg.do_sample=False
