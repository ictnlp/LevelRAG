#!/bin/bash

LEVELRAG_PATH="<path_to_levelrag>"
MODEL_NAME="gpt-4o"
API_KEY="<your_openai_api_key>"


python -m flexrag.entrypoints.run_assistant \
    user_module=$LEVELRAG_PATH/searchers \
    name=nq \
    split=test \
    assistant_type=highlevel \
    highlevel_config.searchers=[dense] \
    highlevel_config.decompose=True \
    highlevel_config.summarize_for_decompose=True \
    highlevel_config.summarize_for_answer=True \
    highlevel_config.dense_config.rewrite_query=adaptive \
    highlevel_config.dense_config.response_type=short \
    highlevel_config.dense_config.generator_type=openai \
    highlevel_config.dense_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.dense_config.openai_config.api_key=$API_KEY \
    highlevel_config.dense_config.gen_cfg.do_sample=False \
    highlevel_config.dense_config.hf_repo='FlexRAG/wiki2021_atlas_contriever' \
    highlevel_config.response_type=short \
    highlevel_config.generator_type=openai \
    highlevel_config.openai_config.model_name=$MODEL_NAME \
    highlevel_config.openai_config.api_key=$api_key \
    highlevel_config.gen_cfg.do_sample=False \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
