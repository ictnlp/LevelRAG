#!/bin/bash

set -euo pipefail

WIKI_FILE=text-list-100-sec.jsonl
WIKI_INFOBOX=infobox.jsonl
DENSE_PATH=<path_to_your_dense_retriever_database>
ENCODER_PATH=facebook/contriever-msmarco

python flexrag.entrypoints.prepare_index \
    retriever_type=dense \
    corpus_path=[$WIKI_FILE,$WIKI_INFOBOX] \
    dense_config.database_path=$DENSE_PATH \
    dense_config.passage_encoder_config.encoder_type=hf \
    dense_config.passage_encoder_config.hf_config.model_path=$ENCODER_PATH \
    dense_config.passage_encoder_config.hf_config.device_id=[0] \
    dense_config.index_type=faiss \
    dense_config.batch_size=1024 \
    dense_config.log_interval=100000

