#!/bin/bash

set -euo pipefail

WIKI_FILE=text-list-100-sec.jsonl
WIKI_INFOBOX=infobox.jsonl
ELASTIC_HOST=http://127.0.0.1:9200/

python -m flexrag.entrypoints.prepare_index \
    retriever_type=elastic \
    file_paths=[$WIKI_FILE,$WIKI_INFOBOX] \
    saving_fields=[title,section,text] \
    id_field=id \
    elastic_config.host=$ELASTIC_HOST \
    elastic_config.index_name=wiki \
    elastic_config.batch_size=512 \
    elastic_config.log_interval=100 \
    reinit=True
