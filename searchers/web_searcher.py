import logging
import os
from copy import deepcopy
from dataclasses import dataclass

from flexrag.assistant import ASSISTANTS
from flexrag.prompt import ChatTurn, ChatPrompt
from flexrag.retriever import RetrievedContext, BingRetriever, BingRetrieverConfig

from .searcher import BaseSearcher, BaseSearcherConfig


logger = logging.getLogger("WebSearcher")


@dataclass
class WebSearcherConfig(BaseSearcherConfig, BingRetrieverConfig):
    rewrite_query: bool = False


@ASSISTANTS("web", config_class=WebSearcherConfig)
class WebSearcher(BaseSearcher):
    def __init__(self, cfg: WebSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Web Searcher
        self.rewrite = cfg.rewrite_query

        # load Web Retrieve
        self.retriever = BingRetriever(cfg)

        # load prompt
        self.rewrite_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "web_rewrite_prompt.json",
            )
        )
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        # initialize search stack
        if self.rewrite:
            query_to_search = self.rewrite_query(question)
        else:
            query_to_search = question
        ctxs = self.retriever.search(query=[query_to_search])[0]
        for ctx in ctxs:
            ctx.data["text"] = ctx.data["snippet"]
        return ctxs, []

    def rewrite_query(self, info: str) -> str:
        # Rewrite the query to be more informative
        user_prompt = f"Query: {info}"
        prompt = deepcopy(self.rewrite_prompt)
        prompt.update(ChatTurn(role="user", content=user_prompt))
        query = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return query
