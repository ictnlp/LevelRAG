import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from flexrag.assistant import ASSISTANTS
from flexrag.common_dataclass import RetrievedContext
from flexrag.prompt import ChatTurn, ChatPrompt
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig, LocalRetriever
from flexrag.utils import Choices, LOGGER_MANAGER

from .searcher import BaseSearcher, BaseSearcherConfig


logger = LOGGER_MANAGER.getLogger("levelrag.dense_searcher")


@dataclass
class DenseSearcherConfig(BaseSearcherConfig, DenseRetrieverConfig):
    rewrite_query: Choices(["never", "pseudo", "adaptive"]) = "never"  # type: ignore
    max_rewrite_depth: int = 3
    hf_repo: Optional[str] = None


@ASSISTANTS("dense", config_class=DenseSearcherConfig)
class DenseSearcher(BaseSearcher):
    def __init__(self, cfg: DenseSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Dense Searcher
        self.rewrite = cfg.rewrite_query
        self.rewrite_depth = cfg.max_rewrite_depth

        # load Dense Retrieve
        if cfg.hf_repo is not None:
            self.retriever = LocalRetriever.load_from_hub(cfg.hf_repo)
        else:
            self.retriever = DenseRetriever(cfg)

        # load prompts
        self.rewrite_with_ctx_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "rewrite_by_answer_with_context_prompt.json",
            )
        )
        self.rewrite_wo_ctx_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "rewrite_by_answer_without_context_prompt.json",
            )
        )
        self.verify_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "verify_prompt.json",
            )
        )
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        # rewrite the query
        if self.rewrite == "pseudo":
            query_to_search = self.rewrite_query(question)
        else:
            query_to_search = question

        # begin adaptive search
        ctxs = []
        search_history = []
        verification = False
        rewrite_depth = 0
        while (not verification) and (rewrite_depth < self.rewrite_depth):
            rewrite_depth += 1

            # search
            ctxs = self.retriever.search(query=[query_to_search])[0]
            search_history.append(
                {
                    "query": query_to_search,
                    "ctxs": ctxs,
                }
            )

            # verify the contexts
            if self.rewrite == "adaptive":
                verification = self.verify_contexts(ctxs, question)
            else:
                verification = True

            # adaptive rewrite
            if (not verification) and (rewrite_depth < self.rewrite_depth):
                if rewrite_depth == 1:
                    query_to_search = self.rewrite_query(question)
                else:
                    query_to_search = self.rewrite_query(question, ctxs)
        return ctxs, search_history

    def rewrite_query(
        self, question: str, contexts: list[RetrievedContext] = []
    ) -> str:
        # Rewrite the query to be more informative
        if len(contexts) == 0:
            prompt = deepcopy(self.rewrite_wo_ctx_prompt)
            user_prompt = f"Question: {question}"
        else:
            prompt = deepcopy(self.rewrite_with_ctx_prompt)
            user_prompt = ""
            for n, ctx in enumerate(contexts):
                user_prompt += f"Context {n}: {ctx.data['text']}\n\n"
            user_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        query = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return f"{question} {query}"

    def verify_contexts(
        self,
        contexts: list[RetrievedContext],
        question: str,
    ) -> bool:
        prompt = deepcopy(self.verify_prompt)
        user_prompt = ""
        for n, ctx in enumerate(contexts):
            user_prompt += f"Context {n}: {ctx.data['text']}\n\n"
        user_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return "yes" in response.lower()
