import os
import re
from copy import deepcopy
from dataclasses import dataclass

from flexrag.assistant import ASSISTANTS
from flexrag.common_dataclass import RetrievedContext
from flexrag.prompt import ChatPrompt, ChatTurn

from .hybrid_searcher import HybridSearcher, HybridSearcherConfig


@dataclass
class HighLevelSearcherConfig(HybridSearcherConfig):
    decompose: bool = False
    max_decompose_times: int = 100
    summarize_for_decompose: bool = False
    summarize_for_answer: bool = False


@ASSISTANTS("highlevel", config_class=HighLevelSearcherConfig)
class HighLevalSearcher(HybridSearcher):
    def __init__(self, cfg: HighLevelSearcherConfig) -> None:
        super().__init__(cfg)
        # set basic args
        if not cfg.decompose:
            self.max_decompose_times = 0
        else:
            self.max_decompose_times = cfg.max_decompose_times
        self.summarize_for_decompose = cfg.summarize_for_decompose
        self.summarize_for_answer = cfg.summarize_for_answer

        # load prompt
        self.decompose_prompt_w_ctx = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "decompose_with_context_prompt.json",
            )
        )
        self.decompose_prompt_wo_ctx = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "decompose_without_context_prompt.json",
            )
        )
        self.summarize_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "summarize_by_answer_prompt.json",
            )
        )
        return

    def decompose_question(
        self,
        question: str,
        search_history: list[dict[str, str | RetrievedContext]] = [],
    ) -> list[str]:
        # form prompt
        if len(search_history) > 0:
            prompt = deepcopy(self.decompose_prompt_w_ctx)
            ctx_str = self.compose_contexts(search_history)
            prompt.update(
                ChatTurn(role="user", content=f"Question: {question}\n\n{ctx_str}")
            )
        else:
            prompt = deepcopy(self.decompose_prompt_wo_ctx)
            prompt.update(ChatTurn(role="user", content=f"Question: {question}"))

        # get response
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        if "No additional information is required" in response:
            return []
        split_pattern = r"\[\d+\] ([^\[]+)"
        decompsed = re.findall(split_pattern, response)
        # If the question is not decomposed, fallback to original question
        if (len(decompsed) == 0) and (len(search_history) == 0):
            decompsed = [question]

        # deduplicate questions
        searched = set()
        for s in search_history:
            searched.add(s["question"])
        decompsed = [i for i in decompsed if i not in searched]
        return decompsed

    def compose_contexts(
        self, search_history: list[dict[str, str | list[RetrievedContext]]]
    ) -> str:
        if self.summarize_for_decompose:
            summed_text = self.summarize_history(search_history)
            ctx_text = ""
            for n, text in enumerate(summed_text):
                ctx_text += f"Context {n + 1}: {text}\n\n"
            ctx_text = ctx_text[:-2]
        else:
            ctx_text = ""
            n = 1
            for item in search_history:
                for ctx in item["contexts"]:
                    ctx_text += f"Context {n}: {ctx.data['text']}\n\n"
                    n += 1
            ctx_text = ctx_text[:-2]
        return ctx_text

    def summarize_history(
        self, search_history: list[dict[str, str | list[RetrievedContext]]]
    ) -> list[str]:
        summed_text = []
        for item in search_history:
            prompt = deepcopy(self.summarize_prompt)
            q = item["question"]
            usr_prompt = ""
            for n, ctx in enumerate(item["contexts"]):
                usr_prompt += f"Context {n + 1}: {ctx.data['text']}\n\n"
            usr_prompt += f"Question: {q}"
            prompt.update(ChatTurn(role="user", content=usr_prompt))
            ans = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
            summed_text.append(ans)
        return summed_text

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        contexts = []
        search_history = []
        decompose_times = self.max_decompose_times
        if decompose_times > 0:
            decomposed_questions = self.decompose_question(question)
            decompose_times -= 1
        else:
            decomposed_questions = [question]

        # search the decomposed_questions
        while len(decomposed_questions) > 0:
            q = decomposed_questions.pop(0)
            ctxs, _ = super().search(q)
            search_history.append({"question": q, "contexts": ctxs})
            contexts.extend(ctxs)
            if (len(decomposed_questions) == 0) and (decompose_times > 0):
                decomposed_questions = self.decompose_question(question, search_history)
                decompose_times -= 1

        # post process
        if self.summarize_for_answer:
            summed_text = self.summarize_history(search_history)
            contexts = [
                RetrievedContext(
                    retriever="highlevel_searcher",
                    query=j["question"],
                    data={"text": i},
                )
                for i, j in zip(summed_text, search_history)
            ]
        return contexts, search_history
