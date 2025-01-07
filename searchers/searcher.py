from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

from flexrag.assistant import PREDEFINED_PROMPTS, AssistantBase
from flexrag.models import GENERATORS, GenerationConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import RetrievedContext
from flexrag.utils import LOGGER_MANAGER, Choices

logger = LOGGER_MANAGER.getLogger("levelrag.searcher")


GeneratorConfig = GENERATORS.make_config()


@dataclass
class BaseSearcherConfig(GeneratorConfig):
    gen_cfg: GenerationConfig = field(default_factory=GenerationConfig)
    response_type: Choices(["short", "long", "original"]) = "short"  # type: ignore


class BaseSearcher(AssistantBase):
    def __init__(self, cfg: BaseSearcherConfig) -> None:
        self.agent = GENERATORS.load(cfg)
        self.gen_cfg = cfg.gen_cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Searcher")
            self.gen_cfg.sample_num = 1

        # load assistant prompt
        match cfg.response_type:
            case "short":
                self.prompt_with_ctx = PREDEFINED_PROMPTS["shortform_with_context"]
                self.prompt_wo_ctx = PREDEFINED_PROMPTS["shortform_without_context"]
            case "long":
                self.prompt_with_ctx = PREDEFINED_PROMPTS["longform_with_context"]
                self.prompt_wo_ctx = PREDEFINED_PROMPTS["longform_without_context"]
            case "original":
                self.prompt_with_ctx = ChatPrompt()
                self.prompt_wo_ctx = ChatPrompt()
            case _:
                raise ValueError(f"Invalid response type: {cfg.response_type}")
        return

    @abstractmethod
    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        return

    def answer(self, question: str)  -> tuple[str, list[RetrievedContext], dict]:
        ctxs, history = self.search(question)
        response, prompt = self.answer_with_contexts(question, ctxs)
        return response, ctxs, {"prompt": prompt, "search_histories": history}

    def answer_with_contexts(
        self, question: str, contexts: list[RetrievedContext] = []
    ) -> tuple[str, ChatPrompt]:
        """Answer question with given contexts

        Args:
            question (str): The question to answer.
            contexts (list): The contexts searched by the searcher.

        Returns:
            response (str): response to the question
            prompt (ChatPrompt): prompt used.
        """
        # prepare system prompt
        if len(contexts) > 0:
            prompt = deepcopy(self.prompt_with_ctx)
        else:
            prompt = deepcopy(self.prompt_wo_ctx)

        # prepare user prompt
        usr_prompt = ""
        for n, context in enumerate(contexts):
            ctx = context.data.get("text")
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"

        # generate response
        prompt.update(ChatTurn(role="user", content=usr_prompt))
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return response, prompt
