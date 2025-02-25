from dataclasses import dataclass, field

from flexrag.assistant import ASSISTANTS
from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import Choices

from .keyword_searcher import KeywordSearcher, KeywordSearcherConfig
from .dense_searcher import DenseSearcher, DenseSearcherConfig
from .searcher import BaseSearcher, BaseSearcherConfig
from .web_searcher import WebSearcher, WebSearcherConfig


@dataclass
class HybridSearcherConfig(BaseSearcherConfig):
    searchers: list[Choices(["keyword", "web", "dense"])] = field(default_factory=list)  # type: ignore
    keyword_config: KeywordSearcherConfig = field(default_factory=KeywordSearcherConfig)  # fmt: skip
    web_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    dense_config: DenseSearcherConfig = field(default_factory=DenseSearcherConfig)  # fmt: skip


@ASSISTANTS("hybrid", config_class=HybridSearcherConfig)
class HybridSearcher(BaseSearcher):
    def __init__(self, cfg: HybridSearcherConfig) -> None:
        super().__init__(cfg)
        # load searchers
        self.searchers = self.load_searchers(
            searchers=cfg.searchers,
            bm25_cfg=cfg.keyword_config,
            web_cfg=cfg.web_config,
            dense_cfg=cfg.dense_config,
        )
        return

    def load_searchers(
        self,
        searchers: list[str],
        bm25_cfg: KeywordSearcherConfig,
        web_cfg: WebSearcherConfig,
        dense_cfg: DenseSearcherConfig,
    ) -> dict[str, BaseSearcher]:
        searcher_list = {}
        for searcher in searchers:
            match searcher:
                case "keyword":
                    searcher_list[searcher] = KeywordSearcher(bm25_cfg)
                case "web":
                    searcher_list[searcher] = WebSearcher(web_cfg)
                case "dense":
                    searcher_list[searcher] = DenseSearcher(dense_cfg)
                case _:
                    raise ValueError(f"Searcher {searcher} not supported")
        return searcher_list

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        # search the question using sub-searchers
        contexts = []
        search_history = []
        for name, searcher in self.searchers.items():
            ctxs = searcher.search(question)[0]
            contexts.extend(ctxs)
            search_history.append(
                {
                    "searcher": name,
                    "context": ctxs,
                }
            )
        return contexts, search_history
