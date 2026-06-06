from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool

from agent.config import AgentConfig
from agent.emailer import send_email
from agent.fetcher import fetch_properties
from agent.llm_scorer import enrich_finalists_with_llm
from agent.scoring import rank_properties
from agent.tracker import append_new_report_entries

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    raw_df: pd.DataFrame
    deduped_df: pd.DataFrame
    filtered_df: pd.DataFrame
    ranked_df: pd.DataFrame
    enriched_df: pd.DataFrame
    new_tracker_rows: pd.DataFrame


class HomeScoutLangChainAgent:
    """LangChain tool orchestrator for the HomeScout daily workflow.

    The LLM enriches finalist analysis, but data fetching, ranking, persistence,
    and email side effects stay in deterministic Python functions.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._state: dict[str, Any] = {}
        self.tools = self._build_tools()

    def _build_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.fetch_listings,
                name="fetch_listings",
                description="Fetch and filter real estate listings from configured locations.",
            ),
            StructuredTool.from_function(
                func=self.rank_listings,
                name="rank_listings",
                description="Score and rank filtered listings deterministically.",
            ),
            StructuredTool.from_function(
                func=self.enrich_finalists,
                name="enrich_finalists",
                description="Use LangChain model calls to add qualitative finalist analysis.",
            ),
            StructuredTool.from_function(
                func=self.append_tracker,
                name="append_tracker",
                description="Append new finalist listings to the live report tracker.",
            ),
            StructuredTool.from_function(
                func=self.send_report_email,
                name="send_report_email",
                description="Send or dry-run the daily real estate email report.",
            ),
        ]

    def fetch_listings(self) -> str:
        raw_df, deduped_df, filtered_df = fetch_properties(self.config)
        self._state["raw_df"] = raw_df
        self._state["deduped_df"] = deduped_df
        self._state["filtered_df"] = filtered_df
        fetch_failures = filtered_df.attrs.get("fetch_failures", [])
        self._state["fetch_failures"] = fetch_failures

        if fetch_failures:
            logger.warning("Fetch failures: %s", "; ".join(fetch_failures))

        return (
            f"Fetched {len(raw_df)} raw listings, deduplicated to {len(deduped_df)}, "
            f"and filtered to {len(filtered_df)} candidates."
        )

    def rank_listings(self) -> str:
        filtered_df = self._state.get("filtered_df")
        if not isinstance(filtered_df, pd.DataFrame):
            raise RuntimeError("fetch_listings must run before rank_listings.")

        ranked_df = rank_properties(filtered_df, self.config)
        ranked_df.attrs["fetch_failures"] = self._state.get("fetch_failures", [])
        self._state["ranked_df"] = ranked_df
        return f"Ranked top {len(ranked_df)} listings."

    def enrich_finalists(self) -> str:
        ranked_df = self._state.get("ranked_df")
        if not isinstance(ranked_df, pd.DataFrame):
            raise RuntimeError("rank_listings must run before enrich_finalists.")

        enriched_df = enrich_finalists_with_llm(ranked_df, self.config)
        enriched_df.attrs["fetch_failures"] = self._state.get("fetch_failures", [])
        self._state["enriched_df"] = enriched_df
        return f"Prepared {len(enriched_df)} enriched finalists."

    def append_tracker(self) -> str:
        enriched_df = self._state.get("enriched_df")
        if not isinstance(enriched_df, pd.DataFrame):
            raise RuntimeError("enrich_finalists must run before append_tracker.")

        new_tracker_rows = append_new_report_entries(enriched_df, self.config)
        self._state["new_tracker_rows"] = new_tracker_rows
        return f"Added {len(new_tracker_rows)} new tracker rows."

    def send_report_email(self) -> str:
        enriched_df = self._state.get("enriched_df")
        if not isinstance(enriched_df, pd.DataFrame):
            raise RuntimeError("enrich_finalists must run before send_report_email.")

        send_email(enriched_df, self.config)
        return f"Sent report for {len(enriched_df)} finalists."

    def run(self) -> AgentRunResult:
        for tool in self.tools:
            logger.info("Running LangChain tool: %s", tool.name)
            tool.invoke({})

        return AgentRunResult(
            raw_df=self._state["raw_df"],
            deduped_df=self._state["deduped_df"],
            filtered_df=self._state["filtered_df"],
            ranked_df=self._state["ranked_df"],
            enriched_df=self._state["enriched_df"],
            new_tracker_rows=self._state["new_tracker_rows"],
        )


def run_home_scout_agent(config: AgentConfig) -> AgentRunResult:
    return HomeScoutLangChainAgent(config).run()
