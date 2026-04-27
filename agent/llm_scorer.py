from __future__ import annotations

import json
import logging
import re

import pandas as pd

from agent.config import AgentConfig

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, str]:
    cleaned = text.strip()
    fenced_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(0)
    payload = json.loads(cleaned)
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "criteria_match": str(payload.get("criteria_match", "")).strip(),
        "possible_concern": str(payload.get("possible_concern", "")).strip(),
    }


def enrich_finalists_with_llm(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    enriched = df.copy()
    for column in ["llm_summary", "llm_criteria_match", "llm_possible_concern"]:
        if column not in enriched.columns:
            enriched[column] = None

    if enriched.empty:
        return enriched

    if not config.enable_openai_scoring:
        logger.info("OpenAI scoring disabled; skipping qualitative summaries.")
        return enriched

    if not config.openai_api_key:
        logger.warning("ENABLE_OPENAI_SCORING is true but OPENAI_API_KEY is missing; skipping.")
        return enriched

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI SDK is not installed; skipping qualitative summaries.")
        return enriched

    client = OpenAI(api_key=config.openai_api_key)

    for index, row in enriched.iterrows():
        prompt = (
            "You are helping summarize a real estate listing.\n"
            "Return compact JSON with keys summary, criteria_match, possible_concern.\n"
            "Do not include markdown.\n\n"
            f"Subjective criteria: {config.subjective_criteria or 'None provided'}\n"
            f"Address: {row.get('formatted_address')}\n"
            f"City: {row.get('city')}, {row.get('state')}\n"
            f"Price: {row.get('list_price')}\n"
            f"Beds: {row.get('beds')}\n"
            f"Full baths: {row.get('full_baths')}\n"
            f"Half baths: {row.get('half_baths')}\n"
            f"Sqft: {row.get('sqft')}\n"
            f"Year built: {row.get('year_built')}\n"
            f"HOA fee: {row.get('hoa_fee')}\n"
            f"Price per sqft: {row.get('price_per_sqft')}\n"
            f"Days on market: {row.get('days_on_mls')}\n"
            f"Nearby schools: {row.get('nearby_schools')}\n"
            f"Listing text: {row.get('text')}\n"
            f"Deterministic score reason: {row.get('score_reason')}\n"
            f"Red flags: {row.get('red_flags')}\n"
        )

        try:
            response = client.responses.create(
                model=config.openai_model,
                input=prompt,
            )
            parsed = _extract_json(response.output_text)
        except Exception as exc:
            logger.warning("OpenAI summary failed for %s: %s", row.get("property_url"), exc)
            continue

        enriched.at[index, "llm_summary"] = parsed["summary"] or None
        enriched.at[index, "llm_criteria_match"] = parsed["criteria_match"] or None
        enriched.at[index, "llm_possible_concern"] = parsed["possible_concern"] or None

    return enriched

