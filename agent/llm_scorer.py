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


def _extract_batch_json(text: str) -> dict:
    cleaned = text.strip()
    fenced_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(0)
    return json.loads(cleaned)


def enrich_finalists_with_llm(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    enriched = df.copy()
    for column in ["llm_summary", "llm_criteria_match", "llm_possible_concern"]:
        if column not in enriched.columns:
            enriched[column] = None
    enriched.attrs["llm_email_intro"] = None

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

    finalist_lines: list[str] = []
    for offset, (_, row) in enumerate(enriched.iterrows(), start=1):
        finalist_lines.append(
            "\n".join(
                [
                    f"Listing {offset}",
                    f"address: {row.get('formatted_address')}",
                    f"city: {row.get('city')}, {row.get('state')}",
                    f"price: {row.get('list_price')}",
                    f"beds: {row.get('beds')}",
                    f"full_baths: {row.get('full_baths')}",
                    f"half_baths: {row.get('half_baths')}",
                    f"sqft: {row.get('sqft')}",
                    f"year_built: {row.get('year_built')}",
                    f"hoa_fee: {row.get('hoa_fee')}",
                    f"price_per_sqft: {row.get('price_per_sqft')}",
                    f"days_on_mls: {row.get('days_on_mls')}",
                    f"nearby_schools: {row.get('nearby_schools')}",
                    f"text: {row.get('text')}",
                    f"deterministic_score_reason: {row.get('score_reason')}",
                    f"red_flags: {row.get('red_flags')}",
                ]
            )
        )

    prompt = (
        "You are helping with a low-cost real estate email workflow.\n"
        "The deterministic score has already ranked these finalists. "
        "Only use the subjective criteria to add concise qualitative notes and draft a short email intro.\n"
        "Do not rerank the listings and do not repeat the full email.\n"
        "Return strict JSON with keys email_intro and listings.\n"
        "email_intro must be one short paragraph.\n"
        "listings must be an array with one item per listing in the same order.\n"
        "Each listing item must have keys summary, criteria_match, possible_concern.\n\n"
        f"Subjective criteria: {config.subjective_criteria or 'None provided'}\n\n"
        + "\n\n".join(finalist_lines)
    )

    try:
        response = client.responses.create(
            model=config.openai_model,
            input=prompt,
        )
        parsed = _extract_batch_json(response.output_text)
    except Exception as exc:
        logger.warning("OpenAI finalist enrichment failed: %s", exc)
        return enriched

    enriched.attrs["llm_email_intro"] = str(parsed.get("email_intro", "")).strip() or None
    listings = parsed.get("listings", [])
    for row_position, (_, row) in enumerate(enriched.iterrows()):
        if row_position >= len(listings) or not isinstance(listings[row_position], dict):
            continue
        listing = listings[row_position]
        parsed_listing = {
            "summary": str(listing.get("summary", "")).strip(),
            "criteria_match": str(listing.get("criteria_match", "")).strip(),
            "possible_concern": str(listing.get("possible_concern", "")).strip(),
        }
        enriched.at[row.name, "llm_summary"] = parsed_listing["summary"] or None
        enriched.at[row.name, "llm_criteria_match"] = parsed_listing["criteria_match"] or None
        enriched.at[row.name, "llm_possible_concern"] = parsed_listing["possible_concern"] or None

    return enriched
