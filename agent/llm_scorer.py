from __future__ import annotations

import json
import logging
import re

import pandas as pd

from agent.config import AgentConfig

logger = logging.getLogger(__name__)

LLM_ASSESSMENT_FIELDS = (
    "safety",
    "neighborhood",
    "appreciation",
    "schools",
    "commute",
    "value",
    "condition",
    "risk",
)


def llm_assessment_columns() -> list[str]:
    columns = ["llm_summary", "llm_criteria_match", "llm_possible_concern", "llm_research_sources"]
    for field in LLM_ASSESSMENT_FIELDS:
        columns.extend([f"llm_{field}_score", f"llm_{field}_comment"])
    return columns


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


def _coerce_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, round(score, 2)))


def _format_sources(value: object) -> str | None:
    if not value:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(cleaned) if cleaned else None
    return str(value).strip() or None


def _ensure_llm_columns(enriched: pd.DataFrame) -> pd.DataFrame:
    for column in llm_assessment_columns():
        if column not in enriched.columns:
            enriched[column] = None
    return enriched


def _responses_create(client: object, config: AgentConfig, prompt: str, *, with_web_search: bool) -> object:
    kwargs: dict[str, object] = {
        "model": config.openai_model,
        "input": prompt,
    }
    if with_web_search:
        kwargs["tools"] = [
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "region": "California",
                    "timezone": "America/Los_Angeles",
                },
            }
        ]
        kwargs["tool_choice"] = "auto"
    return client.responses.create(**kwargs)


def enrich_finalists_with_llm(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    enriched = df.copy()
    enriched = _ensure_llm_columns(enriched)
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
                    f"assigned_primary_school: {row.get('assigned_primary_school')} ({row.get('assigned_primary_school_rating')}/10)",
                    f"assigned_middle_school: {row.get('assigned_middle_school')} ({row.get('assigned_middle_school_rating')}/10)",
                    f"assigned_high_school: {row.get('assigned_high_school')} ({row.get('assigned_high_school_rating')}/10)",
                    f"style: {row.get('style')}",
                    f"lot_sqft: {row.get('lot_sqft')}",
                    f"text: {row.get('text')}",
                    f"deterministic_score_reason: {row.get('score_reason')}",
                    f"deterministic_score_breakdown: {row.get('score_breakdown')}",
                    f"red_flags: {row.get('red_flags')}",
                ]
            )
        )

    prompt = (
        "You are helping with a low-cost real estate email workflow.\n"
        "The deterministic score has already ranked these finalists. "
        "Use subjective criteria and public web context to add concise qualitative notes and a short email intro.\n"
        "For safety, neighborhood quality, and appreciation, ground comments in public web research when tools are available. "
        "Use sources such as local crime/safety pages, city or police data, school/neighborhood pages, Redfin/Zillow/Realtor market pages, "
        "or comparable public housing-market references. Avoid inventing exact statistics if you cannot verify them.\n"
        "Do not rerank the listings and do not repeat the full email.\n"
        "Return strict JSON with keys email_intro and listings.\n"
        "email_intro must be one short paragraph.\n"
        "listings must be an array with one item per listing in the same order.\n"
        "Each listing item must have keys summary, criteria_match, possible_concern, research_sources, and field_assessments.\n"
        "research_sources must be a short array of source names or URLs used for the location-level assessment.\n"
        "field_assessments must contain exactly these keys: "
        f"{', '.join(LLM_ASSESSMENT_FIELDS)}.\n"
        "Each field assessment must have numeric score from 0 to 10 and a brief comment. "
        "For appreciation, judge typical neighborhood/city appreciation over the last few years, not just the listing price. "
        "For risk, 10 means low risk and 0 means high risk.\n\n"
        f"Subjective criteria: {config.subjective_criteria or 'None provided'}\n\n"
        + "\n\n".join(finalist_lines)
    )

    try:
        response = _responses_create(
            client,
            config,
            prompt,
            with_web_search=config.enable_openai_web_search,
        )
        parsed = _extract_batch_json(response.output_text)
    except Exception as exc:
        if not config.enable_openai_web_search:
            logger.warning("OpenAI finalist enrichment failed: %s", exc)
            return enriched
        logger.warning("OpenAI finalist enrichment with web search failed: %s", exc)
        try:
            response = _responses_create(client, config, prompt, with_web_search=False)
            parsed = _extract_batch_json(response.output_text)
        except Exception as fallback_exc:
            logger.warning("OpenAI finalist enrichment fallback failed: %s", fallback_exc)
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
        enriched.at[row.name, "llm_research_sources"] = _format_sources(listing.get("research_sources"))

        field_assessments = listing.get("field_assessments", {})
        if not isinstance(field_assessments, dict):
            continue
        for field in LLM_ASSESSMENT_FIELDS:
            assessment = field_assessments.get(field, {})
            if not isinstance(assessment, dict):
                continue
            enriched.at[row.name, f"llm_{field}_score"] = _coerce_score(assessment.get("score"))
            comment = str(assessment.get("comment", "")).strip()
            enriched.at[row.name, f"llm_{field}_comment"] = comment or None

    return enriched
