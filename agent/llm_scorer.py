from __future__ import annotations

import logging

import pandas as pd
from pydantic import BaseModel, Field

from agent.config import AgentConfig
from agent.langchain_models import create_chat_model

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


class FieldAssessment(BaseModel):
    score: float | None = Field(default=None, description="Numeric score from 0 to 10.")
    comment: str = Field(default="", description="Brief supporting comment.")


class ListingAssessment(BaseModel):
    summary: str = ""
    criteria_match: str = ""
    possible_concern: str = ""
    research_sources: list[str] = Field(default_factory=list)
    safety: FieldAssessment = Field(default_factory=FieldAssessment)
    neighborhood: FieldAssessment = Field(default_factory=FieldAssessment)
    appreciation: FieldAssessment = Field(default_factory=FieldAssessment)
    schools: FieldAssessment = Field(default_factory=FieldAssessment)
    commute: FieldAssessment = Field(default_factory=FieldAssessment)
    value: FieldAssessment = Field(default_factory=FieldAssessment)
    condition: FieldAssessment = Field(default_factory=FieldAssessment)
    risk: FieldAssessment = Field(default_factory=FieldAssessment)


class FinalistAssessmentBatch(BaseModel):
    email_intro: str = ""
    listings: list[ListingAssessment] = Field(default_factory=list)


def llm_assessment_columns() -> list[str]:
    columns = ["llm_summary", "llm_criteria_match", "llm_possible_concern", "llm_research_sources"]
    for field in LLM_ASSESSMENT_FIELDS:
        columns.extend([f"llm_{field}_score", f"llm_{field}_comment"])
    return columns


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
    del config, with_web_search
    return client.with_structured_output(FinalistAssessmentBatch).invoke(prompt)


def _listing_field_assessment(listing: ListingAssessment, field: str) -> FieldAssessment:
    assessment = getattr(listing, field, None)
    if isinstance(assessment, FieldAssessment):
        return assessment
    return FieldAssessment()


def enrich_finalists_with_llm(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    enriched = df.copy()
    enriched = _ensure_llm_columns(enriched)
    enriched.attrs["llm_email_intro"] = None

    if enriched.empty:
        return enriched

    if not config.enable_openai_scoring:
        logger.info("LangChain scoring disabled; skipping qualitative summaries.")
        return enriched

    if not config.langchain_api_key:
        logger.warning("ENABLE_OPENAI_SCORING is true but no LangChain provider API key is configured; skipping.")
        return enriched

    try:
        client = create_chat_model(config)
    except Exception as exc:
        logger.warning("LangChain chat model initialization failed: %s", exc)
        return enriched

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
        "For safety, neighborhood quality, and appreciation, use verifiable public context when you have it. "
        "Use sources such as local crime/safety pages, city or police data, school/neighborhood pages, Redfin/Zillow/Realtor market pages, "
        "or comparable public housing-market references. Avoid inventing exact statistics if you cannot verify them.\n"
        "Do not rerank the listings and do not repeat the full email.\n"
        "Return one structured item per listing in the same order.\n"
        "Each listing must include summary, criteria match, possible concern, research sources, and field assessments.\n"
        "Field assessments must contain exactly these keys: "
        f"{', '.join(LLM_ASSESSMENT_FIELDS)}.\n"
        "Each field assessment must have numeric score from 0 to 10 and a brief comment. "
        "For appreciation, judge typical neighborhood/city appreciation over the last few years, not just the listing price. "
        "For risk, 10 means low risk and 0 means high risk.\n\n"
        f"Subjective criteria: {config.subjective_criteria or 'None provided'}\n\n"
        + "\n\n".join(finalist_lines)
    )

    try:
        parsed = _responses_create(
            client,
            config,
            prompt,
            with_web_search=config.enable_openai_web_search,
        )
    except Exception as exc:
        logger.warning("LangChain finalist enrichment failed: %s", exc)
        return enriched

    if isinstance(parsed, dict):
        parsed = FinalistAssessmentBatch.model_validate(parsed)
    if not isinstance(parsed, FinalistAssessmentBatch):
        logger.warning("LangChain finalist enrichment returned an unexpected payload; skipping.")
        return enriched

    enriched.attrs["llm_email_intro"] = parsed.email_intro.strip() or None
    listings = parsed.listings
    for row_position, (_, row) in enumerate(enriched.iterrows()):
        if row_position >= len(listings):
            continue
        listing = listings[row_position]
        enriched.at[row.name, "llm_summary"] = listing.summary.strip() or None
        enriched.at[row.name, "llm_criteria_match"] = listing.criteria_match.strip() or None
        enriched.at[row.name, "llm_possible_concern"] = listing.possible_concern.strip() or None
        enriched.at[row.name, "llm_research_sources"] = _format_sources(listing.research_sources)

        for field in LLM_ASSESSMENT_FIELDS:
            assessment = _listing_field_assessment(listing, field)
            enriched.at[row.name, f"llm_{field}_score"] = _coerce_score(assessment.score)
            comment = assessment.comment.strip()
            enriched.at[row.name, f"llm_{field}_comment"] = comment or None

    return enriched
