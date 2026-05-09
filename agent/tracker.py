from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
import re
from urllib.parse import quote_plus

import pandas as pd

from agent.config import AgentConfig
from agent.llm_scorer import LLM_ASSESSMENT_FIELDS

logger = logging.getLogger(__name__)

CORE_REPORT_COLUMNS = [
    "House",
    "Address",
    "Overall Score",
    "Overall Comment",
]

ENDING_REPORT_COLUMNS = [
    "AI Summary",
    "Criteria Match",
    "Concern",
    "Research Sources",
    "Zillow Link",
]


def _has_value(value: object) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return False
        if not isinstance(missing, bool) and hasattr(missing, "all") and bool(missing.all()):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(value, str) and value.strip() == "")


def _display_value(value: object) -> str:
    if not _has_value(value):
        return ""
    return str(value).strip()


def _clean_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _format_score(value: object) -> str:
    if not _has_value(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _display_value(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _simplified_house_name(row: pd.Series) -> str:
    address = _display_value(row.get("formatted_address"))
    if address:
        return address.split(",", maxsplit=1)[0].strip()

    city = _display_value(row.get("city"))
    if city:
        return f"Home in {city}"

    property_id = _display_value(row.get("property_id"))
    if property_id:
        return f"Home {property_id}"

    return "Home"


def _full_address(row: pd.Series) -> str:
    address = _display_value(row.get("formatted_address"))
    if address:
        return address

    parts = [
        _display_value(row.get("street")),
        _display_value(row.get("city")),
        _display_value(row.get("state")),
        _display_value(row.get("zip_code")),
    ]
    return ", ".join(part for part in parts if part)


def _zillow_search_url(row: pd.Series) -> str:
    address = _full_address(row)
    if address:
        return f"https://www.zillow.com/homes/{quote_plus(address)}_rb/"

    property_url = _display_value(row.get("property_url"))
    if "zillow.com" in property_url:
        return property_url
    return ""


def _score_column_name(field: str) -> str:
    return field.replace("_", " ").title() + " Score"


def _comment_column_name(field: str) -> str:
    return field.replace("_", " ").title() + " Comment"


def report_columns() -> list[str]:
    columns = [*CORE_REPORT_COLUMNS]
    for field in LLM_ASSESSMENT_FIELDS:
        columns.extend([_score_column_name(field), _comment_column_name(field)])
    columns.extend(ENDING_REPORT_COLUMNS)
    return columns


def listing_tracker_key(row: pd.Series) -> str:
    zillow_url = _display_value(row.get("Zillow Link"))
    if zillow_url:
        return f"zillow:{_clean_key(zillow_url)}"

    address = _display_value(row.get("Address"))
    if address:
        return f"address:{_clean_key(address)}"

    house = _display_value(row.get("House"))
    if house:
        return f"house:{_clean_key(house)}"

    return f"row:{row.name}"


def _build_report_row(row: pd.Series) -> dict[str, object]:
    report_row: dict[str, object] = {
        "House": _simplified_house_name(row),
        "Address": _full_address(row),
        "Overall Score": (_format_score(row.get("score")) + "/100") if _format_score(row.get("score")) else "",
        "Overall Comment": _display_value(row.get("score_reason")),
    }

    for field in LLM_ASSESSMENT_FIELDS:
        score_text = _format_score(row.get(f"llm_{field}_score"))
        report_row[_score_column_name(field)] = f"{score_text}/10" if score_text else ""
        report_row[_comment_column_name(field)] = _display_value(row.get(f"llm_{field}_comment"))

    report_row.update(
        {
            "AI Summary": _display_value(row.get("llm_summary")),
            "Criteria Match": _display_value(row.get("llm_criteria_match")),
            "Concern": _display_value(row.get("llm_possible_concern") or row.get("red_flags")),
            "Research Sources": _display_value(row.get("llm_research_sources")),
            "Zillow Link": _zillow_search_url(row),
        }
    )
    return report_row


def _prepare_tracker_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_build_report_row(row) for _, row in df.reset_index(drop=True).iterrows()]
    return pd.DataFrame(rows, columns=report_columns())


def _read_existing_tracker(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=report_columns())
    try:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=report_columns())

    if existing.columns.tolist() == report_columns():
        existing.attrs["needs_rewrite"] = False
        return existing

    if "House" not in existing.columns and any(
        column in existing.columns for column in ("formatted_address", "score", "property_url")
    ):
        migrated = _prepare_tracker_rows(existing)
        migrated.attrs["needs_rewrite"] = True
        return migrated

    for column in report_columns():
        if column not in existing.columns:
            existing[column] = ""
    simplified = existing.reindex(columns=report_columns())
    simplified.attrs["needs_rewrite"] = True
    return simplified


def append_new_report_entries(
    df: pd.DataFrame,
    config: AgentConfig,
    *,
    run_date: date | None = None,
) -> pd.DataFrame:
    del run_date
    path = Path(config.report_tracker_path)
    existing = _read_existing_tracker(path)

    prepared = _prepare_tracker_rows(df.head(config.top_n))
    if prepared.empty:
        logger.info("No report tracker entries to append.")
        return prepared

    existing_keys = {listing_tracker_key(row) for _, row in existing.iterrows()}
    new_rows = prepared[~prepared.apply(listing_tracker_key, axis=1).isin(existing_keys)].copy()
    skipped_count = len(prepared) - len(new_rows)

    if new_rows.empty:
        if existing.attrs.get("needs_rewrite"):
            path.parent.mkdir(parents=True, exist_ok=True)
            existing.reindex(columns=report_columns()).to_csv(path, index=False)
        logger.info("Report tracker unchanged; %s top listings were already tracked.", skipped_count)
        return new_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    combined = combined.reindex(columns=report_columns())
    combined.to_csv(path, index=False)

    logger.info(
        "Added %s new listings to %s; skipped %s repeats.",
        len(new_rows),
        path,
        skipped_count,
    )
    return new_rows
