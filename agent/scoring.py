from __future__ import annotations

from datetime import datetime
import math
from typing import Iterable

import pandas as pd

from agent.config import AgentConfig


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        raise TypeError("_safe_float expects scalar values.")
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str) and value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    converted = _safe_float(value)
    if converted is None:
        return None
    return int(converted)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _format_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f}"


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def _listing_corpus(row: pd.Series) -> str:
    fields = [
        row.get("text"),
        row.get("nearby_schools"),
        row.get("style"),
        row.get("formatted_address"),
        row.get("city"),
        row.get("state"),
    ]
    parts: list[str] = []
    for field in fields:
        if field is None:
            continue
        try:
            if pd.isna(field):
                continue
        except TypeError:
            pass
        if isinstance(field, str) and field == "":
            continue
        parts.append(str(field))
    return " ".join(parts).lower()


def _matches_in_text(text: str, phrases: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for phrase in phrases:
        clean_phrase = phrase.strip().lower()
        if clean_phrase and clean_phrase in text:
            matches.append(clean_phrase)
    return matches


def _score_price(list_price: float | None, config: AgentConfig) -> tuple[float, str | None]:
    if list_price is None:
        return 8.0, "price missing"

    if config.price_max == config.price_min:
        return 14.0, "price within target"

    relative = (list_price - config.price_min) / (config.price_max - config.price_min)
    relative = _clamp(relative, 0.0, 1.0)
    score = 20.0 - (relative * 12.0)
    reason = f"priced at {_format_currency(list_price)} within target range"
    return round(_clamp(score, 0.0, 20.0), 2), reason


def _score_price_per_sqft(price_per_sqft: float | None, median_ppsf: float | None) -> tuple[float, str | None]:
    if price_per_sqft is None or median_ppsf in (None, 0):
        return 7.0, "price per sqft unavailable"

    ratio = price_per_sqft / median_ppsf
    score = 7.5 + ((1 - ratio) * 7.5)
    reason = f"price/sqft {_format_currency(price_per_sqft)} vs median {_format_currency(median_ppsf)}"
    return round(_clamp(score, 0.0, 15.0), 2), reason


def _score_size(sqft: int | None, config: AgentConfig) -> tuple[float, str | None]:
    if sqft is None:
        return 4.0, "square footage missing"

    cap = max(config.sqft_min or 0, 1800)
    score = min(sqft, cap) / cap * 10.0
    reason = f"{_format_number(sqft)} sqft"
    return round(_clamp(score, 0.0, 10.0), 2), reason


def _score_beds_baths(beds: int | None, baths: float | None, config: AgentConfig) -> tuple[float, str | None]:
    score = 0.0
    notes: list[str] = []

    if config.beds_min is None:
        score += 5.0 if beds is not None else 2.5
    elif beds is not None:
        if beds >= config.beds_min:
            score += 5.0
            notes.append(f"{beds} beds meets target")
        else:
            score += max(0.0, 5.0 - (config.beds_min - beds) * 2.0)
            notes.append(f"{beds} beds below target")
    else:
        notes.append("beds missing")

    if config.baths_min is None:
        score += 5.0 if baths is not None else 2.5
    elif baths is not None:
        if baths >= config.baths_min:
            score += 5.0
            notes.append(f"{baths:g} baths meets target")
        else:
            score += max(0.0, 5.0 - (config.baths_min - baths) * 2.0)
            notes.append(f"{baths:g} baths below target")
    else:
        notes.append("baths missing")

    return round(_clamp(score, 0.0, 10.0), 2), ", ".join(notes) if notes else None


def _score_year_built(year_built: int | None) -> tuple[float, str | None]:
    if year_built is None:
        return 5.0, "year built missing"

    current_year = datetime.now().year
    age = current_year - year_built
    if age <= 10:
        score = 10.0
    elif age <= 25:
        score = 8.5
    elif age <= 40:
        score = 7.0
    elif age <= 60:
        score = 5.5
    else:
        score = 4.0

    return score, f"built in {year_built}"


def _score_hoa(hoa_fee: float | None, config: AgentConfig) -> tuple[float, str | None]:
    if hoa_fee is None:
        return 5.0, "HOA not listed"
    if hoa_fee <= 0:
        return 10.0, "no HOA fee"
    if config.hoa_max:
        ratio = min(hoa_fee / config.hoa_max, 1.5)
        score = 10.0 - (ratio * 6.0)
    else:
        score = 8.0 if hoa_fee <= 300 else 6.0 if hoa_fee <= 600 else 4.0
    return round(_clamp(score, 0.0, 10.0), 2), f"HOA {_format_currency(hoa_fee)}"


def _score_freshness(days_on_mls: int | None) -> tuple[float, str | None]:
    if days_on_mls is None:
        return 5.0, "days on market unavailable"
    if days_on_mls <= 7:
        score = 10.0
    elif days_on_mls <= 14:
        score = 8.0
    elif days_on_mls <= 30:
        score = 6.0
    elif days_on_mls <= 60:
        score = 4.0
    else:
        score = 2.0
    return score, f"{days_on_mls} days on market"


def _score_keywords(row: pd.Series, config: AgentConfig) -> tuple[float, list[str], list[str], str | None]:
    text = _listing_corpus(row)
    positive_matches = _matches_in_text(text, config.positive_keywords)
    negative_matches = _matches_in_text(text, config.negative_keywords)
    criteria_matches = _matches_in_text(text, config.criteria_terms)

    score = 0.0
    score += min(len(positive_matches) * 3.0, 9.0)
    score += min(len(criteria_matches) * 1.5, 6.0)
    score -= min(len(negative_matches) * 3.5, 10.5)

    notes: list[str] = []
    if positive_matches:
        notes.append(f"positive keywords: {', '.join(sorted(set(positive_matches)))}")
    if criteria_matches:
        notes.append(f"criteria overlap: {', '.join(sorted(set(criteria_matches)))}")
    if negative_matches:
        notes.append(f"negative keywords: {', '.join(sorted(set(negative_matches)))}")

    return (
        round(_clamp(score, -15.0, 15.0), 2),
        sorted(set(positive_matches + criteria_matches)),
        sorted(set(negative_matches)),
        "; ".join(notes) if notes else None,
    )


def score_properties(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    if df.empty:
        scored = df.copy()
        for column in ["score", "score_reason", "red_flags"]:
            scored[column] = pd.Series(dtype="object")
        return scored

    scored = df.copy()
    median_ppsf = _numeric_series(scored, "price_per_sqft").dropna().median()
    if isinstance(median_ppsf, float) and math.isnan(median_ppsf):
        median_ppsf = None

    records: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        list_price = _safe_float(row.get("list_price"))
        price_per_sqft = _safe_float(row.get("price_per_sqft"))
        sqft = _safe_int(row.get("sqft"))
        beds = _safe_int(row.get("beds"))
        full_baths = _safe_float(row.get("full_baths"))
        half_baths = _safe_float(row.get("half_baths"))
        baths = None
        if full_baths is not None:
            baths = full_baths + (0.5 * (half_baths or 0.0))
        year_built = _safe_int(row.get("year_built"))
        hoa_fee = _safe_float(row.get("hoa_fee"))
        days_on_mls = _safe_int(row.get("days_on_mls"))

        price_score, price_reason = _score_price(list_price, config)
        ppsf_score, ppsf_reason = _score_price_per_sqft(price_per_sqft, median_ppsf)
        size_score, size_reason = _score_size(sqft, config)
        beds_score, beds_reason = _score_beds_baths(beds, baths, config)
        age_score, age_reason = _score_year_built(year_built)
        hoa_score, hoa_reason = _score_hoa(hoa_fee, config)
        freshness_score, freshness_reason = _score_freshness(days_on_mls)
        keyword_score, good_matches, bad_matches, keyword_reason = _score_keywords(row, config)

        score = round(
            _clamp(
                price_score
                + ppsf_score
                + size_score
                + beds_score
                + age_score
                + hoa_score
                + freshness_score
                + keyword_score,
                0.0,
                100.0,
            ),
            2,
        )

        score_reasons = [
            reason
            for reason in [
                price_reason,
                ppsf_reason,
                size_reason,
                beds_reason,
                age_reason,
                hoa_reason,
                freshness_reason,
                keyword_reason,
            ]
            if reason
        ]

        red_flags: list[str] = []
        if list_price is None:
            red_flags.append("missing list price")
        if price_per_sqft is None:
            red_flags.append("missing price per sqft")
        if year_built is not None and year_built < 1970:
            red_flags.append("older home")
        if hoa_fee is not None and config.hoa_max is not None and hoa_fee > config.hoa_max * 0.8:
            red_flags.append("HOA near max budget")
        if days_on_mls is not None and days_on_mls > 45:
            red_flags.append("stale listing")
        if bad_matches:
            red_flags.append(f"listing mentions {', '.join(bad_matches)}")

        if not red_flags:
            red_flags.append("no major red flags found")

        records.append(
            {
                "score": score,
                "score_reason": "; ".join(score_reasons),
                "red_flags": "; ".join(red_flags),
                "_sort_price": list_price if list_price is not None else float("inf"),
            }
        )

    extras = pd.DataFrame(records, index=scored.index)
    scored = pd.concat([scored, extras], axis=1)
    return scored.drop(columns=["_sort_price"], errors="ignore")


def rank_properties(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    scored = score_properties(df, config)
    if scored.empty:
        return scored

    priced = _numeric_series(scored, "list_price")
    days_on_mls = _numeric_series(scored, "days_on_mls")
    ranked = scored.assign(
        _sort_price=priced.fillna(float("inf")),
        _sort_days_on_mls=days_on_mls.fillna(float("inf")),
    ).sort_values(
        by=["score", "_sort_price", "_sort_days_on_mls"],
        ascending=[False, True, True],
        na_position="last",
    )
    return ranked.head(config.top_n).drop(
        columns=["_sort_price", "_sort_days_on_mls"], errors="ignore"
    ).reset_index(drop=True)
