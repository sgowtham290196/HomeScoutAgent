from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


_STATE_TOKEN_RE = re.compile(r"^[A-Za-z]{2,20}$")
_CRITERIA_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "good",
    "home",
    "homes",
    "in",
    "is",
    "it",
    "low",
    "not",
    "of",
    "or",
    "prefer",
    "potential",
    "price",
    "relative",
    "safe",
    "the",
    "to",
    "value",
    "with",
}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _parse_float(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_csv(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    parts = re.split(r"[,\n;|]+", str(value))
    return [part.strip() for part in parts if part.strip()]


def _parse_locations(value: Any) -> list[str]:
    if value is None or value == "":
        return []

    raw_value = str(value).strip()
    if ";" in raw_value or "\n" in raw_value or "|" in raw_value:
        return [part.strip() for part in re.split(r"[;\n|]+", raw_value) if part.strip()]

    pattern = r"[^,;|]+,\s*[^,;|]+(?:\s+\d{5})?"
    matches = [match.strip() for match in re.findall(pattern, raw_value)]
    if matches:
        return matches

    pieces = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if len(pieces) >= 2:
        grouped: list[str] = []
        index = 0
        while index < len(pieces):
            current = pieces[index]
            if index + 1 < len(pieces) and _STATE_TOKEN_RE.match(pieces[index + 1]):
                grouped.append(f"{current}, {pieces[index + 1]}")
                index += 2
            else:
                grouped.append(current)
                index += 1
        return grouped

    return pieces


def _criteria_terms(value: str | None) -> list[str]:
    if not value:
        return []
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z]{4,}", value)
        if token.lower() not in _CRITERIA_STOPWORDS
    }
    return sorted(terms)


class AgentConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    real_estate_locations: list[str]
    price_min: int
    price_max: int
    email_from: str
    email_to: str
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str

    listing_type: str = "for_sale"
    property_types: list[str] = Field(default_factory=list)
    beds_min: int | None = None
    beds_max: int | None = None
    baths_min: float | None = None
    baths_max: float | None = None
    sqft_min: int | None = None
    sqft_max: int | None = None
    lot_sqft_min: int | None = None
    lot_sqft_max: int | None = None
    year_built_min: int | None = None
    year_built_max: int | None = None
    hoa_max: int | None = None
    past_days: int = 7
    limit_per_location: int = 100
    top_n: int = 5
    subjective_criteria: str | None = None
    negative_keywords: list[str] = Field(default_factory=list)
    positive_keywords: list[str] = Field(default_factory=list)
    enable_openai_scoring: bool = False
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    dry_run: bool = False

    @field_validator("real_estate_locations")
    @classmethod
    def validate_locations(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("REAL_ESTATE_LOCATIONS must contain at least one location.")
        return value

    @field_validator("listing_type")
    @classmethod
    def normalize_listing_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("LISTING_TYPE cannot be empty.")
        return normalized

    @field_validator("property_types", "negative_keywords", "positive_keywords")
    @classmethod
    def normalize_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @model_validator(mode="after")
    def validate_ranges(self) -> "AgentConfig":
        if self.price_min > self.price_max:
            raise ValueError("PRICE_MIN cannot be greater than PRICE_MAX.")

        range_pairs = [
            ("beds_min", "beds_max"),
            ("baths_min", "baths_max"),
            ("sqft_min", "sqft_max"),
            ("lot_sqft_min", "lot_sqft_max"),
            ("year_built_min", "year_built_max"),
        ]
        for minimum_name, maximum_name in range_pairs:
            minimum_value = getattr(self, minimum_name)
            maximum_value = getattr(self, maximum_name)
            if minimum_value is not None and maximum_value is not None and minimum_value > maximum_value:
                raise ValueError(
                    f"{minimum_name.upper()} cannot be greater than {maximum_name.upper()}."
                )

        if self.smtp_port <= 0:
            raise ValueError("SMTP_PORT must be positive.")
        if self.limit_per_location <= 0:
            raise ValueError("LIMIT_PER_LOCATION must be positive.")
        if self.top_n <= 0:
            raise ValueError("TOP_N must be positive.")
        if self.past_days <= 0:
            raise ValueError("PAST_DAYS must be positive.")

        return self

    @property
    def locations_display(self) -> str:
        return ", ".join(self.real_estate_locations)

    @property
    def criteria_terms(self) -> list[str]:
        return _criteria_terms(self.subjective_criteria)


def _config_values(env: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "real_estate_locations": _parse_locations(env.get("REAL_ESTATE_LOCATIONS")),
        "price_min": _parse_int(env.get("PRICE_MIN"), "PRICE_MIN"),
        "price_max": _parse_int(env.get("PRICE_MAX"), "PRICE_MAX"),
        "email_from": env.get("EMAIL_FROM"),
        "email_to": env.get("EMAIL_TO"),
        "smtp_host": env.get("SMTP_HOST"),
        "smtp_port": _parse_int(env.get("SMTP_PORT"), "SMTP_PORT"),
        "smtp_username": env.get("SMTP_USERNAME"),
        "smtp_password": env.get("SMTP_PASSWORD"),
        "listing_type": env.get("LISTING_TYPE") or "for_sale",
        "property_types": _parse_csv(env.get("PROPERTY_TYPES")),
        "beds_min": _parse_int(env.get("BEDS_MIN"), "BEDS_MIN"),
        "beds_max": _parse_int(env.get("BEDS_MAX"), "BEDS_MAX"),
        "baths_min": _parse_float(env.get("BATHS_MIN"), "BATHS_MIN"),
        "baths_max": _parse_float(env.get("BATHS_MAX"), "BATHS_MAX"),
        "sqft_min": _parse_int(env.get("SQFT_MIN"), "SQFT_MIN"),
        "sqft_max": _parse_int(env.get("SQFT_MAX"), "SQFT_MAX"),
        "lot_sqft_min": _parse_int(env.get("LOT_SQFT_MIN"), "LOT_SQFT_MIN"),
        "lot_sqft_max": _parse_int(env.get("LOT_SQFT_MAX"), "LOT_SQFT_MAX"),
        "year_built_min": _parse_int(env.get("YEAR_BUILT_MIN"), "YEAR_BUILT_MIN"),
        "year_built_max": _parse_int(env.get("YEAR_BUILT_MAX"), "YEAR_BUILT_MAX"),
        "hoa_max": _parse_int(env.get("HOA_MAX"), "HOA_MAX"),
        "past_days": _parse_int(env.get("PAST_DAYS"), "PAST_DAYS") or 7,
        "limit_per_location": _parse_int(env.get("LIMIT_PER_LOCATION"), "LIMIT_PER_LOCATION") or 100,
        "top_n": _parse_int(env.get("TOP_N"), "TOP_N") or 5,
        "subjective_criteria": env.get("SUBJECTIVE_CRITERIA"),
        "negative_keywords": _parse_csv(env.get("NEGATIVE_KEYWORDS")),
        "positive_keywords": _parse_csv(env.get("POSITIVE_KEYWORDS")),
        "enable_openai_scoring": _parse_bool(env.get("ENABLE_OPENAI_SCORING"), default=False),
        "openai_api_key": env.get("OPENAI_API_KEY"),
        "openai_model": env.get("OPENAI_MODEL") or "gpt-4.1-mini",
        "dry_run": _parse_bool(env.get("DRY_RUN"), default=False),
    }


def load_config(env: Mapping[str, Any] | None = None, dotenv_path: str | os.PathLike[str] | None = None) -> AgentConfig:
    if env is None:
        env_file = Path(dotenv_path) if dotenv_path else Path(".env")
        load_dotenv(dotenv_path=env_file if env_file.exists() else None)
        source: Mapping[str, Any] = os.environ
    else:
        source = env

    try:
        return AgentConfig.model_validate(_config_values(source))
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to load agent configuration: {exc}") from exc
