from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent.config import load_config
from agent.emailer import build_email_message, render_html_email, render_text_email, send_email
from agent.fetcher import deduplicate_properties, fetch_properties
from agent.scoring import rank_properties, score_properties


def sample_env(**overrides: str) -> dict[str, str]:
    env = {
        "REAL_ESTATE_LOCATIONS": "Santa Clara, CA;Sunnyvale, CA;Mountain View, CA",
        "PRICE_MIN": "700000",
        "PRICE_MAX": "1200000",
        "EMAIL_FROM": "from@example.com",
        "EMAIL_TO": "to@example.com",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "user",
        "SMTP_PASSWORD": "pass",
        "PROPERTY_TYPES": "single_family,condos,townhomes",
        "BEDS_MIN": "2",
        "BATHS_MIN": "2",
        "SQFT_MIN": "900",
        "YEAR_BUILT_MIN": "1970",
        "HOA_MAX": "600",
        "PAST_DAYS": "7",
        "LIMIT_PER_LOCATION": "100",
        "TOP_N": "2",
        "SUBJECTIVE_CRITERIA": "Prefer remodeled homes with excellent schools and good resale value.",
        "POSITIVE_KEYWORDS": "remodeled,excellent schools,quiet",
        "NEGATIVE_KEYWORDS": "fixer,as-is,TLC",
        "ENABLE_OPENAI_SCORING": "false",
        "OPENAI_MODEL": "gpt-4.1-mini",
        "DRY_RUN": "true",
    }
    env.update(overrides)
    return env


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "property_id": "1",
                "property_url": "https://example.com/1",
                "formatted_address": "123 Main St, Santa Clara, CA 95050",
                "city": "Santa Clara",
                "state": "CA",
                "beds": 3,
                "full_baths": 2,
                "half_baths": 0,
                "sqft": 1500,
                "lot_sqft": 5000,
                "year_built": 1998,
                "list_price": 900000,
                "days_on_mls": 6,
                "hoa_fee": 250,
                "price_per_sqft": 600,
                "nearby_schools": "excellent schools",
                "text": "Move-in ready remodeled home on a quiet street.",
                "primary_photo": "https://example.com/photo1.jpg",
            },
            {
                "property_id": "2",
                "property_url": "https://example.com/2",
                "formatted_address": "456 Oak Ave, Sunnyvale, CA 94086",
                "city": "Sunnyvale",
                "state": "CA",
                "beds": 2,
                "full_baths": 2,
                "half_baths": 1,
                "sqft": 1100,
                "lot_sqft": 1800,
                "year_built": 1975,
                "list_price": 1150000,
                "days_on_mls": 35,
                "hoa_fee": 590,
                "price_per_sqft": 850,
                "nearby_schools": "good schools",
                "text": "Clean townhouse but sold as-is and needs some TLC.",
                "primary_photo": None,
            },
            {
                "property_id": "3",
                "property_url": "https://example.com/3",
                "formatted_address": "789 Pine Rd, Mountain View, CA 94040",
                "city": "Mountain View",
                "state": "CA",
                "beds": 4,
                "full_baths": 3,
                "half_baths": 0,
                "sqft": 1750,
                "lot_sqft": 6200,
                "year_built": 2008,
                "list_price": 980000,
                "days_on_mls": 12,
                "hoa_fee": None,
                "price_per_sqft": 560,
                "nearby_schools": "excellent schools",
                "text": "Updated kitchen with quiet backyard and strong resale appeal.",
                "primary_photo": "https://example.com/photo3.jpg",
            },
        ]
    )


def test_config_parsing_supports_semicolon_locations() -> None:
    config = load_config(sample_env())
    assert config.real_estate_locations == [
        "Santa Clara, CA",
        "Sunnyvale, CA",
        "Mountain View, CA",
    ]
    assert config.property_types == ["single_family", "condos", "townhomes"]
    assert config.top_n == 2
    assert config.dry_run is True


def test_deduplication_prefers_unique_properties() -> None:
    df = pd.DataFrame(
        [
            {"property_id": "1", "property_url": "https://example.com/1", "city": "Santa Clara"},
            {"property_id": "1", "property_url": "https://example.com/1?dup", "city": "Sunnyvale"},
            {"property_id": None, "property_url": "https://example.com/2", "city": "Mountain View"},
            {"property_id": None, "property_url": "https://example.com/2", "city": "Cupertino"},
        ]
    )

    deduped = deduplicate_properties(df)
    assert len(deduped) == 2


def test_fetch_properties_uses_scrape_property_mock(monkeypatch) -> None:
    config = load_config(sample_env())

    def fake_scrape_property(**kwargs):
        location = kwargs["location"]
        return pd.DataFrame(
            [
                {
                    "property_id": "shared-1",
                    "property_url": "https://example.com/shared-1",
                    "formatted_address": f"Shared Home, {location}",
                    "hoa_fee": 300,
                },
                {
                    "property_id": f"unique-{location}",
                    "property_url": f"https://example.com/{location.replace(' ', '-').replace(',', '').lower()}",
                    "formatted_address": f"Unique Home, {location}",
                    "hoa_fee": 450,
                },
            ]
        )

    monkeypatch.setattr("agent.fetcher.scrape_property", fake_scrape_property)

    raw_df, deduped_df, filtered_df = fetch_properties(config)

    assert len(raw_df) == 6
    assert len(deduped_df) == 4
    assert len(filtered_df) == 4


def test_scoring_adds_reason_and_red_flags() -> None:
    config = load_config(sample_env())
    scored = score_properties(sample_dataframe(), config)

    assert "score" in scored.columns
    assert "score_reason" in scored.columns
    assert "red_flags" in scored.columns
    assert scored["score"].between(0, 100).all()
    assert scored.loc[0, "score"] > scored.loc[1, "score"]


def test_negative_keywords_reduce_score() -> None:
    config = load_config(sample_env(NEGATIVE_KEYWORDS="fixer,as-is", POSITIVE_KEYWORDS=""))
    df = pd.DataFrame(
        [
            {
                "formatted_address": "Clean Home",
                "list_price": 900000,
                "price_per_sqft": 600,
                "sqft": 1200,
                "year_built": 1990,
                "days_on_mls": 5,
                "text": "well-kept home",
            },
            {
                "formatted_address": "Fixer Home",
                "list_price": 900000,
                "price_per_sqft": 600,
                "sqft": 1200,
                "year_built": 1990,
                "days_on_mls": 5,
                "text": "fixer sold as-is",
            },
        ]
    )

    scored = score_properties(df, config)
    assert scored.loc[0, "score"] > scored.loc[1, "score"]


def test_ranking_handles_missing_optional_columns() -> None:
    config = load_config(sample_env())
    df = pd.DataFrame([{"formatted_address": "123 Main St"}])

    ranked = rank_properties(df, config)

    assert len(ranked) == 1
    assert "score" in ranked.columns


def test_ranking_returns_top_n() -> None:
    config = load_config(sample_env(TOP_N="2"))
    ranked = rank_properties(sample_dataframe(), config)

    assert len(ranked) == 2
    assert ranked.iloc[0]["property_id"] in {"1", "3"}
    assert ranked.iloc[0]["score"] >= ranked.iloc[1]["score"]


def test_invalid_zero_values_are_not_silently_defaulted() -> None:
    with pytest.raises(ValueError, match="TOP_N must be positive"):
        load_config(sample_env(TOP_N="0"))

    with pytest.raises(ValueError, match="PAST_DAYS must be positive"):
        load_config(sample_env(PAST_DAYS="0"))


def test_email_rendering_does_not_crash() -> None:
    config = load_config(sample_env())
    ranked = rank_properties(sample_dataframe(), config)

    html_body = render_html_email(ranked, config)
    text_body = render_text_email(ranked, config)
    message = build_email_message(ranked, config)

    assert "Daily Real Estate Picks" in html_body
    assert "Why it ranked" in text_body
    assert message["Subject"].startswith("Daily Real Estate Picks")


def test_send_email_uses_smtp_when_not_dry_run(monkeypatch) -> None:
    config = load_config(sample_env(DRY_RUN="false"))
    ranked = rank_properties(sample_dataframe(), config)

    smtp_instance = MagicMock()
    smtp_context = MagicMock()
    smtp_context.__enter__.return_value = smtp_instance
    smtp_context.__exit__.return_value = False

    monkeypatch.setattr("agent.emailer.smtplib.SMTP", MagicMock(return_value=smtp_context))

    send_email(ranked, config)

    smtp_instance.starttls.assert_called_once()
    smtp_instance.login.assert_called_once_with(config.smtp_username, config.smtp_password)
    smtp_instance.send_message.assert_called_once()
