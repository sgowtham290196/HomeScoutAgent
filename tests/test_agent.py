from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent.config import load_config
from agent.emailer import build_email_message, render_html_email, render_text_email, send_email
from agent.main import SchedulerAlreadyRunning, _next_run_time, _write_pid_file, main
from agent.fetcher import deduplicate_properties, fetch_properties
from agent.scoring import rank_properties, score_properties
from agent.tracker import append_new_report_entries, report_columns
from homeharvest.core.scrapers import Scraper, ScraperInput
from homeharvest.core.scrapers.models import ListingType
from homeharvest.core.scrapers.realtor.processors import process_extra_property_details


def sample_env(**overrides: str) -> dict[str, str]:
    env = {
        "REAL_ESTATE_LOCATIONS": "Santa Clara, CA;Sunnyvale, CA;Mountain View, CA",
        "PRICE_MIN": "700000",
        "PRICE_MAX": "1200000",
        "EMAIL_FROM": "from@example.com",
        "EMAIL_TO": "to@example.com,team@example.com",
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
        "MIN_ASSIGNED_PRIMARY_SCHOOL_RATING": "8",
        "MIN_ASSIGNED_MIDDLE_SCHOOL_RATING": "8",
        "MIN_ASSIGNED_HIGH_SCHOOL_RATING": "8",
        "PAST_DAYS": "7",
        "LIMIT_PER_LOCATION": "100",
        "TOP_N": "2",
        "SUBJECTIVE_CRITERIA": "Prefer remodeled homes with excellent schools and good resale value.",
        "POSITIVE_KEYWORDS": "remodeled,excellent schools,quiet",
        "NEGATIVE_KEYWORDS": "fixer,as-is,TLC",
        "ENABLE_OPENAI_SCORING": "false",
        "ENABLE_OPENAI_WEB_SEARCH": "true",
        "OPENAI_MODEL": "gpt-4.1-mini",
        "REPORT_TRACKER_PATH": "reports/live_report_tracker.csv",
        "SCHEDULE_TIME": "17:00",
        "UPDATE_FREQUENCY": "daily",
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
                "assigned_primary_school": "North Elementary",
                "assigned_primary_school_rating": 9,
                "assigned_middle_school": "Central Middle",
                "assigned_middle_school_rating": 8,
                "assigned_high_school": "West High",
                "assigned_high_school_rating": 9,
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
                "assigned_primary_school": "South Elementary",
                "assigned_primary_school_rating": 8,
                "assigned_middle_school": "South Middle",
                "assigned_middle_school_rating": 8,
                "assigned_high_school": "South High",
                "assigned_high_school_rating": 8,
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
                "assigned_primary_school": "View Elementary",
                "assigned_primary_school_rating": 10,
                "assigned_middle_school": "View Middle",
                "assigned_middle_school_rating": 9,
                "assigned_high_school": "View High",
                "assigned_high_school_rating": 10,
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
    assert config.email_to == ["to@example.com", "team@example.com"]
    assert config.schedule_time == "17:00"
    assert config.update_frequency == "daily"
    assert config.top_n == 2
    assert config.min_assigned_primary_school_rating == 8
    assert config.min_assigned_middle_school_rating == 8
    assert config.min_assigned_high_school_rating == 8
    assert config.enable_openai_web_search is True
    assert config.report_tracker_path == "reports/live_report_tracker.csv"
    assert config.dry_run is True


def test_schedule_config_validation() -> None:
    with pytest.raises(ValueError, match="SCHEDULE_TIME must be in HH:MM 24-hour format"):
        load_config(sample_env(SCHEDULE_TIME="5PM"))

    with pytest.raises(ValueError, match="UPDATE_FREQUENCY must be either 'daily' or 'hourly'"):
        load_config(sample_env(UPDATE_FREQUENCY="weekly"))


def test_next_run_time_daily_and_hourly() -> None:
    daily_now = datetime(2026, 4, 26, 16, 30)
    assert _next_run_time("17:00", "daily", daily_now) == datetime(2026, 4, 26, 17, 0)

    hourly_now = datetime(2026, 4, 26, 16, 45)
    assert _next_run_time("17:15", "hourly", hourly_now) == datetime(2026, 4, 26, 17, 15)


def test_write_pid_file_reports_existing_running_scheduler(tmp_path, monkeypatch) -> None:
    pid_file = tmp_path / "scheduler.pid"
    pid_file.write_text("12345\n")
    monkeypatch.setattr("agent.main.PID_FILE", pid_file)
    monkeypatch.setattr("agent.main._is_process_running", lambda pid: True)

    with pytest.raises(SchedulerAlreadyRunning) as exc_info:
        _write_pid_file()

    assert exc_info.value.pid == 12345
    assert pid_file.read_text() == "12345\n"


def test_run_now_and_schedule_exits_cleanly_when_scheduler_running(monkeypatch) -> None:
    run_once = MagicMock()
    run_scheduler_mock = MagicMock()
    monkeypatch.setattr("sys.argv", ["agent.main", "--run-now-and-schedule"])
    monkeypatch.setattr("agent.main.load_config", MagicMock(return_value=load_config(sample_env())))
    monkeypatch.setattr("agent.main._write_pid_file", MagicMock(side_effect=SchedulerAlreadyRunning(12345)))
    monkeypatch.setattr("agent.main.run_agent_once", run_once)
    monkeypatch.setattr("agent.main.run_scheduler", run_scheduler_mock)

    main()

    run_once.assert_not_called()
    run_scheduler_mock.assert_not_called()


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
                    "assigned_primary_school_rating": 8,
                    "assigned_middle_school_rating": 8,
                    "assigned_high_school_rating": 8,
                },
                {
                    "property_id": f"unique-{location}",
                    "property_url": f"https://example.com/{location.replace(' ', '-').replace(',', '').lower()}",
                    "formatted_address": f"Unique Home, {location}",
                    "hoa_fee": 450,
                    "assigned_primary_school_rating": 9,
                    "assigned_middle_school_rating": 9,
                    "assigned_high_school_rating": 9,
                },
            ]
        )

    monkeypatch.setattr("agent.fetcher.scrape_property", fake_scrape_property)

    raw_df, deduped_df, filtered_df = fetch_properties(config)

    assert len(raw_df) == 6
    assert len(deduped_df) == 4
    assert len(filtered_df) == 4


def test_scraper_honors_extra_property_data_flag() -> None:
    scraper = Scraper(ScraperInput(location="Santa Clara, CA", listing_type=ListingType.FOR_SALE))
    disabled_scraper = Scraper(
        ScraperInput(location="Santa Clara, CA", listing_type=ListingType.FOR_SALE, extra_property_data=False)
    )

    assert scraper.extra_property_data is True
    assert disabled_scraper.extra_property_data is False


def test_process_extra_property_details_extracts_assigned_greatschools_ratings() -> None:
    details = process_extra_property_details(
        {
            "assignedSchools": {
                "schools": [
                    {
                        "name": "North Elementary",
                        "education_levels": ["elementary"],
                        "rating": 9,
                        "assigned": True,
                        "district": {"name": "North District"},
                    },
                    {
                        "name": "Central Middle",
                        "education_levels": ["middle"],
                        "rating": 8,
                        "assigned": True,
                        "district": {"name": "North District"},
                    },
                    {
                        "name": "West High",
                        "education_levels": ["high"],
                        "rating": 10,
                        "assigned": True,
                        "district": {"name": "North District"},
                    },
                ]
            },
            "taxHistory": [],
        }
    )

    assert details["assigned_primary_school"] == "North Elementary"
    assert details["assigned_primary_school_rating"] == 9
    assert details["assigned_middle_school_rating"] == 8
    assert details["assigned_high_school"] == "West High"
    assert "North Elementary (9/10)" in details["assigned_schools"]


def test_school_rating_filter_requires_all_assigned_school_levels() -> None:
    config = load_config(sample_env())
    df = sample_dataframe()
    df.loc[1, "assigned_middle_school_rating"] = 7
    df.loc[2, "assigned_high_school_rating"] = None

    from agent.fetcher import apply_client_side_filters

    filtered = apply_client_side_filters(df, config)

    assert filtered["property_id"].tolist() == ["1"]


def test_school_rating_filter_can_use_mid_alias() -> None:
    config = load_config(
        sample_env(
            MIN_ASSIGNED_MIDDLE_SCHOOL_RATING="",
            MIN_ASSIGNED_MID_SCHOOL_RATING="9",
        )
    )

    assert config.min_assigned_middle_school_rating == 9


def test_scoring_adds_reason_and_red_flags() -> None:
    config = load_config(sample_env())
    scored = score_properties(sample_dataframe(), config)

    assert "score" in scored.columns
    assert "score_reason" in scored.columns
    assert "red_flags" in scored.columns
    assert "score_breakdown" in scored.columns
    assert "detailed_analysis" in scored.columns
    assert scored["score"].between(0, 100).all()
    assert scored.loc[0, "score"] > scored.loc[1, "score"]
    assert "Strengths:" in scored.loc[0, "detailed_analysis"]


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


def test_ranking_handles_pandas_na_text_fields() -> None:
    config = load_config(sample_env())
    df = pd.DataFrame(
        [
            {
                "formatted_address": "123 Main St",
                "text": pd.NA,
                "nearby_schools": pd.NA,
                "style": pd.NA,
                "city": "Santa Clara",
                "state": "CA",
            }
        ]
    )

    ranked = rank_properties(df, config)

    assert len(ranked) == 1
    assert "score" in ranked.columns


def test_ranking_handles_pandas_na_numeric_fields() -> None:
    config = load_config(sample_env())
    df = pd.DataFrame(
        [
            {
                "formatted_address": "123 Main St",
                "list_price": 900000,
                "price_per_sqft": 650,
                "sqft": 1400,
                "full_baths": 2,
                "half_baths": pd.NA,
                "beds": 3,
                "year_built": 1995,
                "days_on_mls": 4,
            }
        ]
    )

    ranked = rank_properties(df, config)

    assert len(ranked) == 1
    assert "score" in ranked.columns


def test_ranking_returns_top_n() -> None:
    config = load_config(sample_env(TOP_N="2"))
    ranked = rank_properties(sample_dataframe(), config)

    assert len(ranked) == 2
    assert ranked.iloc[0]["property_id"] in {"1", "3"}
    assert ranked.iloc[0]["score"] >= ranked.iloc[1]["score"]


def test_report_tracker_skips_repeated_top_listings(tmp_path) -> None:
    tracker_path = tmp_path / "live_report_tracker.csv"
    config = load_config(sample_env(TOP_N="2", REPORT_TRACKER_PATH=str(tracker_path)))
    ranked = rank_properties(sample_dataframe(), config)
    ranked.loc[0, "llm_safety_score"] = 8
    ranked.loc[0, "llm_safety_comment"] = "Public sources indicate relatively favorable local safety."
    ranked.loc[0, "llm_appreciation_score"] = 7
    ranked.loc[0, "llm_appreciation_comment"] = "Area has shown steady buyer demand in recent years."

    first_add = append_new_report_entries(ranked, config, run_date=date(2026, 4, 29))
    second_add = append_new_report_entries(ranked, config, run_date=date(2026, 4, 30))

    stored = pd.read_csv(tracker_path)
    assert len(first_add) == 2
    assert second_add.empty
    assert len(stored) == 2
    assert stored.columns.tolist() == report_columns()
    assert stored.loc[0, "House"] == "123 Main St"
    assert stored.loc[0, "Address"] == "123 Main St, Santa Clara, CA 95050"
    assert stored.loc[0, "Overall Score"] != ""
    assert stored.loc[0, "Safety Score"] == "8/10"
    assert "favorable local safety" in stored.loc[0, "Safety Comment"]
    assert stored.columns[-1] == "Zillow Link"
    assert "zillow.com/homes/" in stored.loc[0, "Zillow Link"]
    assert "tracker_rank" not in stored.columns
    assert "property_id" not in stored.columns


def test_report_tracker_migrates_old_raw_tracker_shape(tmp_path) -> None:
    tracker_path = tmp_path / "live_report_tracker.csv"
    config = load_config(sample_env(TOP_N="1", REPORT_TRACKER_PATH=str(tracker_path)))
    ranked = rank_properties(sample_dataframe(), config)
    old_raw = ranked.head(1).copy()
    old_raw.to_csv(tracker_path, index=False)

    added = append_new_report_entries(ranked, config, run_date=date(2026, 4, 30))

    stored = pd.read_csv(tracker_path)
    assert added.empty
    assert stored.columns.tolist() == report_columns()
    assert len(stored) == 1
    assert stored.loc[0, "House"] == "123 Main St"
    assert "property_id" not in stored.columns


def test_report_tracker_ignores_raw_array_like_listing_fields(tmp_path) -> None:
    tracker_path = tmp_path / "live_report_tracker.csv"
    config = load_config(sample_env(TOP_N="1", REPORT_TRACKER_PATH=str(tracker_path)))
    ranked = rank_properties(sample_dataframe(), config)
    ranked["photo_urls"] = pd.Series([None] * len(ranked), dtype=object)
    ranked["tax_history"] = pd.Series([None] * len(ranked), dtype=object)
    ranked.at[0, "photo_urls"] = ["https://example.com/a.jpg", "https://example.com/b.jpg"]
    ranked.at[0, "tax_history"] = {"2025": 12000}

    added = append_new_report_entries(ranked, config, run_date=date(2026, 4, 29))

    stored = pd.read_csv(tracker_path)
    assert len(added) == 1
    assert "photo_urls" not in stored.columns
    assert "tax_history" not in stored.columns
    assert stored.columns.tolist() == report_columns()


def test_invalid_zero_values_are_not_silently_defaulted() -> None:
    with pytest.raises(ValueError, match="TOP_N must be positive"):
        load_config(sample_env(TOP_N="0"))

    with pytest.raises(ValueError, match="PAST_DAYS must be positive"):
        load_config(sample_env(PAST_DAYS="0"))

    with pytest.raises(ValueError, match="MIN_ASSIGNED_HIGH_SCHOOL_RATING must be between 1 and 10"):
        load_config(sample_env(MIN_ASSIGNED_HIGH_SCHOOL_RATING="11"))


def test_email_rendering_does_not_crash() -> None:
    config = load_config(sample_env())
    ranked = rank_properties(sample_dataframe(), config)
    ranked.loc[0, "llm_safety_score"] = 8
    ranked.loc[0, "llm_safety_comment"] = "Public sources indicate relatively favorable local safety."
    ranked.loc[0, "llm_appreciation_score"] = 7
    ranked.loc[0, "llm_appreciation_comment"] = "Area has shown steady buyer demand in recent years."
    ranked.loc[0, "llm_research_sources"] = "city safety page; housing market page"

    html_body = render_html_email(ranked, config)
    text_body = render_text_email(ranked, config)
    message = build_email_message(ranked, config)

    assert "Daily Real Estate Picks" in html_body
    assert "Why it ranked" in text_body
    assert "Assigned GreatSchools ratings" in text_body
    assert "Assigned schools" in text_body
    assert "Primary: North Elementary (9/10)" in text_body
    assert "Strengths:" in text_body
    assert "- above-average usable square footage" in text_body
    assert "Score breakdown" in html_body
    assert "<li>Price:" in html_body
    assert "<li>above-average usable square footage" in html_body
    assert "Detailed analysis:" not in html_body
    assert "LLM field scores" in text_body
    assert "Safety: 8/10" in html_body
    assert "Zillow" in text_body
    assert "zillow.com/homes/" in html_body
    assert message["Subject"].startswith("Daily Real Estate Picks")
    assert message["To"] == "to@example.com, team@example.com"


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
