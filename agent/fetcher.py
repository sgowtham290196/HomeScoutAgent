from __future__ import annotations

import logging

import pandas as pd

from homeharvest import scrape_property

from agent.config import AgentConfig

logger = logging.getLogger(__name__)


def fetch_properties_by_location(config: AgentConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for location in config.real_estate_locations:
        logger.info("Fetching listings for %s", location)
        frame = scrape_property(
            location=location,
            listing_type=config.listing_type,
            property_type=config.property_types or None,
            price_min=config.price_min,
            price_max=config.price_max,
            beds_min=config.beds_min,
            beds_max=config.beds_max,
            baths_min=config.baths_min,
            baths_max=config.baths_max,
            sqft_min=config.sqft_min,
            sqft_max=config.sqft_max,
            lot_sqft_min=config.lot_sqft_min,
            lot_sqft_max=config.lot_sqft_max,
            year_built_min=config.year_built_min,
            year_built_max=config.year_built_max,
            past_days=config.past_days,
            limit=config.limit_per_location,
        )
        if frame is None or frame.empty:
            logger.info("No listings returned for %s", location)
            continue

        location_frame = frame.copy()
        location_frame["search_location"] = location
        frames.append(location_frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def deduplicate_properties(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    deduped = df.copy()
    deduped["dedupe_key"] = deduped.get("property_id")

    if "property_url" in deduped.columns:
        deduped["dedupe_key"] = deduped["dedupe_key"].fillna(deduped["property_url"])

    fallback_keys = pd.Series(deduped.index.astype(str), index=deduped.index)
    deduped["dedupe_key"] = deduped["dedupe_key"].fillna(fallback_keys)
    deduped = deduped.drop_duplicates(subset=["dedupe_key"], keep="first").drop(columns=["dedupe_key"])
    return deduped.reset_index(drop=True)


def apply_client_side_filters(df: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if config.hoa_max is not None and "hoa_fee" in filtered.columns:
        hoa_values = pd.to_numeric(filtered["hoa_fee"], errors="coerce")
        filtered = filtered[(hoa_values.isna()) | (hoa_values <= config.hoa_max)]

    return filtered.reset_index(drop=True)


def fetch_properties(config: AgentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = fetch_properties_by_location(config)
    deduped_df = deduplicate_properties(raw_df)
    filtered_df = apply_client_side_filters(deduped_df, config)
    return raw_df, deduped_df, filtered_df
