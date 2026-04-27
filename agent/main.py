from __future__ import annotations

import logging

from agent.config import load_config
from agent.emailer import send_email
from agent.fetcher import fetch_properties
from agent.llm_scorer import enrich_finalists_with_llm
from agent.scoring import rank_properties


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    config = load_config()
    logger.info("Loaded config for %s locations", len(config.real_estate_locations))

    raw_df, deduped_df, filtered_df = fetch_properties(config)
    logger.info("Fetched %s raw properties", len(raw_df))
    logger.info("Deduplicated to %s properties", len(deduped_df))
    logger.info("Filtered down to %s properties", len(filtered_df))

    ranked_df = rank_properties(filtered_df, config)
    enriched_df = enrich_finalists_with_llm(ranked_df, config)

    logger.info("Prepared top %s properties for email", len(enriched_df))
    send_email(enriched_df, config)


if __name__ == "__main__":
    main()

