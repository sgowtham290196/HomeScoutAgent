from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import signal
import time

from agent.config import load_config
from agent.emailer import send_email
from agent.fetcher import fetch_properties
from agent.llm_scorer import enrich_finalists_with_llm
from agent.scoring import rank_properties

PID_FILE = Path(".agent_scheduler.pid")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def run_agent_once() -> None:
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


def _parse_schedule_parts(schedule_time: str) -> tuple[int, int]:
    hours, minutes = schedule_time.split(":")
    return int(hours), int(minutes)


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_pid_file() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


def _write_pid_file() -> None:
    logger = logging.getLogger(__name__)
    current_pid = os.getpid()

    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
        except ValueError:
            existing_pid = None

        if existing_pid and _is_process_running(existing_pid):
            raise RuntimeError(
                f"Scheduler appears to already be running with pid {existing_pid}. "
                f"Use the stop launcher first if needed."
            )
        _remove_pid_file()

    PID_FILE.write_text(f"{current_pid}\n")
    atexit.register(_remove_pid_file)
    logger.info("Wrote scheduler PID file at %s", PID_FILE.resolve())


def stop_scheduler_process() -> int:
    if not PID_FILE.exists():
        return 1

    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        _remove_pid_file()
        return 1

    if not _is_process_running(pid):
        _remove_pid_file()
        return 1

    os.kill(pid, signal.SIGTERM)
    return pid


def _next_run_time(config_schedule_time: str, update_frequency: str, now: datetime | None = None) -> datetime:
    current = now or datetime.now()
    hour_value, minute_value = _parse_schedule_parts(config_schedule_time)

    if update_frequency == "hourly":
        scheduled = current.replace(minute=minute_value, second=0, microsecond=0)
        if scheduled <= current:
            scheduled += timedelta(hours=1)
        return scheduled

    scheduled = current.replace(hour=hour_value, minute=minute_value, second=0, microsecond=0)
    if scheduled <= current:
        scheduled += timedelta(days=1)
    return scheduled


def run_scheduler() -> None:
    logger = logging.getLogger(__name__)
    config = load_config()
    _write_pid_file()
    logger.info(
        "Scheduler enabled with frequency=%s at %s",
        config.update_frequency,
        config.schedule_time,
    )

    while True:
        next_run = _next_run_time(config.schedule_time, config.update_frequency)
        logger.info("Next scheduled run at %s", next_run.strftime("%Y-%m-%d %H:%M:%S"))

        while True:
            remaining_seconds = (next_run - datetime.now()).total_seconds()
            if remaining_seconds <= 0:
                break
            time.sleep(min(remaining_seconds, 60))

        logger.info("Running scheduled real estate email job.")
        try:
            run_agent_once()
        except Exception:
            logger.exception("Scheduled run failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HomeScoutAgent real estate emailer.")
    parser.add_argument(
        "--run-now-and-schedule",
        action="store_true",
        help="Send immediately, then keep running on the configured schedule.",
    )
    parser.add_argument(
        "--stop-scheduler",
        action="store_true",
        help="Stop the local scheduler process started by the launcher.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)

    if args.stop_scheduler:
        stopped_pid = stop_scheduler_process()
        if stopped_pid == 1:
            logger.info("No running scheduler process was found.")
        else:
            logger.info("Sent stop signal to scheduler pid %s", stopped_pid)
        return

    if args.run_now_and_schedule:
        run_agent_once()
        run_scheduler()
        return

    run_agent_once()


if __name__ == "__main__":
    main()
