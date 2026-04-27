# HomeScoutAgent

HomeScoutAgent is a simple daily real estate email agent built on top of the existing HomeHarvest scraper.

It searches Realtor.com listings through `homeharvest.scrape_property()`, applies hard filters from `.env`, ranks the best matches with transparent deterministic scoring, and sends a daily email with the top picks.

## What This Project Does

- searches multiple locations every day
- filters by price, beds, baths, sqft, lot size, year built, property type, and HOA
- deduplicates overlapping listings across nearby cities
- scores listings from `0` to `100`
- explains each score with `score_reason`
- highlights issues with `red_flags`
- optionally uses one compact LLM pass for finalist subjective-fit notes and a short email intro
- sends a plain text and HTML email, or prints the result in dry-run mode

## Project Structure

```text
agent/
  config.py        Load and validate environment variables
  fetcher.py       Fetch listings with HomeHarvest and deduplicate them
  scoring.py       Deterministic scoring and ranking
  llm_scorer.py    Optional OpenAI finalist summaries
  emailer.py       HTML/plain text email rendering and SMTP sending
  main.py          Main entrypoint

homeharvest/       Existing scraping library
tests/             Tests for scraper behavior and the new agent layer
```

## Requirements

- Python 3.9+
- SMTP account for sending email
- optional OpenAI API key if you want finalist summaries

## Installation

```bash
python3 -m pip install -e .
```

If you prefer a non-editable install:

```bash
python3 -m pip install .
```

## Configuration

Copy [`.env.example`](</Users/goth/Desktop/Projects /Bay_Area_Housing_Analysis/.env.example>) to `.env` and fill it in.

The real `.env` file is ignored by Git. The example file stays committed.

### Required Variables

- `REAL_ESTATE_LOCATIONS`
- `PRICE_MIN`
- `PRICE_MAX`
- `EMAIL_FROM`
- `EMAIL_TO`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`

`EMAIL_TO` can contain multiple recipients separated by commas.

### Optional Variables

- `LISTING_TYPE`
- `PROPERTY_TYPES`
- `BEDS_MIN`, `BEDS_MAX`
- `BATHS_MIN`, `BATHS_MAX`
- `SQFT_MIN`, `SQFT_MAX`
- `LOT_SQFT_MIN`, `LOT_SQFT_MAX`
- `YEAR_BUILT_MIN`, `YEAR_BUILT_MAX`
- `HOA_MAX`
- `PAST_DAYS`
- `LIMIT_PER_LOCATION`
- `TOP_N`
- `SUBJECTIVE_CRITERIA`
- `POSITIVE_KEYWORDS`
- `NEGATIVE_KEYWORDS`
- `ENABLE_OPENAI_SCORING`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `SCHEDULE_TIME`
- `UPDATE_FREQUENCY`
- `DRY_RUN`

## Example `.env`

```env
REAL_ESTATE_LOCATIONS=Santa Clara, CA;Sunnyvale, CA;Mountain View, CA;Cupertino, CA
LISTING_TYPE=for_sale

PRICE_MIN=700000
PRICE_MAX=1200000

PROPERTY_TYPES=single_family,condos,townhomes
BEDS_MIN=2
BATHS_MIN=2
SQFT_MIN=900
YEAR_BUILT_MIN=1970
HOA_MAX=600

PAST_DAYS=7
LIMIT_PER_LOCATION=100
TOP_N=5

SUBJECTIVE_CRITERIA=Prefer homes with good resale potential, low HOA, safe neighborhood, good schools, reasonable commute to San Jose or Santa Clara, newer or remodeled condition, and strong value relative to price per square foot.
POSITIVE_KEYWORDS=remodeled,updated,excellent schools,quiet,new roof,solar,corner lot,move-in ready
NEGATIVE_KEYWORDS=fixer,TLC,as-is,auction,needs work,fire damage,foundation

ENABLE_OPENAI_SCORING=false
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
SCHEDULE_TIME=17:00
UPDATE_FREQUENCY=daily

EMAIL_FROM=your_email@gmail.com
EMAIL_TO=your_email@gmail.com,partner@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password

DRY_RUN=true
```

## How To Use

### Run locally

```bash
python -m agent.main
```

or:

```bash
python3 -m agent.main
```

### Run in dry mode

```bash
DRY_RUN=true python -m agent.main
```

Dry mode prints the full email content instead of sending it.

### Configure local schedule

The built-in local scheduler uses:

- `SCHEDULE_TIME=17:00` by default
- `UPDATE_FREQUENCY=daily` by default

Supported frequencies:

- `daily`
- `hourly`

For `daily`, `SCHEDULE_TIME` is the 24-hour send time.
For `hourly`, the minute portion of `SCHEDULE_TIME` is used.

This local scheduler only works while your machine is awake and the process is still running.

### Run on Windows

Use the batch launcher:

```bat
run_daily_agent.bat
```

This sends one email immediately, then keeps the agent running and sends again on the configured schedule.

### Run on macOS or Linux

Use the clickable Finder launcher:

```bash
run_daily_agent.command
```

This sends one email immediately, then keeps the agent running and sends again on the configured schedule.

### Stop the local scheduler

Windows:

```bat
stop_daily_agent.bat
```

macOS from Finder:

```bash
stop_daily_agent.command
```

## How Scoring Works

The agent does not hide ranking inside an LLM.

Each property is scored deterministically using:

- affordability within your price range
- price per sqft versus the fetched set median
- home size
- beds and baths fit
- year built
- HOA
- days on market
- positive keyword matches
- negative keyword penalties
- overlap with your subjective criteria text

Each ranked property includes:

- `score`
- `score_reason`
- `red_flags`

## Email Output

Each daily email includes:

- score
- address
- price
- beds, baths, sqft
- year built
- HOA
- price per sqft
- days on market
- listing URL
- reason for ranking
- possible concerns
- primary photo when available
- supports multiple recipients via comma-separated `EMAIL_TO`

## Optional OpenAI Summaries

If you set:

```env
ENABLE_OPENAI_SCORING=true
OPENAI_API_KEY=your_openai_api_key
```

the agent will make one low-cost LLM call for the already-ranked finalists. That call is only used to:

- look at subjective criteria fit for the finalists
- draft a short intro paragraph for the email

The OpenAI step is optional and does not control ranking order. Deterministic scoring remains the primary ranking method.

## Scheduling

### GitHub Actions

This repo includes:

- [`.github/workflows/daily-real-estate-agent.yml`](</Users/goth/Desktop/Projects /Bay_Area_Housing_Analysis/.github/workflows/daily-real-estate-agent.yml>)

It supports:

- daily cloud runs that do not depend on your Mac being open
- manual runs with `workflow_dispatch`

Store your configuration values in GitHub Actions Secrets.

Recommended always-on setup:

1. push this repo to GitHub
2. open your repo `Settings` -> `Secrets and variables` -> `Actions`
3. add the same values from `.env` as repository secrets
4. set `DRY_RUN=false`
5. let GitHub Actions run the workflow daily in the cloud

Default workflow timing:

- the workflow is set to `00:00 UTC`
- that is `5:00 PM Pacific` during daylight time
- during standard time it will run at `4:00 PM Pacific`

Important:

- `SCHEDULE_TIME` and `UPDATE_FREQUENCY` are for the local launcher-based scheduler
- the GitHub Actions schedule is controlled by the workflow cron expression, not `.env`

### Local cron example

```cron
0 7 * * * cd /path/to/HomeScoutAgent && /usr/bin/env python -m agent.main >> real-estate-agent.log 2>&1
```

## SMTP Notes

For Gmail:

1. enable 2-Step Verification
2. create a Gmail App Password
3. use that App Password as `SMTP_PASSWORD`
4. set `SMTP_HOST=smtp.gmail.com`
5. set `SMTP_PORT=587`

## Testing

Run the agent tests with:

```bash
python3 -m pytest tests/test_agent.py
```

## Compliance

This project uses the existing HomeHarvest Realtor.com scraping interface as-is.

- it does not scrape Zillow directly
- it does not bypass anti-bot protections
- users are responsible for complying with the terms of the source websites and email providers they use

## Library Note

The original `homeharvest.scrape_property()` behavior is preserved. The new agent is a separate application layer under `agent/`.
