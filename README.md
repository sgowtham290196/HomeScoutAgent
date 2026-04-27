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
- optionally adds short OpenAI summaries for the finalists
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

EMAIL_FROM=your_email@gmail.com
EMAIL_TO=your_email@gmail.com
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

### Run on Windows

Use the batch launcher:

```bat
run_daily_agent.bat
```

### Run on macOS or Linux

Use the shell launcher:

```bash
./run_daily_agent.sh
```

If needed, make it executable once:

```bash
chmod +x run_daily_agent.sh
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

## Optional OpenAI Summaries

If you set:

```env
ENABLE_OPENAI_SCORING=true
OPENAI_API_KEY=your_openai_api_key
```

the agent will generate short qualitative summaries for the finalists only.

The OpenAI step is optional and does not control ranking order.

## Scheduling

### GitHub Actions

This repo includes:

- [`.github/workflows/daily-real-estate-agent.yml`](</Users/goth/Desktop/Projects /Bay_Area_Housing_Analysis/.github/workflows/daily-real-estate-agent.yml>)

It supports:

- daily cron runs
- manual runs with `workflow_dispatch`

Store your configuration values in GitHub Actions Secrets.

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
