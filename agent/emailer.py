from __future__ import annotations

import html
import logging
import smtplib
from email.message import EmailMessage

import pandas as pd

from agent.config import AgentConfig

logger = logging.getLogger(__name__)


def _display_text(value: object, default: str = "n/a") -> str:
    if value is None or value == "":
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return str(value)


def _format_currency(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "n/a"
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def build_subject(config: AgentConfig) -> str:
    cities = []
    for location in config.real_estate_locations:
        city = location.split(",")[0].strip()
        if city:
            cities.append(city)
    return f"Daily Real Estate Picks: Top {config.top_n} homes for {', '.join(cities)}"


def _search_criteria_lines(config: AgentConfig) -> list[str]:
    beds_baths = []
    if config.beds_min is not None or config.beds_max is not None:
        beds_baths.append(f"Beds {config.beds_min or 0}+")
    if config.baths_min is not None or config.baths_max is not None:
        beds_baths.append(f"Baths {config.baths_min or 0}+")

    lines = [
        f"Locations: {config.locations_display}",
        f"Price range: {_format_currency(config.price_min)} - {_format_currency(config.price_max)}",
    ]
    if beds_baths:
        lines.append("Beds/Baths: " + ", ".join(beds_baths))
    if config.subjective_criteria:
        lines.append(f"Subjective criteria: {config.subjective_criteria}")
    return lines


def render_text_email(df: pd.DataFrame, config: AgentConfig) -> str:
    email_intro = df.attrs.get("llm_email_intro")

    lines = [
        "Daily Real Estate Picks",
        "",
        "Search criteria:",
        *[f"- {line}" for line in _search_criteria_lines(config)],
        "",
    ]

    if email_intro:
        lines.extend([email_intro, ""])

    if df.empty:
        lines.append("No matching properties were found today.")
        return "\n".join(lines)

    for index, row in df.reset_index(drop=True).iterrows():
        lines.extend(
            [
                f"#{index + 1} - {_display_text(row.get('formatted_address'), 'Unknown address')}",
                f"Score: {_display_text(row.get('score', 0), '0')}/100",
                f"Price: {_format_currency(row.get('list_price'))}",
                (
                    "Beds/Baths/Sqft: "
                    f"{_display_text(row.get('beds'))} bd / "
                    f"{_display_text(row.get('full_baths'))} ba / "
                    f"{_format_number(row.get('sqft'))} sqft"
                ),
                f"Year built: {_display_text(row.get('year_built'))}",
                f"HOA: {_format_currency(row.get('hoa_fee'))}",
                f"Price/sqft: {_format_currency(row.get('price_per_sqft'))}",
                f"Days on market: {_display_text(row.get('days_on_mls'))}",
                f"Why it ranked: {_display_text(row.get('score_reason'))}",
                f"Possible concerns: {_display_text(row.get('red_flags'))}",
            ]
        )
        if row.get("llm_summary"):
            lines.append(f"Summary: {row.get('llm_summary')}")
        if row.get("llm_criteria_match"):
            lines.append(f"Criteria match: {row.get('llm_criteria_match')}")
        if row.get("llm_possible_concern"):
            lines.append(f"LLM concern: {row.get('llm_possible_concern')}")
        lines.append(f"Link: {_display_text(row.get('property_url'))}")
        lines.append("")

    return "\n".join(lines).strip()


def render_html_email(df: pd.DataFrame, config: AgentConfig) -> str:
    email_intro = df.attrs.get("llm_email_intro")
    criteria_items = "".join(
        f"<li>{html.escape(line)}</li>"
        for line in _search_criteria_lines(config)
    )

    cards: list[str] = []
    if df.empty:
        cards.append("<p>No matching properties were found today.</p>")
    else:
        for index, row in df.reset_index(drop=True).iterrows():
            photo_html = ""
            if row.get("primary_photo"):
                photo_html = (
                    f'<div style="margin:12px 0;"><img src="{html.escape(str(row.get("primary_photo")))}" '
                    'alt="Property photo" style="max-width:100%;height:auto;border-radius:6px;"></div>'
                )

            llm_html = ""
            if row.get("llm_summary"):
                llm_html += f"<p><strong>Summary:</strong> {html.escape(str(row.get('llm_summary')))}</p>"
            if row.get("llm_criteria_match"):
                llm_html += (
                    f"<p><strong>Criteria match:</strong> "
                    f"{html.escape(str(row.get('llm_criteria_match')))}</p>"
                )
            if row.get("llm_possible_concern"):
                llm_html += (
                    f"<p><strong>LLM concern:</strong> "
                    f"{html.escape(str(row.get('llm_possible_concern')))}</p>"
                )

            cards.append(
                f"""
                <div style="border:1px solid #d1d5db;padding:16px;margin:16px 0;border-radius:6px;">
                  <h2 style="margin:0 0 8px 0;font-size:20px;">#{index + 1} - {html.escape(_display_text(row.get('formatted_address'), 'Unknown address'))}</h2>
                  <p style="margin:4px 0;"><strong>Score:</strong> {html.escape(_display_text(row.get('score', 0), '0'))}/100</p>
                  <p style="margin:4px 0;"><strong>Price:</strong> {_format_currency(row.get('list_price'))}</p>
                  <p style="margin:4px 0;"><strong>Beds/Baths/Sqft:</strong> {html.escape(_display_text(row.get('beds')))} bd / {html.escape(_display_text(row.get('full_baths')))} ba / {_format_number(row.get('sqft'))} sqft</p>
                  <p style="margin:4px 0;"><strong>Year built:</strong> {html.escape(_display_text(row.get('year_built')))}</p>
                  <p style="margin:4px 0;"><strong>HOA:</strong> {_format_currency(row.get('hoa_fee'))}</p>
                  <p style="margin:4px 0;"><strong>Price/sqft:</strong> {_format_currency(row.get('price_per_sqft'))}</p>
                  <p style="margin:4px 0;"><strong>Days on market:</strong> {html.escape(_display_text(row.get('days_on_mls')))}</p>
                  <p style="margin:4px 0;"><strong>Why it ranked:</strong> {html.escape(_display_text(row.get('score_reason')))}</p>
                  <p style="margin:4px 0;"><strong>Possible concerns:</strong> {html.escape(_display_text(row.get('red_flags')))}</p>
                  {llm_html}
                  {photo_html}
                  <p style="margin:8px 0 0 0;"><a href="{html.escape(_display_text(row.get('property_url'), '#'))}">View listing</a></p>
                </div>
                """
            )

    return (
        "<html><body style=\"font-family:Arial,sans-serif;color:#111827;\">"
        "<h1 style=\"margin-bottom:4px;\">Daily Real Estate Picks</h1>"
        "<p style=\"margin-top:0;color:#4b5563;\">Top ranked homes from today's HomeHarvest run.</p>"
        "<h2 style=\"font-size:18px;\">Search criteria</h2>"
        f"<ul>{criteria_items}</ul>"
        + (
            f"<p style=\"margin:12px 0 16px 0;\">{html.escape(str(email_intro))}</p>"
            if email_intro
            else ""
        )
        + "".join(cards)
        + "</body></html>"
    )


def build_email_message(df: pd.DataFrame, config: AgentConfig) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = build_subject(config)
    message["From"] = config.email_from
    message["To"] = config.email_to_display

    text_body = render_text_email(df, config)
    html_body = render_html_email(df, config)
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    return message


def send_email(df: pd.DataFrame, config: AgentConfig) -> None:
    message = build_email_message(df, config)
    if config.dry_run:
        print(message["Subject"])
        print()
        print(message.get_body(preferencelist=("plain",)).get_content())
        print()
        print(message.get_body(preferencelist=("html",)).get_content())
        return

    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(config.smtp_username, config.smtp_password)
        smtp.send_message(message)

    logger.info("Email sent to %s", config.email_to_display)
