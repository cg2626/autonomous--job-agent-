"""Unit tests for the vastness + better lanes (2026-10-29).

Covers: ATS board parsers (Greenhouse/Lever/Ashby/SmartRecruiters),
Adzuna normalization, fuzzy skill extraction, and fuzzy cross-source dedup.
All fixtures are synthetic; no network access.
"""

import json
import os

import pytest

from app.agents import sources as src
from app.agents.parser import extract_skills_from_text, extract_skills_fuzzy
from app.agents.crawler import MarketDiscoveryAgent, fuzzy_dedupe_jobs


# ---------------------------------------------------------------- ATS parsers

def test_parse_greenhouse():
    payload = {
        "jobs": [
            {
                "id": 123,
                "title": "SAP Fiori Developer",
                "absolute_url": "https://boards.greenhouse.io/example/jobs/123",
                "location": {"name": "Bengaluru, India"},
                "first_published": "2026-10-20T00:00:00Z",
            }
        ]
    }
    records = src._parse_greenhouse("example", payload)
    assert len(records) == 1
    job = records[0]
    assert job["title"] == "SAP Fiori Developer"
    assert job["url"].endswith("/123")
    assert "Bengaluru" in job["location"]
    assert job["remote"] is False


def test_parse_lever():
    payload = [
        {
            "id": "abc-1",
            "text": "SAP BTP Consultant",
            "hostedUrl": "https://jobs.lever.co/example/abc-1",
            "categories": {"department": "Engineering", "team": "SAP", "location": "Remote"},
            "workplaceType": "remote",
        }
    ]
    records = src._parse_lever("example", payload)
    assert len(records) == 1
    assert records[0]["remote"] is True
    assert "SAP" in records[0]["description"]


def test_parse_ashby_skips_unlisted():
    payload = {
        "jobs": [
            {
                "id": "1", "title": "SAP UI5 Developer", "isListed": True,
                "jobUrl": "https://jobs.ashbyhq.com/example/1",
                "location": "Hyderabad, India", "department": "Product",
                "isRemote": False, "publishedAt": "2026-10-21",
            },
            {
                "id": "2", "title": "Hidden Role", "isListed": False,
                "jobUrl": "https://jobs.ashbyhq.com/example/2", "location": "Nowhere",
            },
        ]
    }
    records = src._parse_ashby("example", payload)
    assert [r["title"] for r in records] == ["SAP UI5 Developer"]


def test_parse_smartrecruiters():
    payload = {
        "totalFound": 1,
        "content": [
            {
                "id": "744000117984002",
                "name": "SAP BTP CAPM Developer",
                "location": {"city": "Pune", "country": "India", "remote": False},
                "department": {"label": "Engineering"},
                "releasedDate": "2026-10-27",
            }
        ],
    }
    records = src._parse_smartrecruiters("AMOLTECHNOLOGIES", payload)
    assert len(records) == 1
    job = records[0]
    assert job["title"] == "SAP BTP CAPM Developer"
    assert "jobs.smartrecruiters.com/AMOLTECHNOLOGIES/744000117984002" in job["url"]
    assert "Pune" in job["location"]


def test_ats_keyword_prefilter():
    records = [
        {"title": "SAP Fiori Developer", "description": "Engineering"},
        {"title": "Office Assistant", "description": "Admin"},
    ]
    assert src.ATSBoardSource._matches_keywords(records[0], {"sap", "fiori"})
    assert not src.ATSBoardSource._matches_keywords(records[1], {"sap", "fiori"})


def test_ats_configured_boards_default_and_env(monkeypatch):
    monkeypatch.delenv("ATS_COMPANY_BOARDS", raising=False)
    assert src.ATSBoardSource().boards == [
        {"ats": "smartrecruiters", "company": "AMOLTECHNOLOGIES"}
    ]
    monkeypatch.setenv(
        "ATS_COMPANY_BOARDS",
        json.dumps([{"ats": "greenhouse", "company": "example"}, {"company": "bare-slug"}]),
    )
    assert src.ATSBoardSource().boards == [
        {"ats": "greenhouse", "company": "example"},
        {"ats": None, "company": "bare-slug"},
    ]


# ---------------------------------------------------------------- Adzuna

def test_adzuna_not_configured_without_keys(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    assert src.AdzunaSource().is_configured is False


def test_adzuna_country_codes():
    source = src.AdzunaSource(app_id="x", app_key="y")
    assert source.country_code("India") == "in"
    assert source.country_code("United States") == "us"
    assert source.country_code("Atlantis") is None


def test_adzuna_normalize():
    item = {
        "title": "SAP UI5 Developer",
        "company": {"display_name": "Example Ltd"},
        "redirect_url": "https://www.adzuna.in/land/ad/123",
        "location": {"display_name": "Bengaluru, India"},
        "description": "<p>Build <b>Fiori</b> apps.</p>",
        "created": "2026-10-28T10:00:00Z",
    }
    record = src.AdzunaSource._normalize(item)
    assert record["title"] == "SAP UI5 Developer"
    assert record["company"] == "Example Ltd"
    assert record["description"] == "Build Fiori apps."
    assert "Bengaluru" in record["location"]


# ---------------------------------------------------------------- Fuzzy skills

def test_fuzzy_skills_keeps_exact_behavior():
    text = "4 years of SAP Fiori and UI5 development with OData services."
    exact = set(extract_skills_from_text(text))
    fuzzy = set(extract_skills_fuzzy(text))
    assert exact <= fuzzy  # fuzzy only adds, never removes
    assert {"fiori", "ui5", "odata"} <= fuzzy


def test_fuzzy_skills_sap_variants():
    text = "Senior SAPUI5 developer, S/4 HANA implementation, OpenUI5 components."
    found = set(extract_skills_fuzzy(text))
    assert "ui5" in found  # via sapui5 / openui5 aliases
    assert "s/4hana" in found  # via "s/4 hana" alias
    assert "hana" in found


def test_fuzzy_skills_no_short_skill_noise():
    # Short skills stay exact-only: "go" must not appear from "google",
    # "bas" must not appear from "based".
    text = "We googled the answer based on good practice."
    found = set(extract_skills_fuzzy(text))
    assert "go" not in found
    assert "bas" not in found


# ---------------------------------------------------------------- Fuzzy dedup

def _job(title, company, relevance, location="Bengaluru"):
    return {
        "title": title, "company": company, "location": location,
        "relevance_score": relevance, "portal_url": f"https://example/{title}",
    }


def test_fuzzy_dedupe_keeps_best_copy():
    jobs = [
        _job("SAP Fiori Developer", "Amol Technologies Pvt Ltd", 2),
        _job("SAP Fiori Developer ", "Amol Technologies", 5),  # trailing space + suffix
        _job("SAP BTP Consultant", "Amol Technologies", 3),
    ]
    deduped = fuzzy_dedupe_jobs(jobs)
    assert len(deduped) == 2
    fiori = [j for j in deduped if "Fiori" in j["title"]]
    assert len(fiori) == 1 and fiori[0]["relevance_score"] == 5


def test_fuzzy_dedupe_distinct_roles_survive():
    jobs = [
        _job("SAP Fiori Developer", "Example Corp", 2),
        _job("SAP BTP Architect", "Example Corp", 2),
        _job("SAP Fiori Developer", "Other Corp", 2),
    ]
    assert len(fuzzy_dedupe_jobs(jobs)) == 3


# ---------------------------------------------------------------- Normalization

def test_normalize_ats_record_weights():
    agent = MarketDiscoveryAgent()
    record = {
        "title": "SAP BTP CAPM Developer",
        "company": "AMOLTECHNOLOGIES",
        "url": "https://jobs.smartrecruiters.com/AMOLTECHNOLOGIES/1",
        "location": "Pune, India",
        "description": "Engineering",
        "publication_date": "2026-10-27",
    }
    job = agent._normalize_job(record, "ATS/smartrecruiters", "https://ats")
    assert job is not None
    assert job["company"] == "AMOLTECHNOLOGIES"
    assert job["country"] == "India"
    assert "btp" in job["weights"] and "capm" in job["weights"]
