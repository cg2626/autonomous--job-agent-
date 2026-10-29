"""Extra job sources: ATS career-board APIs and the Adzuna API.

Vastness lane (added 2026-10-29). Two ToS-safe source families that extend
discovery far beyond the built-in board APIs:

1. ATS career boards (Greenhouse, Lever, Ashby, SmartRecruiters). Each ATS
   publishes a public JSON feed for every company job board — the same data
   the careers page shows — with no API key or login. Endpoint patterns and
   the unified-schema idea are adapted from axmaier/ats-jobs
   (MIT License, https://github.com/axmaier/ats-jobs).
2. Adzuna (https://developer.adzuna.com): official jobs API with a free tier
   (app_id/app_key), covering India (`in`) and 20+ other countries. Opt-in
   via the ADZUNA_APP_ID / ADZUNA_APP_KEY environment variables; the source
   is skipped gracefully when they are unset.

Both families use a polite User-Agent, per-request timeouts, and never touch
credentials or accounts. No auto-apply anywhere: these sources only
*discover* openings.

Boards are configured with the ATS_COMPANY_BOARDS environment variable as a
JSON array, e.g.:
    [{"ats": "smartrecruiters", "company": "AMOLTECHNOLOGIES"},
     {"ats": "greenhouse", "company": "example"}]
A bare company slug may be given without "ats" to try all four boards.
"""

import json
import os
import re
from typing import Any

from bs4 import BeautifulSoup

try:
    import httpx
except ImportError:  # Keep parsing usable in a minimal test environment.
    httpx = None

USER_AGENT = "TalentAlignAI/3.0 job-discovery"
ATS_NAMES = ("greenhouse", "lever", "ashby", "smartrecruiters")
LEVER_HOSTS = ("https://api.lever.co", "https://api.eu.lever.co")  # EU accounts 404 on the global host

# Boards verified live (HTTP 200 + non-empty) on 2026-10-29. Extend with
# ATS_COMPANY_BOARDS; never guess a slug — verify it returns jobs first.
DEFAULT_BOARDS = [
    {"ats": "smartrecruiters", "company": "AMOLTECHNOLOGIES"},
]


def _get_json(url: str, timeout: float = 10.0) -> Any | None:
    """Parsed JSON, or None on 404 / network failure (a dead board is not fatal)."""
    try:
        if httpx is not None:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            ) as client:
                response = client.get(url)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
        from urllib.request import Request, urlopen

        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _parse_greenhouse(company: str, data: dict) -> list[dict]:
    """Greenhouse boards-api payload -> normalized records."""
    out = []
    for job in data.get("jobs", []):
        location = (job.get("location") or {}).get("name") or ""
        out.append({
            "title": job.get("title") or "",
            "company": company,
            "url": job.get("absolute_url") or "",
            "location": location,
            "description": "",
            "publication_date": job.get("first_published") or job.get("updated_at") or "",
            "remote": bool(re.search("remote", location, re.I)),
        })
    return out


def _parse_lever(company: str, data: list) -> list[dict]:
    """Lever postings payload -> normalized records."""
    out = []
    for job in data or []:
        categories = job.get("categories") or {}
        location = categories.get("location") or ""
        out.append({
            "title": job.get("text") or "",
            "company": company,
            "url": job.get("hostedUrl") or job.get("applyUrl") or "",
            "location": location,
            "description": " ".join(
                part for part in (categories.get("department"), categories.get("team")) if part
            ),
            "publication_date": "",
            "remote": (job.get("workplaceType") or "").lower() == "remote"
            or bool(re.search("remote", location, re.I)),
        })
    return out


def _parse_ashby(company: str, data: dict) -> list[dict]:
    """Ashby posting-api payload -> normalized records."""
    out = []
    for job in data.get("jobs", []):
        if job.get("isListed") is False:
            continue
        out.append({
            "title": job.get("title") or "",
            "company": company,
            "url": job.get("jobUrl") or job.get("applyUrl") or "",
            "location": job.get("location") or "",
            "description": " ".join(
                part for part in (job.get("department"), job.get("team")) if part
            ),
            "publication_date": job.get("publishedAt") or "",
            "remote": bool(job.get("isRemote")),
        })
    return out


def _parse_smartrecruiters(company: str, data: dict) -> list[dict]:
    """SmartRecruiters public postings payload -> normalized records."""
    out = []
    for job in data.get("content", []):
        location = job.get("location") or {}
        where = (
            location.get("fullLocation")
            or ", ".join(
                part for part in (location.get("city"), location.get("region"), location.get("country")) if part
            )
        )
        out.append({
            "title": job.get("name") or "",
            "company": company,
            "url": f"https://jobs.smartrecruiters.com/{company}/{job.get('id')}",
            "location": where or "",
            "description": " ".join(
                part
                for part in (
                    (job.get("department") or {}).get("label"),
                    (job.get("function") or {}).get("label"),
                )
                if part
            ),
            "publication_date": job.get("releasedDate") or "",
            "remote": bool(location.get("remote")),
        })
    return out


def _fetch_greenhouse(company: str, timeout: float) -> list[dict] | None:
    data = _get_json(f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs", timeout)
    return None if data is None else _parse_greenhouse(company, data)


def _fetch_lever(company: str, timeout: float) -> list[dict] | None:
    for host in LEVER_HOSTS:
        data = _get_json(f"{host}/v0/postings/{company}?mode=json", timeout)
        if data is not None:
            return _parse_lever(company, data)
    return None


def _fetch_ashby(company: str, timeout: float) -> list[dict] | None:
    data = _get_json(f"https://api.ashbyhq.com/posting-api/job-board/{company}", timeout)
    return None if data is None else _parse_ashby(company, data)


def _fetch_smartrecruiters(company: str, timeout: float, limit: int = 100) -> list[dict] | None:
    out, offset = [], 0
    while True:
        data = _get_json(
            f"https://api.smartrecruiters.com/v1/companies/{company}/postings"
            f"?limit={limit}&offset={offset}",
            timeout,
        )
        if data is None:
            return None if offset == 0 else out
        page = _parse_smartrecruiters(company, data)
        out.extend(page)
        offset += len(page)
        # SmartRecruiters answers 200 with zero postings for unknown companies.
        if len(page) < limit or offset >= int(data.get("totalFound") or 0):
            return out if out else None


_FETCHERS = {
    "greenhouse": _fetch_greenhouse,
    "lever": _fetch_lever,
    "ashby": _fetch_ashby,
    "smartrecruiters": _fetch_smartrecruiters,
}


class ATSBoardSource:
    """Fetch open roles from company career boards hosted on public ATS APIs."""

    def __init__(self, boards: list[dict] | None = None, timeout: float = 10.0):
        self.boards = boards if boards is not None else self._configured_boards()
        self.timeout = timeout
        self.last_errors: list[dict] = []

    @staticmethod
    def _configured_boards() -> list[dict]:
        configured = os.getenv("ATS_COMPANY_BOARDS")
        if not configured:
            return [dict(board) for board in DEFAULT_BOARDS]
        try:
            boards = json.loads(configured)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid ATS_COMPANY_BOARDS: {exc}") from exc
        if not isinstance(boards, list):
            raise ValueError("ATS_COMPANY_BOARDS must be a JSON array")
        normalized = []
        for board in boards:
            if not isinstance(board, dict) or not board.get("company"):
                continue
            ats = str(board.get("ats") or "").lower()
            if ats and ats not in ATS_NAMES:
                continue
            normalized.append({"ats": ats or None, "company": str(board["company"])})
        return normalized

    def _fetch_board(self, ats: str | None, company: str) -> tuple[str, list[dict]] | None:
        """(ats_used, jobs) for the first board that exists, trying all four for a bare slug."""
        for name in ([ats] if ats else ATS_NAMES):
            jobs = _FETCHERS[name](company, self.timeout)
            if jobs is not None:
                return name, jobs
        return None

    @staticmethod
    def _matches_keywords(record: dict, keywords: set[str]) -> bool:
        haystack = f"{record.get('title', '')} {record.get('description', '')}".lower()
        return any(
            len(keyword) >= 2 and keyword.lower() in haystack
            for keyword in keywords
        )

    def discover(self, keywords: set[str]) -> list[tuple[str, list[dict], str]]:
        """[(source_name, records, source_url)] for every configured board.

        Records are shaped for MarketDiscoveryAgent._normalize_job and
        pre-filtered to jobs mentioning one of the keywords.
        """
        results = []
        for board in self.boards:
            company = board["company"]
            try:
                fetched = self._fetch_board(board.get("ats"), company)
            except Exception as exc:  # never let one board kill discovery
                self.last_errors.append({"source": f"ATS:{company}", "error": str(exc)})
                continue
            if not fetched:
                continue
            ats_name, jobs = fetched
            matching = [job for job in jobs if self._matches_keywords(job, keywords)] if keywords else jobs
            if not matching:
                continue
            source_name = f"ATS/{ats_name}"
            source_url = f"https://{ats_name}.ats/{company}"
            results.append((source_name, matching, source_url))
        return results


class AdzunaSource:
    """Adzuna official jobs API (free tier). Opt-in via env credentials."""

    API_ROOT = "https://api.adzuna.com/v1/api/jobs"
    COUNTRY_CODES = {
        "india": "in", "united states": "us", "usa": "us",
        "united kingdom": "uk", "germany": "de", "france": "fr",
        "australia": "au", "canada": "ca", "singapore": "sg",
        "netherlands": "nl", "ireland": "ie", "spain": "es",
        "italy": "it", "poland": "pl", "brazil": "br",
        "mexico": "mx", "south africa": "za", "austria": "at",
        "belgium": "be", "switzerland": "ch", "new zealand": "nz",
    }

    def __init__(
        self,
        app_id: str | None = None,
        app_key: str | None = None,
        timeout: float = 10.0,
    ):
        self.app_id = app_id or os.getenv("ADZUNA_APP_ID")
        self.app_key = app_key or os.getenv("ADZUNA_APP_KEY")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    def country_code(self, country: str) -> str | None:
        return self.COUNTRY_CODES.get((country or "").strip().lower())

    @staticmethod
    def _normalize(item: dict) -> dict | None:
        title = (item.get("title") or "").strip()
        company = ((item.get("company") or {}).get("display_name") or "").strip()
        url = (item.get("redirect_url") or "").strip()
        if not title or not url:
            return None
        location = ((item.get("location") or {}).get("display_name") or "").strip()
        raw_description = str(item.get("description") or "")
        description = " ".join(BeautifulSoup(raw_description, "html.parser").get_text(" ").split())
        return {
            "title": title,
            "company": company or "Unknown",
            "url": url,
            "location": location,
            "description": description,
            "publication_date": item.get("created") or "",
            "remote": bool(re.search("remote", f"{title} {location}", re.I)),
        }

    def search(
        self,
        what: str,
        country: str = "India",
        where: str = "",
        max_pages: int = 3,
        results_per_page: int = 50,
    ) -> tuple[str, list[dict], str]:
        """(source_name, records, source_url) for an Adzuna keyword search."""
        code = self.country_code(country)
        if code is None:
            raise ValueError(f"Adzuna has no country mapping for {country!r}")
        records: list[dict] = []
        for page in range(1, (max_pages or 1) + 1):
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": results_per_page,
                "what": what,
            }
            if where:
                params["where"] = where
            url = f"{self.API_ROOT}/{code}/search/{page}"
            if httpx is not None:
                with httpx.Client(
                    timeout=self.timeout, headers={"User-Agent": USER_AGENT}
                ) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    payload = response.json()
            else:
                from urllib.parse import urlencode
                from urllib.request import Request, urlopen

                request = Request(
                    f"{url}?{urlencode(params)}", headers={"User-Agent": USER_AGENT}
                )
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            items = payload.get("results", []) if isinstance(payload, dict) else []
            if not items:
                break
            for item in items:
                record = self._normalize(item)
                if record:
                    records.append(record)
            if len(items) < results_per_page:
                break
        return "Adzuna", records, f"{self.API_ROOT}/{code}/search"
