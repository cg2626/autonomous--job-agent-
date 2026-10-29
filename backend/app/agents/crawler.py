"""Job-page cleaning and multi-source job discovery.

The discovery agent deliberately does not contain a role-specific fixture or a
result limit. Public feeds are paginated until they are exhausted, and the
resume is used to decide which roles are relevant.
"""

import hashlib
import json
import os
import re
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bs4 import BeautifulSoup

try:
    import httpx
except ImportError:  # Keep page parsing usable in a minimal test environment.
    httpx = None

from app.agents.parser import SKILL_TAXONOMY_WEIGHTS, extract_skills_from_text, extract_skills_fuzzy
from app.agents.sources import ATSBoardSource, AdzunaSource


def fuzzy_dedupe_jobs(jobs: list[dict], title_threshold: int = 90) -> list[dict]:
    """Drop near-duplicate postings found on multiple sources.

    Two jobs are duplicates when their normalized companies match and their
    normalized titles are fuzzy-similar (rapidfuzz token_set_ratio >=
    threshold). The copy with the highest relevance_score is kept. Falls back
    to exact (company, title) matching when rapidfuzz is unavailable.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        fuzz = None

    def norm_company(name: Any) -> str:
        text = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
        for suffix in (
            "private limited", "pvt ltd", "p ltd", "incorporated",
            "corporation", "technologies", "technology", "solutions",
            "services", "consulting", "consultancy", "inc", "llc", "ltd",
            "gmbh", "srl", "pty",
        ):
            text = re.sub(rf"\b{re.escape(suffix)}\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def norm_title(title: Any) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(title).lower())).strip()

    kept: list[dict] = []
    seen: list[tuple[str, str, dict]] = []  # (company, title, job)
    for job in jobs:
        company = norm_company(job.get("company", ""))
        title = norm_title(job.get("title", ""))
        duplicate_of = None
        for kept_company, kept_title, kept_job in seen:
            if not company or company != kept_company or not title:
                continue
            if fuzz is None:
                is_dup = title == kept_title
            else:
                is_dup = bool(kept_title) and fuzz.token_set_ratio(title, kept_title) >= title_threshold
            if is_dup:
                duplicate_of = kept_job
                break
        if duplicate_of is None:
            seen.append((company, title, job))
            kept.append(job)
        elif job.get("relevance_score", 0) > duplicate_of.get("relevance_score", 0):
            kept[kept.index(duplicate_of)] = job
            for i, (seen_company, seen_title, seen_job) in enumerate(seen):
                if seen_job is duplicate_of:
                    seen[i] = (seen_company, seen_title, job)
                    break
    return kept


class EthicalCrawlerAgent:
    """Fetch a job page and remove non-content DOM before storing it."""

    def prune_raw_html(self, raw_html: str) -> dict:
        raw_bytes = len(raw_html.encode("utf-8"))
        soup = BeautifulSoup(raw_html, "html.parser")
        for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            element.decompose()

        cleaned_text = " ".join(soup.get_text(" ").split())
        clean_bytes = len(cleaned_text.encode("utf-8"))
        reduction_pct = round(
            max(0.0, (1 - clean_bytes / raw_bytes) * 100) if raw_bytes else 0.0,
            2,
        )
        return {
            "cleaned_text": cleaned_text,
            "metrics": {
                "raw_bytes": raw_bytes,
                "clean_bytes": clean_bytes,
                "payload_reduction_pct": reduction_pct,
            },
        }

    async def crawl_and_prune(self, url: str) -> dict:
        if httpx is not None:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "TalentAlignAI/3.0 job-discovery"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
            raw_html = response.text
        else:
            from urllib.request import Request, urlopen

            def fetch_html() -> str:
                request = Request(url, headers={"User-Agent": "TalentAlignAI/3.0 job-discovery"})
                with urlopen(request, timeout=20) as response:
                    return response.read().decode("utf-8", errors="replace")

            raw_html = await asyncio.to_thread(fetch_html)

        result = self.prune_raw_html(raw_html)
        return {
            "clean_text": result["cleaned_text"],
            "raw_bytes": result["metrics"]["raw_bytes"],
            "clean_bytes": result["metrics"]["clean_bytes"],
            "compression_pct": result["metrics"]["payload_reduction_pct"],
        }


class MarketDiscoveryAgent:
    """Discover nationwide openings from public, paginated job-board feeds.

    The default sources include both established employers and smaller/startup
    employers. No company-size filter is applied. Extra sources can be added
    with ``JOB_DISCOVERY_SOURCES`` as a JSON array of ``{"name", "url"}``
    objects.
    """

    DEFAULT_SOURCES = (
        {"name": "Arbeitnow", "url": "https://www.arbeitnow.com/api/job-board-api"},
        {"name": "Remotive", "url": "https://remotive.com/api/remote-jobs"},
        {
            "name": "Himalayas",
            "url": "https://himalayas.app/jobs/api/search",
            "pagination": "page",
            "query_param": "q",
            "country_param": "country",
            "page_size": 20,
        },
    )

    ROLE_TERMS = {
        "developer", "engineer", "architect", "administrator", "analyst",
        "scientist", "designer", "manager", "director", "consultant",
        "marketing", "sales", "finance", "accounting", "legal",
        "human resources", "recruiter", "operations", "research", "product",
        "project", "security", "data", "qa", "tester", "support", "writer",
        "editor", "teacher", "nurse", "medical", "intern", "executive",
        "devops", "sre", "site reliability", "platform", "cloud",
        "infrastructure", "mlops", "gitops", "release engineering",
    }

    ROLE_QUERY_PRIORITY = (
        "devops", "site reliability", "sre", "platform", "cloud",
        "infrastructure", "mlops", "gitops", "release engineering",
        "developer", "engineer", "analyst", "designer", "manager",
    )

    # When the candidate has SAP skills, these lead the board query so the
    # search asks for SAP roles instead of generic developer roles.
    SAP_QUERY_PRIORITY = ("fiori", "ui5", "btp", "capm", "sap")

    def __init__(self, sources: list[dict[str, str]] | None = None, timeout: float = 8.0):
        self.sources = sources or self._configured_sources()
        self.timeout = timeout
        self.last_errors: list[dict[str, str]] = []
        self.last_cache_hit = False
        self._memory_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self.cache_ttl_seconds = int(os.getenv("JOB_DISCOVERY_CACHE_TTL", "900"))
        self._redis = self._connect_redis()

    @staticmethod
    def _connect_redis():
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return None
        try:
            import redis

            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
            client.ping()
            return client
        except Exception:
            # Discovery still works locally when Redis is not running.
            return None

    def _configured_sources(self) -> list[dict[str, str]]:
        configured = os.getenv("JOB_DISCOVERY_SOURCES")
        if not configured:
            return [dict(source) for source in self.DEFAULT_SOURCES]
        try:
            sources = json.loads(configured)
            if not isinstance(sources, list):
                raise ValueError("JOB_DISCOVERY_SOURCES must be a JSON array")
            return [source for source in sources if isinstance(source, dict) and source.get("url")]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JOB_DISCOVERY_SOURCES: {exc}") from exc

    def _request_json(self, url: str, params: dict[str, Any]) -> Any:
        if httpx is not None:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "TalentAlignAI/3.0 job-discovery"},
            ) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        from urllib.parse import urlencode
        from urllib.request import Request, urlopen

        request = Request(
            f"{url}?{urlencode(params)}",
            headers={"User-Agent": "TalentAlignAI/3.0 job-discovery"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _fetch_source(
        self,
        source: dict[str, str],
        max_pages: int | None,
        query: str = "",
        country: str = "",
    ):
        """Yield every page until a source returns no new records."""
        if source.get("pagination") == "cursor":
            cursor = None
            page = 1
            while max_pages is None or page <= max_pages:
                params: dict[str, Any] = {"limit": 20}
                if cursor:
                    params["cursor"] = cursor
                payload = self._request_json(source["url"], params)
                records = self._records_from_payload(payload)
                if not records:
                    return
                yield from records
                cursor = payload.get("nextCursor") if isinstance(payload, dict) else None
                if not cursor:
                    return
                page += 1
            return

        seen_page_keys: set[str] = set()
        page = 1
        while max_pages is None or page <= max_pages:
            params: dict[str, Any] = {
                "page": page,
                "limit": source.get("page_size", 100),
            }
            if query and source.get("query_param"):
                params[source["query_param"]] = query
            if country and source.get("country_param"):
                params[source["country_param"]] = country
            if source.get("worldwide_param"):
                params[source["worldwide_param"]] = "true"

            payload = self._request_json(source["url"], params)
            records = self._records_from_payload(payload)
            if not records:
                return

            page_key = hashlib.sha1(
                json.dumps(records, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if page_key in seen_page_keys:
                return
            seen_page_keys.add(page_key)
            yield from records
            if len(records) < int(source.get("page_size", 100)):
                return
            page += 1

    @staticmethod
    def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "jobs", "results", "items"):
            records = payload.get(key)
            if isinstance(records, list):
                return [item for item in records if isinstance(item, dict)]
        return []

    @staticmethod
    def _cache_key(
        candidate_skills: list | set,
        resume_text: str,
        country: str,
        max_pages: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "skills": sorted(str(skill).lower() for skill in candidate_skills),
                "resume": resume_text.strip().lower(),
                "country": country.strip().lower(),
                "max_pages": max_pages,
            },
            sort_keys=True,
        )
        return "talentalign:market:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> list[dict[str, Any]] | None:
        if self._redis is not None:
            try:
                cached = self._redis.get(key)
                return json.loads(cached) if cached else None
            except Exception:
                self._redis = None

        cached = self._memory_cache.get(key)
        if cached is None:
            return None
        expires_at, jobs = cached
        if expires_at <= time.monotonic():
            self._memory_cache.pop(key, None)
            return None
        return jobs

    def _write_cache(self, key: str, jobs: list[dict[str, Any]]) -> None:
        if self._redis is not None:
            try:
                self._redis.setex(key, self.cache_ttl_seconds, json.dumps(jobs))
                return
            except Exception:
                self._redis = None
        self._memory_cache[key] = (time.monotonic() + self.cache_ttl_seconds, jobs)

    @classmethod
    def _search_query(cls, candidate_terms: set[str]) -> str:
        """Build a short source query while preserving role and skill signals."""
        sap_terms = [term for term in cls.SAP_QUERY_PRIORITY if term in candidate_terms]
        role_terms = sorted(
            (term for term in candidate_terms if term in cls.ROLE_TERMS),
            key=lambda term: (
                cls.ROLE_QUERY_PRIORITY.index(term)
                if term in cls.ROLE_QUERY_PRIORITY else len(cls.ROLE_QUERY_PRIORITY),
                -len(term),
                term,
            ),
        )
        skill_terms = sorted(
            (term for term in candidate_terms if term not in cls.ROLE_TERMS),
            key=lambda term: (-len(term), term),
        )
        return " ".join((sap_terms + role_terms + skill_terms)[:3])

    def discover_market_jobs(
        self,
        candidate_skills: list | set,
        candidate_years: float,
        resume_text: str = "",
        country: str = "India",
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all relevant openings found in all configured source pages.

        Experience is metadata only; it is not a hard filter, so junior,
        mid-level, and senior openings remain visible.
        """
        del candidate_years
        self.last_errors = []
        self.last_cache_hit = False
        configured_page_limit = os.getenv("JOB_DISCOVERY_MAX_PAGES")
        page_limit = max_pages if max_pages and max_pages > 0 else (
            int(configured_page_limit) if configured_page_limit else None
        )
        candidate_terms = self._candidate_terms(candidate_skills, resume_text)
        cache_key = self._cache_key(candidate_skills, resume_text, country, page_limit)
        cached_jobs = self._read_cache(cache_key)
        if cached_jobs is not None:
            self.last_cache_hit = True
            return cached_jobs

        search_query = self._search_query(candidate_terms)
        jobs: list[dict[str, Any]] = []
        seen_jobs: set[str] = set()

        def fetch_source(source: dict[str, str]):
            source_name = source.get("name") or source.get("url", "Unknown source")
            records = list(self._fetch_source(source, page_limit, search_query, country))
            return source, source_name, records

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(self.sources)))) as executor:
            futures = [executor.submit(fetch_source, source) for source in self.sources]
            for future in as_completed(futures):
                try:
                    source, source_name, records = future.result()
                    self._ingest_records(
                        records, source_name, source["url"],
                        jobs, seen_jobs, candidate_terms, country,
                    )
                except Exception as exc:
                    source_name = "Unknown source"
                    if future in futures:
                        source_index = futures.index(future)
                        source_name = self.sources[source_index].get(
                            "name", self.sources[source_index].get("url", source_name)
                        )
                    self.last_errors.append({"source": source_name, "error": str(exc)})

        # Vastness lane (2026-10-29): ATS career boards + Adzuna (opt-in).
        # Failures are isolated per source so a dead board never kills discovery.
        # ATS pre-filtering uses SAP-first terms when the candidate has them,
        # so company boards yield SAP roles instead of every open req.
        sap_terms = {term for term in candidate_terms if term in self.SAP_QUERY_PRIORITY}
        for source_name, records, source_url in self._discover_extra_sources(
            search_query, sap_terms or candidate_terms, country
        ):
            self._ingest_records(
                records, source_name, source_url,
                jobs, seen_jobs, candidate_terms, country,
            )

        # Better lane (2026-10-29): fuzzy cross-source dedup keeps the
        # highest-relevance copy of each posting.
        jobs = fuzzy_dedupe_jobs(jobs)

        sorted_jobs = sorted(
            jobs,
            key=lambda job: (-job["relevance_score"], job["company"].lower(), job["title"].lower()),
        )
        self._write_cache(cache_key, sorted_jobs)
        return sorted_jobs

    def _ingest_records(
        self,
        records: list[dict[str, Any]],
        source_name: str,
        source_url: str,
        jobs: list[dict[str, Any]],
        seen_jobs: set[str],
        candidate_terms: set[str],
        country: str,
    ) -> None:
        """Normalize records and append relevant, unseen jobs (shared by all sources)."""
        for record in records:
            job = self._normalize_job(record, source_name, source_url)
            if not job or not self._matches_country(job, country):
                continue
            relevance = self._relevance(job, candidate_terms)
            if candidate_terms and relevance == 0:
                continue
            job["relevance_score"] = relevance
            key = job["portal_url"] or f"{job['company']}|{job['title']}|{job['location']}"
            if key in seen_jobs:
                continue
            seen_jobs.add(key)
            jobs.append(job)

    def _discover_extra_sources(
        self,
        search_query: str,
        candidate_terms: set[str],
        country: str,
    ) -> list[tuple[str, list[dict[str, Any]], str]]:
        """[(source_name, records, source_url)] from ATS boards and Adzuna."""
        out: list[tuple[str, list[dict[str, Any]], str]] = []
        try:
            ats = ATSBoardSource()
            out.extend(ats.discover(candidate_terms))
            for error in ats.last_errors:
                self.last_errors.append(error)
        except Exception as exc:
            self.last_errors.append({"source": "ATS boards", "error": str(exc)})
        try:
            adzuna = AdzunaSource()
            if adzuna.is_configured:
                out.append(adzuna.search(search_query, country=country))
            # Unconfigured Adzuna is not an error: it is an opt-in source.
        except Exception as exc:
            self.last_errors.append({"source": "Adzuna", "error": str(exc)})
        return out

    @classmethod
    def _candidate_terms(cls, candidate_skills: list | set, resume_text: str) -> set[str]:
        terms = {str(skill).lower() for skill in candidate_skills if str(skill).strip()}
        resume_lower = resume_text.lower()
        terms.update(
            term for term in cls.ROLE_TERMS
            if re.search(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", resume_lower)
        )
        if not terms and resume_text.strip():
            words = re.findall(r"[a-z][a-z+#.-]{2,}", resume_lower)
            stop_words = {"with", "from", "that", "this", "years", "experience", "and", "the"}
            terms.update(word for word in words if word not in stop_words)
        return terms

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value or "")

    def _normalize_job(
        self,
        record: dict[str, Any],
        source_name: str,
        source_url: str,
    ) -> dict[str, Any] | None:
        title = self._text(record.get("title") or record.get("position")).strip()
        company = self._text(
            record.get("company_name")
            or record.get("companyName")
            or record.get("company")
            or record.get("employer")
        ).strip()
        url = self._text(
            record.get("url")
            or record.get("applicationLink")
            or record.get("job_url")
            or record.get("redirect_url")
        ).strip()
        if not title or not company or not url:
            return None

        description = self._text(
            record.get("description") or record.get("excerpt") or record.get("summary")
        )
        description = " ".join(BeautifulSoup(description, "html.parser").get_text(" ").split())
        restrictions = record.get("locationRestrictions")
        location_value = (
            record.get("location")
            or record.get("candidate_required_location")
            or record.get("locations")
            or restrictions
        )
        if isinstance(location_value, list) and all(isinstance(item, dict) for item in location_value):
            location = ", ".join(
                str(item.get("name") or item.get("alpha2") or "")
                for item in location_value
            ).strip(", ")
        else:
            location = self._text(location_value).strip()
        if source_name.lower() == "himalayas" and not location:
            location = "Worldwide remote"
        remote = bool(record.get("remote")) or source_name.lower() == "himalayas" or "remote" in f"{location} {title}".lower()
        full_text = f"{title} {description}"
        weights = {
            skill: SKILL_TAXONOMY_WEIGHTS[skill]
            for skill in extract_skills_fuzzy(full_text)
        }
        if not weights:
            weights = {"general": 1.0}

        source_lower = source_name.lower()
        startup_source = any(term in source_lower for term in ("wellfound", "startup"))
        startup_flag = bool(record.get("startup") or record.get("is_startup")) or startup_source
        raw_id = f"{source_name}|{url}".encode("utf-8")
        job_code = "JOB-LIVE-" + hashlib.sha1(raw_id).hexdigest()[:16].upper()
        return {
            "job_code": job_code,
            "title": title,
            "company": company,
            "portal_url": url,
            "source": source_name,
            "source_url": source_url,
            "location": location or ("Remote" if remote else "Location not specified"),
            "remote": remote,
            "country": "India" if self._matches_country_text(location, "India") else "",
            "is_startup": startup_flag,
            "posted_at": self._text(
                record.get("publication_date") or record.get("pubDate") or record.get("created_at")
            ),
            "req_years": self._required_years(full_text),
            "description": description or title,
            "weights": weights,
        }

    @staticmethod
    def _required_years(text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())
        return float(match.group(1)) if match else 0.0

    @classmethod
    def _matches_country_text(cls, location: str, country: str) -> bool:
        location_lower = location.lower()
        country_lower = country.lower().strip()
        if not location_lower or "worldwide" in location_lower or "anywhere" in location_lower:
            return False
        if country_lower in location_lower:
            return True
        if country_lower == "india":
            return any(
                city in location_lower
                for city in (
                    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad",
                    "chennai", "pune", "kolkata", "noida", "gurgaon", "gurugram",
                    "ahmedabad", "jaipur", "kochi", "india",
                )
            )
        return False

    @classmethod
    def _matches_country(cls, job: dict[str, Any], country: str) -> bool:
        if not country or country.lower() in {"any", "worldwide", "all"}:
            return True
        location = job["location"].lower()
        if job["remote"] and any(term in location for term in ("worldwide", "global", "anywhere")):
            return True
        if cls._matches_country_text(location, country):
            return True
        return False

    @staticmethod
    def _relevance(job: dict[str, Any], terms: set[str]) -> int:
        """Count whole-term hits so java does not match javascript, or go match google."""
        searchable = f"{job['title']} {job['description']}".lower()
        hits = 0
        for term in terms:
            needle = str(term).lower().strip()
            if len(needle) < 2:
                continue
            if re.search(r"(?<![\w-])" + re.escape(needle) + r"(?![\w-])", searchable):
                hits += 1
        return hits


class DOMPruningCrawlerAgent(EthicalCrawlerAgent):
    def __init__(self, min_delay_seconds: float = 2.0):
        self.min_delay_seconds = min_delay_seconds
