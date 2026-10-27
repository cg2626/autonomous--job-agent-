from app.agents.crawler import MarketDiscoveryAgent
from app.agents.parser import extract_skills_from_text


class FakeMarketDiscoveryAgent(MarketDiscoveryAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_count = 0

    def _request_json(self, url, params):
        self.request_count += 1
        return {
            "data": [
                {
                    "title": f"Python Developer {index}",
                    "company_name": f"Employer {index}",
                    "url": f"https://jobs.example.test/{index}",
                    "location": "Bengaluru, India",
                    "description": "Build Python services for a growing product team.",
                    "is_startup": index == 6,
                }
                for index in range(1, 7)
            ]
        }


def test_discovery_has_no_five_job_cap_and_keeps_startups():
    agent = FakeMarketDiscoveryAgent(
        sources=[{"name": "Test feed", "url": "https://jobs.example.test/feed"}]
    )

    jobs = agent.discover_market_jobs(
        candidate_skills=["python"],
        candidate_years=1.0,
        resume_text="Python Developer",
        country="India",
    )

    assert len(jobs) == 6
    assert any(job["is_startup"] for job in jobs)


def test_unknown_resume_does_not_get_devops_fallback_skills():
    assert extract_skills_from_text("Marketing manager with campaign strategy experience") == []


def test_go_to_phrase_is_not_the_go_language():
    assert "go" not in extract_skills_from_text("Docsumo is your go-to Document AI solution")
    assert "go" in extract_skills_from_text("Backend services written in Go and PostgreSQL")
    assert "golang" in extract_skills_from_text("Senior Software Engineer (Golang)")


def test_repeated_discovery_uses_cache():
    agent = FakeMarketDiscoveryAgent(
        sources=[{"name": "Test feed", "url": "https://jobs.example.test/feed"}]
    )
    arguments = {
        "candidate_skills": ["python"],
        "candidate_years": 1.0,
        "resume_text": "Python Developer",
        "country": "India",
    }

    agent.discover_market_jobs(**arguments)
    agent.discover_market_jobs(**arguments)

    assert agent.last_cache_hit is True
    assert agent.request_count == 1


def test_devops_resume_keeps_devops_signal_in_source_query():
    agent = MarketDiscoveryAgent(sources=[])
    terms = agent._candidate_terms(
        ["python", "kubernetes", "terraform"],
        "DevOps Engineer in Hyderabad, India",
    )

    assert "devops" in terms
    assert agent._search_query(terms).startswith("devops engineer")


def test_bare_remote_listing_is_not_assumed_to_be_in_india():
    agent = MarketDiscoveryAgent(sources=[])
    remote_job = {
        "location": "Remote",
        "remote": True,
    }
    worldwide_job = {
        "location": "Worldwide remote",
        "remote": True,
    }

    assert agent._matches_country(remote_job, "India") is False
    assert agent._matches_country(worldwide_job, "India") is True
