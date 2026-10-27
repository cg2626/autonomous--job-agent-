import math

from app.agents.advisor import SkillGapAdvisorAgent
from app.agents.crawler import MarketDiscoveryAgent
from app.agents.matcher import HybridJobMatcherAgent, build_market_brief


def _stage1_matcher():
    return HybridJobMatcherAgent(alpha=0.60, decay_lambda=0.35, use_neural_model=False)


def test_missing_skill_reports_the_exact_score_lift():
    """Learning the one missing skill should move the Stage 1 formula by a known amount."""
    matcher = _stage1_matcher()
    s_sem = 0.8420
    result = matcher.evaluate_job_match(
        resume_text="DevOps Engineer with AWS, Kubernetes, Docker, Terraform, Jenkins, Python",
        job_text="Senior Cloud & DevOps Engineer requiring Kubernetes, Terraform, AWS, Jenkins, Golang",
        resume_skills={"kubernetes", "docker", "terraform", "aws", "jenkins", "python"},
        job_skills_weighted={"kubernetes": 2.0, "terraform": 2.0, "aws": 1.5, "jenkins": 1.5, "golang": 1.0},
        candidate_years=2.5,
        required_years=3.0,
        override_s_sem=s_sem,
    )
    phi = math.exp(-0.35 * 0.5)
    current = phi * ((0.60 * s_sem) + (0.40 * (7.0 / 8.0)))
    upgraded = phi * ((0.60 * s_sem) + (0.40 * 1.0))

    assert result["missing_skills"] == ["golang"]
    assert result["best_next_skill"]["skill"] == "golang"
    assert result["best_next_skill"]["score_lift_pct"] == round((upgraded - current) * 100.0, 2)
    assert result["best_next_skill"]["score_lift_pct"] > 0


def test_empty_requirements_do_not_invent_a_devops_profile():
    matcher = _stage1_matcher()
    result = matcher.evaluate_job_match(
        resume_text="Marketing manager with campaign strategy experience",
        job_text="Marketing manager",
        resume_skills=["seo"],
        job_weights={"general": 1.0},
        cand_years=4,
        req_years=2,
        override_s_sem=0.4,
    )

    assert result["missing_skills"] == []
    assert result["matched_skills"] == []
    assert result["s_skill_pct"] == 100.0
    assert result["experience_fit"] == "Experience-ready"


def test_advisor_only_mentions_skills_the_candidate_has():
    advice = SkillGapAdvisorAgent().generate_advice(
        ["seo", "excel"],
        {"seo": 1.2, "tableau": 1.5},
        job_title="Marketing Analyst",
        score_lifts=[{"skill": "tableau", "weight": 1.5, "score_lift_pct": 12.5}],
    )

    assert advice["matched_skills"] == ["seo"]
    assert advice["missing_skills"] == ["tableau"]
    bullet = advice["ats_bullet_points"][0].lower()
    assert "kubernetes" not in bullet
    assert "seo" in bullet
    assert "tableau" not in bullet
    assert "12.5" in advice["recommendations"][0]


def test_short_skill_names_do_not_match_longer_words():
    agent = MarketDiscoveryAgent(sources=[])
    javascript_job = {
        "title": "JavaScript Developer",
        "description": "Build interfaces with javascript and react.",
    }
    java_job = {
        "title": "Java Developer",
        "description": "Backend java services.",
    }

    assert agent._relevance(javascript_job, {"java"}) == 0
    assert agent._relevance(java_job, {"java"}) == 1
    assert agent._relevance(
        {"title": "Golang Engineer", "description": "Google cloud platform"},
        {"go"},
    ) == 0


def test_offline_similarity_follows_the_wording():
    matcher = _stage1_matcher()
    shared = dict(
        resume_skills=["python"],
        job_weights={"python": 1.0},
        cand_years=2,
        req_years=2,
    )
    close = matcher.evaluate_job_match(
        resume_text="python fastapi postgresql",
        job_text="python fastapi postgresql",
        **shared,
    )
    distant = matcher.evaluate_job_match(
        resume_text="python fastapi postgresql",
        job_text="ballet choreography costume design",
        **shared,
    )

    assert close["semantic_mode"] == "lexical"
    assert close["s_sem_pct"] > distant["s_sem_pct"]
    assert close["s_sem_pct"] != 85.0


def test_rank_jobs_orders_by_hybrid_score():
    matcher = _stage1_matcher()
    ranked, mode = matcher.rank_jobs(
        "python data analyst pandas",
        ["python", "pandas"],
        3,
        [
            {
                "title": "Ward Nurse",
                "company": "City Hospital",
                "description": "patient care ward rounds",
                "weights": {"accounting": 2.0},
                "req_years": 3,
            },
            {
                "title": "Data Analyst",
                "company": "Northwind",
                "description": "python pandas reporting",
                "weights": {"python": 2.0, "pandas": 1.5},
                "req_years": 2,
            },
        ],
    )

    assert mode == "lexical"
    assert ranked[0]["job"]["title"] == "Data Analyst"
    assert ranked[0]["evaluation"]["final_match_score_pct"] > ranked[1]["evaluation"]["final_match_score_pct"]
    assert ranked[0]["evaluation"]["application_queue"] == "Apply now"


def test_market_brief_picks_the_skill_that_promotes_the_most_roles():
    ranked = [
        {
            "job": {"company": "Northwind"},
            "evaluation": {
                "final_match_score_pct": 60.0,
                "application_queue": "One skill away",
                "skill_lifts": [
                    {"skill": "tableau", "score_lift_pct": 8.0},
                    {"skill": "sql", "score_lift_pct": 2.0},
                ],
            },
        },
        {
            "job": {"company": "Fabrikam"},
            "evaluation": {
                "final_match_score_pct": 70.0,
                "application_queue": "Apply now",
                "skill_lifts": [
                    {"skill": "tableau", "score_lift_pct": 12.0},
                ],
            },
        },
        {
            "job": {"company": "Contoso"},
            "evaluation": {
                "final_match_score_pct": 40.0,
                "application_queue": "Build toward",
                "skill_lifts": [
                    {"skill": "sql", "score_lift_pct": 3.0},
                ],
            },
        },
    ]

    brief = build_market_brief(ranked)

    assert brief["apply_now"] == 1
    assert brief["one_skill_away"] == 1
    assert brief["levers"][0]["skill"] == "tableau"
    assert brief["levers"][0]["roles_improved"] == 2
    assert brief["levers"][0]["roles_promoted"] == 2
