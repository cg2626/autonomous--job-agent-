from app.agents.matcher import HybridJobMatcherAgent
from app.agents.parser import ResumeAndJobParserAgent
from app.agents.crawler import DOMPruningCrawlerAgent

def test_stage1_worked_numerical_example():
    """
    Verifies that the Python implementation reproduces the Section 7 numerical
    example from the Stage 1 Mathematical Specification PDF:
    S_sem = 0.8420, S_skill = 7.0/8.0 = 0.8750, Φ_exp = exp(-0.175) = 0.8395 -> S_total ≈ 71.80%
    """
    matcher = HybridJobMatcherAgent(alpha=0.60, decay_lambda=0.35, use_neural_model=False)
    res = matcher.evaluate_job_match(
        resume_text="DevOps Engineer with AWS, Kubernetes, Docker, Terraform, Jenkins, Python",
        job_text="Senior Cloud & DevOps Engineer requiring Kubernetes, Terraform, AWS, Jenkins, Golang",
        resume_skills={"kubernetes", "docker", "terraform", "aws", "jenkins", "python"},
        job_skills_weighted={"kubernetes": 2.0, "terraform": 2.0, "aws": 1.5, "jenkins": 1.5, "golang": 1.0},
        candidate_years=2.5,
        required_years=3.0,
        override_s_sem=0.8420
    )
    assert res["skill_overlap_pct"] == 87.50
    assert res["experience_penalty_factor"] == 0.8395
    assert abs(res["final_match_score_pct"] - 71.80) <= 0.02
    assert res["missing_skills"] == ["golang"]

def test_case_study_3_dom_pruning_efficiency():
    """Verifies that DOM pruning removes script/style/svg bloat and achieves >85% compression."""
    crawler = DOMPruningCrawlerAgent()
    bloated_html = (
        "<html><head><style>" + ("body {color: red;} " * 500) + "</style>"
        "<script>" + ("console.log('spa bloat'); " * 800) + "</script></head>"
        "<body><nav>Menu Links</nav><svg>" + ("<path d='M0 0'/>" * 400) + "</svg>"
        "<main><h1>Cloud DevOps Engineer</h1><p>Requires 3+ years experience in Kubernetes, AWS, and Terraform.</p></main>"
        "<footer>Copyright 2026</footer></body></html>"
    )
    result = crawler.prune_raw_html(bloated_html)
    assert "Cloud DevOps Engineer" in result["cleaned_text"]
    assert "console.log" not in result["cleaned_text"]
    assert result["metrics"]["payload_reduction_pct"] > 85.0
