from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class JobPostingInput(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job posting")
    title: str = Field(..., description="Role title, e.g., Senior Cloud & DevOps Engineer")
    company: str = Field(..., description="Company name")
    url: Optional[str] = Field(None, description="Direct career portal URL")
    description: str = Field(..., description="Cleaned job description text")
    required_years: Optional[float] = Field(None, description="Minimum required experience in years (auto-extracted if None)")
    custom_skill_weights: Optional[Dict[str, float]] = Field(None, description="Optional override for required skill weights")

class ResumeInput(BaseModel):
    candidate_name: str = Field(default="Candidate")
    resume_text: str = Field(..., description="Extracted plain text from candidate resume")
    candidate_years: Optional[float] = Field(None, description="Total years of experience (auto-extracted if None)")
    explicit_skills: Optional[List[str]] = Field(None, description="Optional explicit list of candidate skills")

class MatchEvaluationRequest(BaseModel):
    resume: ResumeInput
    jobs: List[JobPostingInput]
    alpha: float = Field(default=0.60, ge=0.0, le=1.0, description="Convex weight balancing semantic vector vs NER skill overlap")
    decay_lambda: float = Field(default=0.35, gt=0.0, description="Shepard's exponential decay rate for experience deficit")

class SkillGapAdvice(BaseModel):
    missing_skill: str
    priority_level: str
    weight: float
    recommended_action: str
    sample_resume_bullet: str

class SingleJobMatchResult(BaseModel):
    job_id: str
    title: str
    company: str
    url: Optional[str]
    semantic_similarity_pct: float
    skill_overlap_pct: float
    experience_penalty_factor: float
    final_match_score_pct: float
    match_tier: str
    candidate_years_detected: float
    job_required_years_detected: float
    matched_skills: List[str]
    missing_skills: List[str]
    advisor_recommendations: List[SkillGapAdvice]

class BatchMatchResponse(BaseModel):
    candidate_name: str
    total_jobs_evaluated: int
    hyperparameters: Dict[str, float]
    ranked_matches: List[SingleJobMatchResult]
