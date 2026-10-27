import math
import re
import threading

import numpy as np

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+.#/-]{1,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you",
    "are", "our", "job", "role", "years", "year", "experience",
}

# A role asking up to this many years above the candidate still counts as
# meeting the experience request (Harshitha's near-5-year rule: a 4-year
# candidate is fine for a 5-year role).
EXPERIENCE_GRACE_YEARS = 1.0

# Employers and domains that are never a fit for this profile, no matter the
# score. Applied at ranking time so exclusions stay visible, not silent.
EXCLUDED_EMPLOYERS = ("infosys", "accenture")
MAX_BAND_YEARS = 7.0


class HybridJobMatcherAgent:
    def __init__(
        self,
        alpha: float = 0.60,
        decay_lambda: float = 0.35,
        use_neural_model: bool = True,
    ):
        self.alpha = alpha
        self.decay_lambda = decay_lambda
        self.use_neural_model = use_neural_model
        self.embedder = None
        self._embedder_failed = False
        self._embedder_lock = threading.Lock()

    def warm_embedder(self) -> bool:
        return self._semantic_encoder() is not None

    def _semantic_encoder(self):
        if not self.use_neural_model or self._embedder_failed:
            return None
        if self.embedder is not None:
            return self.embedder
        with self._embedder_lock:
            if self.embedder is not None or self._embedder_failed:
                return self.embedder
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedder_failed = True
                self.embedder = None
            return self.embedder

    @classmethod
    def _lexical_similarity(cls, left: str, right: str) -> float:
        """Cosine overlap of distinctive words, used when the embedding model is offline."""
        left_tokens = cls._tokens(left)
        right_tokens = cls._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        shared = len(left_tokens & right_tokens)
        return shared / math.sqrt(len(left_tokens) * len(right_tokens))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token for token in _TOKEN.findall(str(text).lower())
            if token not in _STOPWORDS
        }

    def semantic_scores(self, resume_text: str, job_texts: list[str]) -> tuple[list[float], str]:
        encoder = self._semantic_encoder()
        if encoder is not None and job_texts:
            try:
                resume_vector = encoder.encode([str(resume_text)], normalize_embeddings=True)[0]
                job_vectors = encoder.encode([str(text) for text in job_texts], normalize_embeddings=True)
                scores = [float(np.dot(resume_vector, vector)) for vector in job_vectors]
                return scores, "neural"
            except Exception:
                self._embedder_failed = True
                self.embedder = None
        return [self._lexical_similarity(resume_text, text) for text in job_texts], "lexical"

    def evaluate_job_match(self, *args, **kwargs):
        resume_text = kwargs.get("resume_text", args[0] if len(args) > 0 else "")
        job_text = kwargs.get("job_text", args[1] if len(args) > 1 else "")
        resume_skills = kwargs.get("resume_skills", args[2] if len(args) > 2 else [])
        job_weights = kwargs.get(
            "job_weights",
            kwargs.get("job_skills_weighted", args[3] if len(args) > 3 else {}),
        )
        cand_years = kwargs.get("cand_years", kwargs.get("candidate_years", args[4] if len(args) > 4 else 0.0))
        req_years = kwargs.get("req_years", kwargs.get("required_years", args[5] if len(args) > 5 else 0.0))

        if isinstance(resume_skills, (set, tuple)):
            resume_skills = list(resume_skills)
        if isinstance(job_weights, list):
            job_weights = {skill: 1.0 for skill in job_weights}
        if not isinstance(job_weights, dict):
            job_weights = {}

        specific_weights = {
            skill: weight
            for skill, weight in job_weights.items()
            if str(skill).lower() != "general"
        }
        resume_index = {str(skill).lower() for skill in resume_skills}

        override_s_sem = kwargs.get("override_s_sem")
        if override_s_sem is not None:
            s_sem = float(override_s_sem)
            semantic_mode = "provided"
        else:
            s_sem, semantic_mode = self.semantic_scores(resume_text, [str(job_text)])
            s_sem = s_sem[0]

        if specific_weights:
            matched_weights = sum(
                weight for skill, weight in specific_weights.items()
                if str(skill).lower() in resume_index
            )
            total_weight = sum(specific_weights.values())
            s_skill = matched_weights / total_weight if total_weight > 0 else 1.0
            matched_skills = [
                skill for skill in specific_weights
                if str(skill).lower() in resume_index
            ]
            missing_skills = [
                skill for skill in specific_weights
                if str(skill).lower() not in resume_index
            ]
        else:
            s_skill = 1.0
            matched_skills = []
            missing_skills = []

        cand_years = float(cand_years or 0.0)
        req_years = float(req_years or 0.0)
        shortfall = req_years - cand_years
        if shortfall <= 0:
            phi_exp = 1.0
            experience_fit = "Experience-ready"
            experience_note = "Experience meets the request, so the score is left intact."
        elif shortfall <= EXPERIENCE_GRACE_YEARS:
            # Near-band roles (e.g. asking 5 years of a 4-year candidate) count
            # as meeting the request instead of taking an experience haircut.
            phi_exp = 1.0
            experience_fit = "Experience-ready"
            experience_note = (
                f"Within the {EXPERIENCE_GRACE_YEARS:.0f}-year grace band, so experience "
                "is treated as meeting the request."
            )
        else:
            phi_exp = math.exp(-self.decay_lambda * shortfall)
            experience_fit = "Near fit" if phi_exp >= 0.70 else "Stretch role"
            experience_note = (
                f"{shortfall:.1f} years short of the request, so experience scales the score by {phi_exp:.3f}."
            )

        s_total = phi_exp * ((self.alpha * s_sem) + ((1.0 - self.alpha) * s_skill))
        score_pct = round(s_total * 100.0, 2)
        if score_pct >= 80.0:
            tier = "Tier 1: High Match"
        elif score_pct >= 65.0:
            tier = "Tier 2: Moderate Match"
        elif score_pct >= 50.0:
            tier = "Tier 3: Conditional Match"
        else:
            tier = "Tier 4: Low Match"

        skill_lifts = self._skill_lifts(specific_weights, resume_index, s_sem, phi_exp, s_total)
        semantic_points = round(phi_exp * self.alpha * s_sem * 100.0, 2)
        skill_points = round(phi_exp * (1.0 - self.alpha) * s_skill * 100.0, 2)
        best_next_skill = skill_lifts[0] if skill_lifts else None

        return {
            "final_match_score_pct": score_pct,
            "s_sem_pct": round(s_sem * 100.0, 2),
            "s_skill_pct": round(s_skill * 100.0, 2),
            "skill_overlap_pct": round(s_skill * 100.0, 2),
            "phi_exp": round(phi_exp, 3),
            "experience_penalty_factor": round(phi_exp, 4),
            "experience_fit": experience_fit,
            "experience_note": experience_note,
            "semantic_mode": semantic_mode,
            "semantic_points": semantic_points,
            "skill_points": skill_points,
            "match_tier": tier,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_lifts": skill_lifts,
            "best_next_skill": best_next_skill,
            "application_queue": self._application_queue(score_pct, experience_fit, best_next_skill),
            "score_story": (
                f"Wording contributes {semantic_points} points and listed skills contribute {skill_points} points "
                f"after the experience gate."
            ),
        }

    def _skill_lifts(self, job_weights, resume_index, s_sem, phi_exp, s_total) -> list[dict]:
        """How many match points each missing skill would add if the candidate gained it."""
        total_weight = sum(job_weights.values())
        if total_weight <= 0:
            return []
        lifts = []
        for skill, weight in job_weights.items():
            if str(skill).lower() in resume_index:
                continue
            matched_weight = sum(
                item_weight for item_skill, item_weight in job_weights.items()
                if str(item_skill).lower() in resume_index or str(item_skill).lower() == str(skill).lower()
            )
            upgraded_skill = matched_weight / total_weight
            upgraded_total = phi_exp * ((self.alpha * s_sem) + ((1.0 - self.alpha) * upgraded_skill))
            lifts.append({
                "skill": skill,
                "weight": weight,
                "score_lift_pct": round((upgraded_total - s_total) * 100.0, 2),
            })
        lifts.sort(key=lambda item: (-item["score_lift_pct"], -item["weight"], str(item["skill"])))
        return lifts

    def rank_jobs(self, resume_text, resume_skills, cand_years, jobs: list[dict]) -> tuple[list[dict], str]:
        """Score every opening once, encoding the resume a single time when the model is available."""
        descriptions = [str(job.get("description") or job.get("title") or "") for job in jobs]
        scores, mode = self.semantic_scores(resume_text, descriptions)
        ranked = []
        for job, score in zip(jobs, scores):
            evaluation = self.evaluate_job_match(
                resume_text=resume_text,
                job_text=job.get("description") or "",
                resume_skills=resume_skills,
                job_weights=job.get("weights") or {},
                cand_years=cand_years,
                req_years=job.get("req_years") or 0.0,
                override_s_sem=score,
            )
            evaluation["semantic_mode"] = mode
            ranked.append({"job": job, "evaluation": evaluation})
        ranked.sort(
            key=lambda item: (
                -item["evaluation"]["final_match_score_pct"],
                str(item["job"].get("company") or ""),
                str(item["job"].get("title") or ""),
            )
        )
        return ranked, mode

    @staticmethod
    def _application_queue(score_pct: float, experience_fit: str, best_next_skill: dict | None) -> str:
        lift = float(best_next_skill["score_lift_pct"]) if best_next_skill else 0.0
        if score_pct >= 65 and experience_fit in {"Experience-ready", "Near fit"}:
            return "Apply now"
        if score_pct < 65 and score_pct + lift >= 65:
            return "One skill away"
        if experience_fit == "Stretch role":
            return "Stretch"
        return "Build toward"


def apply_profile_filters(ranked: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split ranked openings into kept ones and exclusions with reasons.

    Kept items pass through untouched. Each excluded item keeps its score
    and gains a reason, so the caller can show what was filtered out and why
    instead of silently dropping it.
    """
    kept: list[dict] = []
    excluded: list[dict] = []
    for item in ranked:
        job = item.get("job") or {}
        company = str(job.get("company") or "").lower()
        text = f"{job.get('title') or ''} {job.get('description') or ''}"
        reason = None
        if any(name in company for name in EXCLUDED_EMPLOYERS):
            reason = "Excluded employer (Infosys/Accenture)"
        elif re.search(r"(?<![\w-])plm(?![\w-])", text.lower()):
            reason = "PLM role excluded (no PLM experience)"
        elif (job.get("req_years") or 0) >= MAX_BAND_YEARS:
            reason = "Above experience band (asks 7+ years)"
        if reason:
            excluded.append({
                "company_name": job.get("company") or "",
                "role_title": job.get("title") or "",
                "portal_url": job.get("url") or "",
                "required_experience_years": job.get("req_years") or 0.0,
                "score_pct": (item.get("evaluation") or {}).get("final_match_score_pct"),
                "reason": reason,
            })
        else:
            kept.append(item)
    return kept, excluded


def _tier_index(score_pct: float) -> int:
    if score_pct >= 80:
        return 1
    if score_pct >= 65:
        return 2
    if score_pct >= 50:
        return 3
    return 4


def build_market_brief(ranked: list[dict]) -> dict:
    """Find the skill that moves the most openings, plus an application triage."""
    apply_now = 0
    one_skill_away = 0
    buckets: dict[str, dict] = {}
    for item in ranked:
        evaluation = item["evaluation"]
        score = float(evaluation["final_match_score_pct"])
        queue = evaluation.get("application_queue")
        if queue == "Apply now":
            apply_now += 1
        elif queue == "One skill away":
            one_skill_away += 1
        company = str((item.get("job") or {}).get("company") or "")
        for lift in evaluation.get("skill_lifts") or []:
            skill = str(lift["skill"])
            gain = float(lift["score_lift_pct"])
            entry = buckets.setdefault(skill, {
                "skill": skill,
                "roles_improved": 0,
                "roles_promoted": 0,
                "total_lift_pct": 0.0,
                "best_lift_pct": 0.0,
                "best_company": company,
            })
            entry["roles_improved"] += 1
            entry["total_lift_pct"] = round(entry["total_lift_pct"] + gain, 2)
            if _tier_index(score + gain) < _tier_index(score):
                entry["roles_promoted"] += 1
            if gain > entry["best_lift_pct"]:
                entry["best_lift_pct"] = round(gain, 2)
                entry["best_company"] = company
    levers = sorted(
        buckets.values(),
        key=lambda entry: (
            -entry["roles_promoted"],
            -entry["roles_improved"],
            -entry["total_lift_pct"],
            entry["skill"],
        ),
    )
    return {
        "apply_now": apply_now,
        "one_skill_away": one_skill_away,
        "levers": levers[:5],
    }
