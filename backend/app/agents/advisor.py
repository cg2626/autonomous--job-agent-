class SkillGapAdvisorAgent:
    """Turn a real skill gap into a next step without inventing experience."""

    def generate_advice(
        self,
        resume_skills: list,
        job_weights: dict,
        job_title: str = "",
        score_lifts: list | None = None,
    ) -> dict:
        return self._build_advice(resume_skills, job_weights, job_title, score_lifts)

    def generate_gap_recommendations(
        self,
        missing_skills: list,
        job_skills_weighted: dict,
        job_title: str = "",
    ) -> dict:
        missing = {str(skill).lower() for skill in missing_skills}
        if isinstance(job_skills_weighted, dict):
            resume_skills = [
                skill for skill in job_skills_weighted
                if str(skill).lower() not in missing and str(skill).lower() != "general"
            ]
        else:
            resume_skills = []
        return self._build_advice(resume_skills, job_skills_weighted, job_title, None)

    def _build_advice(self, resume_skills, job_weights, job_title, score_lifts) -> dict:
        resume_index = {str(skill).lower() for skill in resume_skills}
        if isinstance(job_weights, dict):
            specific = {
                skill: weight for skill, weight in job_weights.items()
                if str(skill).lower() != "general"
            }
            matched_skills = [skill for skill in specific if str(skill).lower() in resume_index]
            missing_skills = [skill for skill in specific if str(skill).lower() not in resume_index]
        else:
            matched_skills = []
            missing_skills = list(resume_skills)

        lift_by_skill = {
            str(item.get("skill", "")).lower(): item
            for item in (score_lifts or [])
            if isinstance(item, dict)
        }
        recommendations = []
        for skill in missing_skills:
            lift = lift_by_skill.get(str(skill).lower())
            if lift and lift.get("score_lift_pct"):
                recommendations.append(
                    f"A project that uses {skill} would raise this match by {lift['score_lift_pct']} points."
                )
            else:
                recommendations.append(
                    f"Close the gap in {skill} with a project you can talk through in an interview."
                )

        return {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "recommendations": recommendations,
            "ats_bullet_points": [self._honest_bullet(matched_skills, job_title)],
        }

    @staticmethod
    def _honest_bullet(matched_skills: list, job_title: str) -> str:
        role = job_title or "the target role"
        if not matched_skills:
            return (
                f"Add a small project that demonstrates the core tools for {role} "
                "before listing them as experience."
            )
        shown = ", ".join(str(skill) for skill in matched_skills[:4])
        return f"Delivered work relevant to {role} using {shown}."
