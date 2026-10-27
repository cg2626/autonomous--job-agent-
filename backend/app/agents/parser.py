import fitz  # PyMuPDF
import re
import json

SKILL_TAXONOMY_WEIGHTS = {
    "python": 2.0, "docker": 1.8, "kubernetes": 2.0, "terraform": 1.5,
    "fastapi": 1.2, "aws": 1.8, "azure": 1.5, "linux": 1.5, "ci/cd": 1.8,
    "git": 1.2, "jenkins": 1.5, "ansible": 1.5, "prometheus": 1.2, "grafana": 1.2,
    "react": 1.8, "node.js": 1.8, "sql": 1.5, "postgresql": 1.5, "java": 1.8,
    "spring": 1.5, "c++": 1.8, "javascript": 1.5, "typescript": 1.5, "html": 1.0,
    "css": 1.0, "machine learning": 2.0, "data analysis": 1.8, "pandas": 1.5,
    "numpy": 1.5, "django": 1.5, "flask": 1.2, "rest apis": 1.5,
    "go": 1.8, "golang": 1.8, "rust": 1.8, "c#": 1.8, ".net": 1.8,
    "php": 1.5, "ruby": 1.5, "kotlin": 1.5, "swift": 1.5, "android": 1.8,
    "ios": 1.8, "flutter": 1.8, "react native": 1.8, "vue": 1.8, "angular": 1.8,
    "next.js": 1.8, "express": 1.5, "graphql": 1.5, "mongodb": 1.5,
    "redis": 1.5, "mysql": 1.5, "oracle": 1.5, "snowflake": 1.8,
    "spark": 1.8, "airflow": 1.8, "tensorflow": 1.8, "pytorch": 1.8,
    "scikit-learn": 1.8, "nlp": 1.8, "computer vision": 1.8, "excel": 1.2,
    "tableau": 1.5, "power bi": 1.5, "salesforce": 1.5, "sap": 1.5,
    "workday": 1.5, "jira": 1.0, "figma": 1.2, "ui/ux": 1.5,
    "product management": 1.5, "project management": 1.5, "agile": 1.0,
    "scrum": 1.0, "seo": 1.2, "content marketing": 1.2, "accounting": 1.2
}


def extract_skills_from_text(text: str) -> list[str]:
    """Extract known skills without injecting a DevOps profile for unknown resumes."""
    text_lower = text.lower()
    return [
        skill for skill in SKILL_TAXONOMY_WEIGHTS
        if re.search(r"(?<![\w-])" + re.escape(skill) + r"(?![\w-])", text_lower)
    ]

class ResumeParserAgent:
    """Dynamic resume parser agent that extracts text, experience tenure, and skills from ANY uploaded PDF resume."""
    def parse_pdf_bytes(self, pdf_bytes: bytes) -> dict:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
            
        found_skills = extract_skills_from_text(full_text)

        exp_match = re.search(r'(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)', full_text.lower())
        exp_years = float(exp_match.group(1)) if exp_match else 2.0
        
        return {
            "raw_text": full_text,
            "exp_years": exp_years,
            "skills": found_skills
        }


class ResumeAndJobParserAgent(ResumeParserAgent):
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    def extract_skills_from_text(self, text: str) -> set:
        return set(extract_skills_from_text(text))

    def extract_years_of_experience(self, text: str, default_years: float = 2.0) -> float:
        exp_match = re.search(r'(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)', text.lower())
        return float(exp_match.group(1)) if exp_match else default_years
