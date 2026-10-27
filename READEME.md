# Autonomous Multi-Agent Framework for Semantic Resume-to-Job Alignment
**Course Code:** `CAPOL601R01` — Project Work & Viva Voce (SASTRA Deemed University, MCA Semester IV)

## Quick Start (Stage 2 Backend Prototype)

### 1. Local Virtual Environment Execution
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# Run Verification Tests (Validates Mathematical Model & Case Studies)
pytest -v

# Start FastAPI Server
uvicorn app.main:app --reload --port 8000
```
Open **`http://localhost:8000/docs`** in your browser to interact with the Swagger UI and test `/api/v1/evaluate-matches` and `/api/v1/parse-resume-pdf`.
