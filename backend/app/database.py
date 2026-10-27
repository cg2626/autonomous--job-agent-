import sqlite3
import json
from datetime import datetime

DB_PATH = "job_agent.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            raw_text TEXT,
            exp_years REAL,
            skills_json TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_code TEXT UNIQUE,
            title TEXT,
            company TEXT,
            url TEXT,
            description TEXT,
            req_years REAL,
            weights_json TEXT,
            location TEXT,
            country TEXT,
            source TEXT,
            posted_at TEXT,
            is_startup INTEGER DEFAULT 0
        )
    ''')

    # Keep existing local databases usable after adding nationwide discovery
    # metadata to the jobs table.
    existing_job_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(jobs)").fetchall()
    }
    for column, definition in {
        "location": "TEXT",
        "country": "TEXT",
        "source": "TEXT",
        "posted_at": "TEXT",
        "is_startup": "INTEGER DEFAULT 0",
    }.items():
        if column not in existing_job_columns:
            cursor.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            job_title TEXT,
            company TEXT,
            s_sem_pct REAL,
            s_skill_pct REAL,
            phi_exp REAL,
            s_total_pct REAL,
            match_tier TEXT,
            missing_json TEXT,
            created_at TEXT
        )
    ''')
    
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ("deepak", "ef92b778bafe774e896db76ef7af8f37", "candidate", datetime.now().isoformat())
        )
        
    cursor.execute("SELECT COUNT(*) FROM resumes")
    if cursor.fetchone()[0] == 0:
        sample_skills = json.dumps(["python", "docker", "kubernetes", "terraform", "aws", "jenkins", "ansible"])
        cursor.execute(
            "INSERT INTO resumes (user_id, filename, raw_text, exp_years, skills_json) VALUES (?, ?, ?, ?, ?)",
            (1, "Ram_Deepak_Resume.pdf", "DevOps Engineer experienced in Python, Docker, Kubernetes, Terraform, AWS, Jenkins, and Ansible.", 2.5, sample_skills)
        )

    cursor.execute("SELECT COUNT(*) FROM jobs")
    if cursor.fetchone()[0] == 0:
        default_weights = json.dumps({"python": 2.0, "docker": 1.8, "kubernetes": 2.0, "terraform": 1.5, "fastapi": 1.2})
        cursor.execute(
            "INSERT INTO jobs (job_code, title, company, url, description, req_years, weights_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("JOB-DEVOPS-01", "Senior DevOps Engineer", "CloudScale Inc", "https://example.com/careers/devops",
             "Seeking a DevOps Engineer with expertise in Python, Docker, Kubernetes, Terraform, and FastAPI for cloud infrastructure automation.",
             3.0, default_weights)
        )
        
    conn.commit()
    conn.close()
