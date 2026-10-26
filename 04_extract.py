import os
import argparse
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

MODEL="claude-haiku-5-5"
MAX_TOKENS = 1024

class JobPosting (BaseModel):
    title : str = Field(description="The precise job title.")
    company : str = Field(description="Name of the hiring company.")
    location: str = Field(description="City, State/Country or 'Remote'.")
    employment_type: str = Field(
        description="Must be exactly one of: 'Full-time', 'Part-time', 'Contract', 'Internship'."
    )
    base_salary_min: Optional[float] = Field(None, description="Minimum base salary if mentioned.")
    base_salary_max: Optional[float] = Field(None, description="Maximum base salary if mentioned.")
    required_skills: List[str] = Field(description="List of technical or core skills explicitly mentioned.")
    experience_years_required: Optional[int] = Field(None, description="Minimum years of experience requested.")

def call_model(client, model: str, system: str | None, user: str) -> JobPosting: 
    kwargs = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.parse(**kwargs,output_format=JobPosting)
    return resp.parsed_output

def parse_job_posting_with_retry(raw_text: str, max_retries: int = 3) -> JobPosting:
    """
    Parses unstructured job text into a validated JobPosting Pydantic model.
    If validation fails, it feeds the error back to the LLM to self-correct.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only write prompts, no API calls")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    load_dotenv()

    client = None

    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set - switching to dry run.")
        else:
            try:
                import anthropic
                client = anthropic.Anthropic()
            except ImportError:
                print("`anthropic` package not installed (pip install anthropic) - switching to dry run.")

    if client:
        system = "Extract the job posting details from the user's text. Use null for any field the text doesn't state."
        raw = call_model(client, args.model, system, raw_text)

    return raw

if __name__ == "__main__":

    sample_job_text = """
        Staff Data Engineer Wanted!
        We are looking for a remote Staff Data Engineer to join our growing platform group at NexusData corp.
        You should have at least 7+ years of experience engineering data platforms.
        Key skills we look for: PySpark, Databricks Delta Live Tables, Terraform, and Python.
        Salary range for this position is between $140,000 and $180,000 base pay depending on experience.
        This is a full time position.
    """
    
    abc = parse_job_posting_with_retry(sample_job_text)
    print(abc.model_dump_json(indent=2))