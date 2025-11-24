#!/usr/bin/env python3
"""
resume_folder_to_json.py

Reads .txt resumes from a folder, sends each file's text to a Gemini-like model
via OpenAI-style client, asks the model to return EXACTLY one JSON object with
the specified fields, and writes one .json per input .txt.

how to run this 
uv run agent.py --source txts --output extracted_json --workers 1 --delay 1


Includes:
- Timeout + jitter between API calls (to avoid rate limits)
- Retry with exponential backoff
- Concurrency with ThreadPoolExecutor
"""

import argparse
import json
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
import os

# ----------------- CONFIG -----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env")

BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

client = OpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)

EXPECTED_FIELDS = [
    "Name of Candidate",
    "Birth Date",
    "Marital Status",
    "Permanent Address",
    "Contact Number",
    "Email ID",
    "Education",
    "Total Years of Experience",
    "Experience Details",
    "Current Monthly Salary",
    "District",
    "Gender",
    "Present Address",
    "PAN Card",
    "Aadhar Card",
    "State",
    "Preferred Job Location"
]

PROMPT_TEMPLATE = """
You are a highly accurate structured-data extraction system.

Your task is to extract information from the resume text and return **ONE and ONLY ONE valid JSON object**.
Follow these rules STRICTLY:

-----------------------------------------
REQUIRED JSON FIELDS (keys must match EXACTLY):
{fields_list}
-----------------------------------------

### EXTRACTION RULES
- If a field is missing, unknown, not clearly stated, or ambiguous → set its value to null.
- Do NOT infer or guess anything not explicitly written in the resume.
- Preserve original text exactly where appropriate (addresses, names, IDs, etc.).
- For dates, use the format as written in the resume (do NOT convert).
- For numeric values (salary, experience), use the same format unless completely unparseable.
- “Experience Details” should be a list of past jobs or roles if available; otherwise null.
- Return **only** the JSON object. No comments, no explanation, no markdown, no text before or after.

### OUTPUT FORMAT (MANDATORY)
Return:
{{
  "Name of Candidate": ...,
  "Birth Date": ...,
  "Marital Status": ...,
  "Permanent Address": ...,
  "Contact Number": ...,
  "Email ID": ...,
  "Education": ...,
  "Total Years of Experience": ...,
  "Experience Details": ...,
  "Current Monthly Salary": ...,
  "District": ...,
  "Gender": ...,
  "Present Address": ...,
  "PAN Card": ...,
  "Aadhar Card": ...,
  "State": ...,
  "Preferred Job Location": ...
}}

No additional text or formatting is allowed.

-----------------------------------------
RESUME TEXT:
{resume_text}
-----------------------------------------
"""


# ----------------- HELPERS -----------------
def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(
        fields_list=json.dumps(EXPECTED_FIELDS, ensure_ascii=False),
        resume_text=text
    )

def sleep_for_rate_limit(base_delay: float):
    """
    Avoid rate-limit by sleeping a fixed delay + random jitter.
    Example: base_delay=1.0 → actual delay between 1.0 and 1.5 seconds
    """
    jitter = random.uniform(0.1, 0.5)
    time.sleep(base_delay + jitter)

def call_model_with_retry_and_delay(prompt: str,
                                    model: str,
                                    max_retries: int,
                                    base_delay: float):
    """
    Apply delay BEFORE each API call + retry logic.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        # --- IMPORTANT: delay to avoid rate limit ---
        sleep_for_rate_limit(base_delay)

        try:
            resp = client.chat.completions.create(
                model=model,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )

            return resp.choices[0].message.content

        except Exception as e:
            last_exc = e
            wait = 1.0 * (2 ** (attempt - 1))
            print(f"[WARN] API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait:.1f}s…")
            time.sleep(wait)

    raise last_exc


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    # direct parse first
    try:
        return json.loads(raw)
    except Exception:
        pass

    # try extracting JSON substring
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end+1])
        except Exception as e:
            raise ValueError(f"JSON substring failed: {e}\nRAW:\n{raw}")

    raise ValueError(f"No JSON found. Raw:\n{raw}")

def normalize_result(data: dict) -> dict:
    return {k: data.get(k) if k in data else None for k in EXPECTED_FIELDS}

# ----------------- WORKER -----------------
def process_one(txt_path: Path,
                out_dir: Path,
                model: str,
                base_delay: float,
                retries: int):

    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"file": str(txt_path), "ok": False, "error": f"Read error: {e}"}

    prompt = build_prompt(text)

    try:
        raw = call_model_with_retry_and_delay(prompt, model, retries, base_delay)
    except Exception as e:
        return {"file": str(txt_path), "ok": False, "error": f"API error: {e}"}

    # Parse JSON
    try:
        parsed = extract_json(raw)
    except Exception as e:
        raw_path = out_dir / f"{txt_path.stem}_raw_response.txt"
        raw_path.write_text(raw, encoding="utf-8")
        return {"file": str(txt_path), "ok": False, "error": f"JSON parse error: {e}"}

    normalized = normalize_result(parsed)
    out_file = out_dir / f"{txt_path.stem}.json"
    out_file.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"file": str(txt_path), "ok": True, "out": str(out_file)}

# ----------------- BATCH -----------------
def collect_txt_files(source: Path, recursive: bool):
    if source.is_file() and source.suffix.lower() == ".txt":
        return [source]
    if source.is_dir():
        pattern = "**/*.txt" if recursive else "*.txt"
        return sorted(list(source.glob(pattern)))
    return []

def batch_run(source: Path,
              output: Path,
              model: str,
              recursive: bool,
              workers: int,
              base_delay: float,
              retries: int):

    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    txt_files = collect_txt_files(source, recursive)
    if not txt_files:
        print("No .txt files found.")
        return

    print(f"Processing {len(txt_files)} files with {workers} worker(s). Delay per call = {base_delay}s")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one, f, output, model, base_delay, retries): f
            for f in txt_files
        }

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

            if res["ok"]:
                print(f"[OK]  {Path(res['file']).name} → {Path(res['out']).name}")
            else:
                print(f"[ERR] {Path(res['file']).name} → {res['error']}")

    print("\nSUMMARY:")
    ok_count = sum(1 for r in results if r["ok"])
    print(f"Successful: {ok_count}/{len(results)}")


# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TXT resumes to JSON using Gemini model.")
    parser.add_argument("--source", "-s", required=True)
    parser.add_argument("--output", "-o", default="extracted_json")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.0, help="Base delay between API calls (default 1.0s)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    batch_run(
        Path(args.source),
        Path(args.output),
        args.model,
        recursive=not args.no_recursive,
        workers=args.workers,
        base_delay=args.delay,
        retries=args.retries
    )

