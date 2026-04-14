#!/usr/bin/env python3
"""
Autograder for HW1 Part 4: Convert WikiHow to PDDL (Extra Credit)

Grading Rubric (20 pts max, before penalty):
  - Domain Definition:   5 pts
  - Problem Definition:  5 pts
  - JSON Annotations:    3 pts
  - Writeup (7 Qs):      7 pts
  - Penalty:             -0.5 if no article rationale

Usage:
    python autograder.py /path/to/submissions [--timeout 30] [--verbose]
    python autograder.py /path/to/submissions --llm [--model claude-haiku-4-5-20251001]

Each student submission should be a subfolder inside the submissions directory.
Outputs per-student Gradescope JSON (results.json) and a summary CSV.

When --llm is enabled, the Writeup and JSON Annotations components are graded
by Claude instead of keyword heuristics, producing much more accurate scores.

Dependencies:
    - PyPDF2 (pip install PyPDF2)
    - anthropic (pip install anthropic)  — only needed with --llm
    - PDDL parser tools from part-3-planning (auto-imported via sys.path)
"""

import argparse
import csv
import json
import os
import re
import signal
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the part-3-planning directory to sys.path so we can import PDDL tools
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLANNING_DIR = SCRIPT_DIR.parent / "part-3-planning"
if str(PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNING_DIR))

from PDDL import PDDL_Parser  # noqa: E402
from planner import Planner    # noqa: E402

# ---------------------------------------------------------------------------
# PDF text extraction (optional dependency)
# ---------------------------------------------------------------------------
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# ---------------------------------------------------------------------------
# Anthropic SDK (optional — needed only with --llm flag)
# ---------------------------------------------------------------------------
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Action Castle baseline actions that don't count toward the 10 required
ACTION_CASTLE_BASELINE = {"go", "get", "drop"}

# Keyword sets for writeup heuristic grading (question index -> keywords)
WRITEUP_KEYWORDS = {
    0: {  # Q1: What wikiHow article did you pick and why?
        "keywords": [
            "wikihow", "article", "chose", "picked", "selected", "choose",
            "why i", "why we", "interesting", "because",
        ],
        "label": "Q1: Article choice and rationale",
    },
    1: {  # Q2: What portions did you translate?
        "keywords": [
            "step", "portion", "translate", "translat", "section", "part",
            "convert", "selected", "focus",
        ],
        "label": "Q2: Portions translated to PDDL",
    },
    2: {  # Q3: Examples of actions, types, predicates
        "keywords": [
            "action", "predicate", "type", "schema", ":action", "parameter",
            "precondition", "effect", "example",
        ],
        "label": "Q3: Actions/types/predicates examples",
    },
    3: {  # Q4: Goal, initial state, solution
        "keywords": [
            "goal", "initial state", "init", "solution", "plan",
            "sequence", "start state", "reach",
        ],
        "label": "Q4: Goal/initial state/solution",
    },
    4: {  # Q5: PDDL limitations
        "keywords": [
            "limitation", "difficult", "cannot", "challenge", "shortcoming",
            "unable", "hard to", "doesn't support", "does not support",
            "drawback", "restrict", "express",
        ],
        "label": "Q5: PDDL limitations",
    },
    5: {  # Q6: Text adventure game potential
        "keywords": [
            "game", "adventure", "interactive fiction", "text adventure",
            "challenge", "player", "puzzle", "narrative",
        ],
        "label": "Q6: Text adventure game discussion",
    },
    6: {  # Q7: GPT-3 / LLM discussion
        "keywords": [
            "gpt", "openai", "language model", "fine-tune", "fine tune",
            "llm", "automat", "large language", "prompt", "chatgpt",
            "generative", "ai model", "transformer",
        ],
        "label": "Q7: LLM/GPT-3 discussion",
    },
}

KEYWORD_THRESHOLD = 2  # Minimum keyword matches to award a writeup question


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------
class PlannerTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PlannerTimeout("Planner exceeded time limit")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(submission_dir: str) -> dict:
    """Scan a student submission folder and classify files."""
    result = {
        "domain_files": [],
        "problem_files": [],
        "json_files": [],
        "pdf_files": [],
        "all_pddl": [],
    }
    for root, _dirs, files in os.walk(submission_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            if ext == "pddl":
                result["all_pddl"].append(fpath)
                try:
                    with open(fpath, "r", errors="replace") as f:
                        content = f.read().lower()
                    if re.search(r"\(\s*define\s*\(\s*domain\b", content):
                        result["domain_files"].append(fpath)
                    elif re.search(r"\(\s*define\s*\(\s*problem\b", content):
                        result["problem_files"].append(fpath)
                    else:
                        # Unclassified PDDL — try both
                        result["problem_files"].append(fpath)
                except Exception:
                    pass
            elif ext == "json":
                result["json_files"].append(fpath)
            elif ext == "pdf":
                result["pdf_files"].append(fpath)
    return result


# ---------------------------------------------------------------------------
# Domain grading (5 pts)
# ---------------------------------------------------------------------------

def grade_domain(domain_files: list, verbose: bool = False) -> dict:
    """Grade the domain definition.

    Breakdown (5 pts):
        - Domain file exists and parses:            1.0
        - Has :types with >= 2 types:               0.5
        - Has :predicates with typed arguments:      0.5
        - Predicates have comments in source:        0.5
        - Action count (excl. baseline) >= 10:       2.5  (0.25/action)
    """
    score = 0.0
    max_score = 5.0
    details = []

    if not domain_files:
        details.append("FAIL: No domain PDDL file found.")
        return {"score": score, "max_score": max_score, "details": details}

    domain_path = domain_files[0]
    if len(domain_files) > 1:
        details.append(
            f"NOTE: Found {len(domain_files)} domain files; grading first: "
            f"{os.path.basename(domain_path)}"
        )

    # --- Parse ---
    parser = PDDL_Parser()
    try:
        parser.parse_domain(domain_path)
        score += 1.0
        details.append("PASS [1.0/1.0]: Domain file parses successfully.")
    except Exception as e:
        details.append(f"FAIL [0.0/1.0]: Domain parse error — {e}")
        return {"score": score, "max_score": max_score, "details": details}

    # --- Types ---
    all_types = set()
    for parent, children in parser.types.items():
        all_types.add(parent)
        all_types.update(children)
    # 'object' is implicit; don't count it
    all_types.discard("object")

    if len(all_types) >= 2:
        score += 0.5
        details.append(
            f"PASS [0.5/0.5]: Types defined ({len(all_types)}): "
            f"{', '.join(sorted(all_types)[:10])}"
            + ("..." if len(all_types) > 10 else "")
        )
    else:
        details.append(
            f"FAIL [0.0/0.5]: Fewer than 2 types defined (found {len(all_types)})."
        )

    # --- Predicates with typed arguments ---
    if parser.predicates:
        has_typed = any(
            any(t != "object" for t in args.values())
            for args in parser.predicates.values()
            if args
        )
        if has_typed:
            score += 0.5
            details.append(
                f"PASS [0.5/0.5]: {len(parser.predicates)} predicates with typed arguments."
            )
        else:
            score += 0.25
            details.append(
                f"PARTIAL [0.25/0.5]: {len(parser.predicates)} predicates found but "
                f"none have typed arguments (all default to 'object')."
            )
    else:
        details.append("FAIL [0.0/0.5]: No predicates defined.")

    # --- Predicate comments in source ---
    try:
        with open(domain_path, "r", errors="replace") as f:
            source = f.read()
        # Look for predicate lines followed by a comment (;)
        predicate_section = re.search(
            r"\(:predicates(.*?)\)", source, re.DOTALL | re.IGNORECASE
        )
        if predicate_section:
            section_text = predicate_section.group(1)
            comment_count = len(re.findall(r";", section_text))
            if comment_count >= max(1, len(parser.predicates) // 2):
                score += 0.5
                details.append(
                    f"PASS [0.5/0.5]: Predicates have descriptive comments "
                    f"({comment_count} comments found)."
                )
            elif comment_count > 0:
                score += 0.25
                details.append(
                    f"PARTIAL [0.25/0.5]: Only {comment_count} predicate comments "
                    f"found (expected ~{len(parser.predicates)})."
                )
            else:
                details.append(
                    "FAIL [0.0/0.5]: No comments found in predicates section."
                )
        else:
            details.append(
                "FAIL [0.0/0.5]: Could not locate :predicates section for comment check."
            )
    except Exception:
        details.append("FAIL [0.0/0.5]: Could not read source for comment check.")

    # --- Action schemas ---
    action_names = [a.name for a in parser.actions]
    novel_actions = [a for a in action_names if a not in ACTION_CASTLE_BASELINE]
    novel_count = len(novel_actions)
    action_pts = min(2.5, novel_count * 0.25)
    score += action_pts
    baseline_found = [a for a in action_names if a in ACTION_CASTLE_BASELINE]
    details.append(
        f"{'PASS' if novel_count >= 10 else 'PARTIAL'} "
        f"[{action_pts:.2f}/2.5]: {novel_count} novel action(s) "
        f"(need 10; baseline excluded: {baseline_found or 'none'})."
    )
    if verbose:
        for a in action_names:
            tag = " [baseline]" if a in ACTION_CASTLE_BASELINE else ""
            details.append(f"  - {a}{tag}")

    return {
        "score": round(score, 2),
        "max_score": max_score,
        "details": details,
        "parser": parser,
        "domain_path": domain_path,
    }


# ---------------------------------------------------------------------------
# Problem grading (5 pts)
# ---------------------------------------------------------------------------

def grade_problems(
    problem_files: list,
    domain_result: dict,
    timeout: int = 30,
    verbose: bool = False,
) -> dict:
    """Grade problem definitions.

    Breakdown (5 pts):
        - >= 3 problem files found:                 0.5
        - Per problem (up to 3):
            - Parses with matching domain:           0.5
            - Has non-empty :init and :goal:         0.25
            - Goal reachable by planner:             0.75
        Per-problem total: 1.5 x 3 = 4.5
    """
    score = 0.0
    max_score = 5.0
    details = []

    if not problem_files:
        details.append("FAIL: No problem PDDL files found.")
        return {"score": score, "max_score": max_score, "details": details}

    # Count credit
    if len(problem_files) >= 3:
        score += 0.5
        details.append(
            f"PASS [0.5/0.5]: Found {len(problem_files)} problem file(s) (need >= 3)."
        )
    else:
        details.append(
            f"FAIL [0.0/0.5]: Only {len(problem_files)} problem file(s) found (need >= 3)."
        )

    domain_path = domain_result.get("domain_path")
    if not domain_path:
        details.append("SKIP: Cannot grade problems without a valid domain.")
        return {"score": score, "max_score": max_score, "details": details}

    # Grade up to 3 problems (try all, take best 3)
    problem_scores = []
    for pf in problem_files:
        ps = _grade_single_problem(pf, domain_path, timeout, verbose)
        problem_scores.append((ps["total"], pf, ps))

    # Sort by score descending, take best 3
    problem_scores.sort(key=lambda x: x[0], reverse=True)
    graded = problem_scores[:3]

    for total, pf, ps in graded:
        score += total
        fname = os.path.basename(pf)
        details.append(f"--- {fname} [{total:.2f}/1.5] ---")
        details.extend(ps["details"])

    if len(problem_scores) > 3:
        skipped = [os.path.basename(pf) for _, pf, _ in problem_scores[3:]]
        details.append(
            f"NOTE: {len(skipped)} additional problem file(s) not graded: "
            + ", ".join(skipped)
        )

    return {"score": round(score, 2), "max_score": max_score, "details": details}


def _grade_single_problem(
    problem_path: str, domain_path: str, timeout: int, verbose: bool
) -> dict:
    """Grade a single problem file. Returns up to 1.5 pts."""
    total = 0.0
    details = []

    # --- Parse ---
    parser = PDDL_Parser()
    try:
        parser.parse_domain(domain_path)
        parser.parse_problem(problem_path)
        total += 0.5
        details.append("  PASS [0.5/0.5]: Parses successfully with matching domain.")
    except Exception as e:
        details.append(f"  FAIL [0.0/0.5]: Parse error — {e}")
        return {"total": total, "details": details}

    # --- Init and goal ---
    has_init = len(parser.state) > 0
    has_goal = len(parser.positive_goals) > 0 or len(parser.negative_goals) > 0
    if has_init and has_goal:
        total += 0.25
        details.append(
            f"  PASS [0.25/0.25]: Has init ({len(parser.state)} predicates) "
            f"and goal ({len(parser.positive_goals)} pos, "
            f"{len(parser.negative_goals)} neg)."
        )
    else:
        missing = []
        if not has_init:
            missing.append("init")
        if not has_goal:
            missing.append("goal")
        details.append(
            f"  FAIL [0.0/0.25]: Missing {' and '.join(missing)}."
        )

    # --- Solvability ---
    if not (has_init and has_goal):
        details.append("  SKIP [0.0/0.75]: Cannot check solvability without init+goal.")
        return {"total": total, "details": details}

    planner = Planner()
    timed_out = False
    plan = None

    # Use signal alarm for timeout (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        plan = planner.solve(domain_path, problem_path)
        signal.alarm(0)
    except PlannerTimeout:
        timed_out = True
    except Exception as e:
        details.append(
            f"  FAIL [0.0/0.75]: Planner error — {e}"
        )
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        return {"total": total, "details": details}
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)

    if timed_out:
        # Fallback: award partial credit since it parses and has init+goal
        total += 0.25
        details.append(
            f"  PARTIAL [0.25/0.75]: Planner timed out after {timeout}s. "
            f"Partial credit for valid structure. NEEDS MANUAL REVIEW."
        )
    elif plan is not None:
        total += 0.75
        plan_summary = (
            ", ".join(
                a.name + " " + " ".join(a.parameters)
                for a in plan[:5]
            )
            + ("..." if len(plan) > 5 else "")
        )
        details.append(
            f"  PASS [0.75/0.75]: Goal reachable! Plan length: {len(plan)}."
        )
        if verbose:
            details.append(f"    Plan: {plan_summary}")
    else:
        details.append(
            "  FAIL [0.0/0.75]: Planner found no solution (goal unreachable)."
        )

    return {"total": total, "details": details}


# ---------------------------------------------------------------------------
# JSON annotations grading (3 pts)
# ---------------------------------------------------------------------------

def grade_json(json_files: list, domain_result: dict, verbose: bool = False) -> dict:
    """Grade JSON annotation file.

    Breakdown (3 pts):
        - File exists and is valid JSON:             1.0
        - References PDDL elements from domain:      1.0
        - Contains NL text / wikiHow mentions:        1.0
    """
    score = 0.0
    max_score = 3.0
    details = []

    if not json_files:
        details.append("FAIL: No JSON annotation file found.")
        return {"score": score, "max_score": max_score, "details": details}

    json_path = json_files[0]
    if len(json_files) > 1:
        details.append(
            f"NOTE: Found {len(json_files)} JSON files; grading first: "
            f"{os.path.basename(json_path)}"
        )

    # --- Valid JSON ---
    try:
        with open(json_path, "r", errors="replace") as f:
            data = json.load(f)
        score += 1.0
        details.append("PASS [1.0/1.0]: Valid JSON file.")
    except json.JSONDecodeError as e:
        details.append(f"FAIL [0.0/1.0]: Invalid JSON — {e}")
        return {"score": score, "max_score": max_score, "details": details}
    except Exception as e:
        details.append(f"FAIL [0.0/1.0]: Could not read file — {e}")
        return {"score": score, "max_score": max_score, "details": details}

    # Flatten JSON to a single string for keyword searching
    json_str = json.dumps(data).lower()

    # --- PDDL element references ---
    parser = domain_result.get("parser")
    pddl_elements = set()
    if parser:
        pddl_elements.update(a.name for a in parser.actions)
        pddl_elements.update(parser.predicates.keys())
        for children in parser.types.values():
            pddl_elements.update(children)
        for children in parser.objects.values():
            pddl_elements.update(children)

    if pddl_elements:
        matched = [e for e in pddl_elements if e in json_str]
        if len(matched) >= 3:
            score += 1.0
            details.append(
                f"PASS [1.0/1.0]: JSON references {len(matched)} PDDL elements "
                f"(e.g. {', '.join(sorted(matched)[:5])})."
            )
        elif matched:
            score += 0.5
            details.append(
                f"PARTIAL [0.5/1.0]: JSON references only {len(matched)} PDDL element(s): "
                f"{', '.join(sorted(matched))}."
            )
        else:
            details.append(
                "FAIL [0.0/1.0]: No PDDL element names found in JSON."
            )
    else:
        # No domain parsed — can't cross-reference, give benefit of the doubt
        score += 0.5
        details.append(
            "PARTIAL [0.5/1.0]: Could not cross-reference with domain "
            "(domain didn't parse). Manual review needed."
        )

    # --- Natural language / wikiHow mentions ---
    wikihow_indicators = [
        "wikihow", "http", "www.", "step ", "how to ",
        "article", "description", "mention", "text",
    ]
    nl_matches = sum(1 for kw in wikihow_indicators if kw in json_str)
    if nl_matches >= 2:
        score += 1.0
        details.append(
            f"PASS [1.0/1.0]: JSON contains natural language / wikiHow references "
            f"({nl_matches} indicators found)."
        )
    elif nl_matches >= 1:
        score += 0.5
        details.append(
            f"PARTIAL [0.5/1.0]: Limited NL / wikiHow references ({nl_matches} indicator)."
        )
    else:
        details.append(
            "FAIL [0.0/1.0]: No natural language or wikiHow references found in JSON."
        )

    return {"score": round(score, 2), "max_score": max_score, "details": details}


# ---------------------------------------------------------------------------
# Writeup grading (7 pts — keyword heuristics)
# ---------------------------------------------------------------------------

def grade_writeup(pdf_files: list, verbose: bool = False) -> dict:
    """Grade the writeup PDF using keyword heuristics.

    Awards 1 pt per question if >= KEYWORD_THRESHOLD relevant keywords found.
    """
    score = 0.0
    max_score = 7.0
    details = []
    q1_passed = False

    if not pdf_files:
        details.append("FAIL: No PDF writeup found.")
        return {
            "score": score,
            "max_score": max_score,
            "details": details,
            "q1_passed": q1_passed,
        }

    pdf_path = pdf_files[0]
    if len(pdf_files) > 1:
        details.append(
            f"NOTE: Found {len(pdf_files)} PDFs; grading first: "
            f"{os.path.basename(pdf_path)}"
        )

    # --- Extract text ---
    if not HAS_PYPDF2:
        details.append(
            "WARNING: PyPDF2 not installed. Cannot extract PDF text. "
            "All 7 question points flagged for MANUAL REVIEW."
        )
        return {
            "score": score,
            "max_score": max_score,
            "details": details,
            "q1_passed": q1_passed,
        }

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text_lower = text.lower()
    except Exception as e:
        details.append(f"WARNING: Could not extract PDF text — {e}. MANUAL REVIEW needed.")
        return {
            "score": score,
            "max_score": max_score,
            "details": details,
            "q1_passed": q1_passed,
        }

    if len(text.strip()) < 100:
        details.append(
            f"WARNING: PDF text very short ({len(text.strip())} chars). "
            "May be image-based. MANUAL REVIEW recommended."
        )

    # --- Check each question ---
    for q_idx in range(7):
        q_info = WRITEUP_KEYWORDS[q_idx]
        kw_list = q_info["keywords"]
        label = q_info["label"]
        matches = [kw for kw in kw_list if kw in text_lower]
        if len(matches) >= KEYWORD_THRESHOLD:
            score += 1.0
            if q_idx == 0:
                q1_passed = True
            details.append(
                f"PASS [1.0/1.0] {label}: "
                f"Found {len(matches)} keyword(s): {', '.join(matches[:4])}"
            )
        elif matches:
            score += 0.5
            details.append(
                f"PARTIAL [0.5/1.0] {label}: "
                f"Only {len(matches)} keyword match(es): {', '.join(matches)}. "
                f"MANUAL REVIEW recommended."
            )
        else:
            details.append(
                f"FAIL [0.0/1.0] {label}: No keyword matches. MANUAL REVIEW needed."
            )

    return {
        "score": round(score, 2),
        "max_score": max_score,
        "details": details,
        "q1_passed": q1_passed,
    }


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _call_claude(client, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    """Send a prompt to Claude and return the text response."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _extract_pdf_text(pdf_files: list) -> tuple:
    """Extract text from the first PDF. Returns (text, pdf_path, error_msg)."""
    if not pdf_files:
        return None, None, "No PDF writeup found."
    pdf_path = pdf_files[0]
    if not HAS_PYPDF2:
        return None, pdf_path, "PyPDF2 not installed — cannot extract PDF text."
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if len(text.strip()) < 50:
            return None, pdf_path, "PDF text too short — may be image-based."
        return text, pdf_path, None
    except Exception as e:
        return None, pdf_path, f"Could not extract PDF text — {e}"


# ---------------------------------------------------------------------------
# LLM-based writeup grading (7 pts)
# ---------------------------------------------------------------------------

WRITEUP_SYSTEM_PROMPT = """\
You are an expert teaching assistant grading student writeups for a university \
AI course assignment. The assignment asked students to convert a wikiHow article \
into PDDL (Planning Domain Definition Language).

You will receive the extracted text of a student's PDF writeup. Grade it against \
the 7 required questions below. For each question, assign a score and provide a \
brief justification.

SCORING per question:
  1.0 — The student clearly and substantively addresses the question.
  0.5 — The student partially addresses it (mentioned but shallow / incomplete).
  0.0 — The question is not addressed at all.

THE 7 REQUIRED QUESTIONS:
  Q1: What wikiHow article did you pick and why?
  Q2: What portions of the article did you select to translate to PDDL?
  Q3: Give examples of the actions, types, and predicates used in your domain.
  Q4: Explain what goal you selected, and give the initial state and solution.
  Q5: What limitations of PDDL make it difficult to precisely convert a wikiHow \
      description into PDDL?
  Q6: Could your PDDL be used as an interesting challenge for a text-adventure \
      game? If so, how? If not, what would be needed?
  Q7: Discuss how you might use GPT-3 (or another LLM) to automatically or \
      semi-automatically convert a wikiHow article to PDDL.

ALSO determine:
  - has_article_rationale (bool): Did the student explain *why* they picked \
    their article (not just *which* one)? This is used for a separate -0.5 penalty.

Respond ONLY with valid JSON in this exact format (no markdown fences):
{
  "questions": [
    {"question": "Q1", "score": 1.0, "justification": "..."},
    {"question": "Q2", "score": 0.5, "justification": "..."},
    {"question": "Q3", "score": 1.0, "justification": "..."},
    {"question": "Q4", "score": 1.0, "justification": "..."},
    {"question": "Q5", "score": 0.0, "justification": "..."},
    {"question": "Q6", "score": 1.0, "justification": "..."},
    {"question": "Q7", "score": 0.5, "justification": "..."}
  ],
  "has_article_rationale": true
}"""


def grade_writeup_llm(
    pdf_files: list,
    client,
    model: str,
    verbose: bool = False,
) -> dict:
    """Grade the writeup PDF using Claude for natural-language evaluation."""
    score = 0.0
    max_score = 7.0
    details = []
    q1_passed = False

    text, pdf_path, error = _extract_pdf_text(pdf_files)
    if error:
        details.append(f"WARNING: {error}. Falling back to keyword heuristics.")
        return grade_writeup(pdf_files, verbose=verbose)

    if len(pdf_files) > 1:
        details.append(
            f"NOTE: Found {len(pdf_files)} PDFs; grading first: "
            f"{os.path.basename(pdf_path)}"
        )

    # Truncate to ~12k chars to stay within token budget for smaller models
    truncated = text[:12000]
    if len(text) > 12000:
        details.append(
            f"NOTE: PDF text truncated from {len(text)} to 12000 chars for LLM evaluation."
        )

    user_prompt = f"Here is the student's writeup text:\n\n{truncated}"

    try:
        raw = _call_claude(client, model, WRITEUP_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
    except json.JSONDecodeError:
        details.append(
            f"WARNING: LLM returned non-JSON response. Falling back to keyword heuristics.\n"
            f"Raw response (first 300 chars): {raw[:300]}"
        )
        return grade_writeup(pdf_files, verbose=verbose)
    except Exception as e:
        details.append(
            f"WARNING: LLM call failed ({e}). Falling back to keyword heuristics."
        )
        return grade_writeup(pdf_files, verbose=verbose)

    details.append("[Graded by LLM]")

    # Parse structured response
    q_labels = [
        "Q1: Article choice and rationale",
        "Q2: Portions translated to PDDL",
        "Q3: Actions/types/predicates examples",
        "Q4: Goal/initial state/solution",
        "Q5: PDDL limitations",
        "Q6: Text adventure game discussion",
        "Q7: LLM/GPT-3 discussion",
    ]

    questions = result.get("questions", [])
    for i, label in enumerate(q_labels):
        if i < len(questions):
            q = questions[i]
            q_score = float(q.get("score", 0))
            q_score = max(0.0, min(1.0, q_score))  # clamp
            justification = q.get("justification", "No justification provided.")
            score += q_score
            if i == 0 and q_score >= 0.5:
                q1_passed = True
            tag = "PASS" if q_score == 1.0 else ("PARTIAL" if q_score > 0 else "FAIL")
            details.append(f"{tag} [{q_score}/1.0] {label}: {justification}")
        else:
            details.append(f"FAIL [0.0/1.0] {label}: LLM did not return a score for this question.")

    # Override q1_passed from the explicit field if present
    if result.get("has_article_rationale") is True:
        q1_passed = True

    return {
        "score": round(score, 2),
        "max_score": max_score,
        "details": details,
        "q1_passed": q1_passed,
    }


# ---------------------------------------------------------------------------
# LLM-based JSON annotation grading (2 pts of 3 — 1 pt stays as parse check)
# ---------------------------------------------------------------------------

JSON_ANNOTATION_SYSTEM_PROMPT = """\
You are an expert teaching assistant grading a student's JSON annotation file \
for a university AI course assignment. The assignment asked students to create \
annotations that link elements from their PDDL domain (actions, predicates, types) \
to phrases in the wikiHow article they chose.

You will receive:
1. The student's JSON annotation content
2. The student's PDDL domain file (if available)

Evaluate the quality of the annotations on these TWO criteria:

CRITERION A — PDDL Element Coverage (score 0.0, 0.5, or 1.0):
  1.0 — The JSON meaningfully references multiple PDDL elements (actions, \
        predicates, types) from the domain.
  0.5 — The JSON references some PDDL elements but coverage is sparse or \
        superficial.
  0.0 — No meaningful references to PDDL elements.

CRITERION B — Natural Language / wikiHow Mapping Quality (score 0.0, 0.5, or 1.0):
  1.0 — The JSON contains clear natural language descriptions or wikiHow \
        article excerpts that are meaningfully mapped to PDDL elements.
  0.5 — Some NL text present but mappings are incomplete, vague, or minimal.
  0.0 — No natural language text or wikiHow references.

Respond ONLY with valid JSON in this exact format (no markdown fences):
{
  "pddl_coverage": {"score": 1.0, "justification": "..."},
  "nl_mapping": {"score": 0.5, "justification": "..."}
}"""


def grade_json_llm(
    json_files: list,
    domain_result: dict,
    client,
    model: str,
    verbose: bool = False,
) -> dict:
    """Grade JSON annotations using Claude for semantic evaluation.

    The 1 pt for 'valid JSON' is still checked programmatically.
    The remaining 2 pts (PDDL coverage + NL mapping) are evaluated by the LLM.
    """
    score = 0.0
    max_score = 3.0
    details = []

    if not json_files:
        details.append("FAIL: No JSON annotation file found.")
        return {"score": score, "max_score": max_score, "details": details}

    json_path = json_files[0]
    if len(json_files) > 1:
        details.append(
            f"NOTE: Found {len(json_files)} JSON files; grading first: "
            f"{os.path.basename(json_path)}"
        )

    # --- Valid JSON (1 pt — always programmatic) ---
    try:
        with open(json_path, "r", errors="replace") as f:
            data = json.load(f)
        score += 1.0
        details.append("PASS [1.0/1.0]: Valid JSON file.")
    except json.JSONDecodeError as e:
        details.append(f"FAIL [0.0/1.0]: Invalid JSON — {e}")
        return {"score": score, "max_score": max_score, "details": details}
    except Exception as e:
        details.append(f"FAIL [0.0/1.0]: Could not read file — {e}")
        return {"score": score, "max_score": max_score, "details": details}

    # --- LLM evaluation of remaining 2 pts ---
    json_str = json.dumps(data, indent=2)
    # Truncate if very large
    if len(json_str) > 8000:
        json_str = json_str[:8000] + "\n... [truncated]"

    # Get domain source if available
    domain_source = ""
    domain_path = domain_result.get("domain_path")
    if domain_path:
        try:
            with open(domain_path, "r", errors="replace") as f:
                domain_source = f.read()
            if len(domain_source) > 6000:
                domain_source = domain_source[:6000] + "\n... [truncated]"
        except Exception:
            domain_source = "(domain file could not be read)"

    user_prompt = (
        f"## Student's JSON Annotations:\n```json\n{json_str}\n```\n\n"
        f"## Student's PDDL Domain:\n```pddl\n{domain_source or '(not available)'}\n```"
    )

    try:
        raw = _call_claude(client, model, JSON_ANNOTATION_SYSTEM_PROMPT, user_prompt, max_tokens=512)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        details.append(
            f"WARNING: LLM evaluation failed ({e}). Falling back to keyword heuristics."
        )
        # Fall back to keyword-based grading for remaining 2 pts
        return grade_json(json_files, domain_result, verbose=verbose)

    details.append("[Graded by LLM]")

    # PDDL coverage (1 pt)
    pddl_cov = result.get("pddl_coverage", {})
    pddl_score = float(pddl_cov.get("score", 0))
    pddl_score = max(0.0, min(1.0, pddl_score))
    score += pddl_score
    pddl_just = pddl_cov.get("justification", "No justification.")
    tag = "PASS" if pddl_score == 1.0 else ("PARTIAL" if pddl_score > 0 else "FAIL")
    details.append(f"{tag} [{pddl_score}/1.0] PDDL element coverage: {pddl_just}")

    # NL mapping quality (1 pt)
    nl_map = result.get("nl_mapping", {})
    nl_score = float(nl_map.get("score", 0))
    nl_score = max(0.0, min(1.0, nl_score))
    score += nl_score
    nl_just = nl_map.get("justification", "No justification.")
    tag = "PASS" if nl_score == 1.0 else ("PARTIAL" if nl_score > 0 else "FAIL")
    details.append(f"{tag} [{nl_score}/1.0] NL/wikiHow mapping quality: {nl_just}")

    return {"score": round(score, 2), "max_score": max_score, "details": details}


# ---------------------------------------------------------------------------
# Penalty check
# ---------------------------------------------------------------------------

def check_penalty(writeup_result: dict) -> dict:
    """Apply -0.5 penalty if Q1 (article rationale) was not detected."""
    q1_passed = writeup_result.get("q1_passed", False)
    if q1_passed:
        return {
            "score": 0.0,
            "max_score": 0.0,
            "details": ["No penalty applied (article rationale detected)."],
        }
    else:
        return {
            "score": -0.5,
            "max_score": 0.0,
            "details": [
                "PENALTY [-0.5]: No article rationale detected in writeup. "
                "If the student did explain their article choice, "
                "override this during manual review."
            ],
        }


# ---------------------------------------------------------------------------
# Grade a single student submission
# ---------------------------------------------------------------------------

def grade_submission(
    submission_dir: str,
    timeout: int = 30,
    verbose: bool = False,
    llm_client=None,
    llm_model: str = DEFAULT_MODEL,
) -> dict:
    """Grade one student submission. Returns Gradescope-format dict.

    If llm_client is provided, uses Claude for writeup and JSON grading.
    Otherwise falls back to keyword heuristics.
    """
    files = discover_files(submission_dir)

    # --- Domain ---
    domain_result = grade_domain(files["domain_files"], verbose=verbose)

    # --- Problems ---
    problem_result = grade_problems(
        files["problem_files"], domain_result, timeout=timeout, verbose=verbose
    )

    # --- JSON ---
    if llm_client:
        json_result = grade_json_llm(
            files["json_files"], domain_result, llm_client, llm_model, verbose=verbose
        )
    else:
        json_result = grade_json(files["json_files"], domain_result, verbose=verbose)

    # --- Writeup ---
    if llm_client:
        writeup_result = grade_writeup_llm(
            files["pdf_files"], llm_client, llm_model, verbose=verbose
        )
    else:
        writeup_result = grade_writeup(files["pdf_files"], verbose=verbose)

    # --- Penalty ---
    penalty_result = check_penalty(writeup_result)

    # --- Aggregate ---
    total = (
        domain_result["score"]
        + problem_result["score"]
        + json_result["score"]
        + writeup_result["score"]
        + penalty_result["score"]
    )
    total = max(0.0, round(total, 2))

    tests = [
        {
            "score": domain_result["score"],
            "max_score": domain_result["max_score"],
            "name": "Domain Definition",
            "output": "\n".join(domain_result["details"]),
        },
        {
            "score": problem_result["score"],
            "max_score": problem_result["max_score"],
            "name": "Problem Definition",
            "output": "\n".join(problem_result["details"]),
        },
        {
            "score": json_result["score"],
            "max_score": json_result["max_score"],
            "name": "JSON Annotations",
            "output": "\n".join(json_result["details"]),
        },
        {
            "score": writeup_result["score"],
            "max_score": writeup_result["max_score"],
            "name": "Writeup",
            "output": "\n".join(writeup_result["details"]),
            "visibility": "after_published",
        },
        {
            "score": penalty_result["score"],
            "max_score": penalty_result["max_score"],
            "name": "Penalty",
            "output": "\n".join(penalty_result["details"]),
        },
    ]

    return {"score": total, "tests": tests}


# ---------------------------------------------------------------------------
# Main: iterate over all submissions
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autograder for HW1 Part 4: WikiHow to PDDL"
    )
    parser.add_argument(
        "submissions_dir",
        help="Path to master directory containing student submission folders",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Planner timeout per problem in seconds (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include extra detail in grading output",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use Claude LLM for writeup and JSON annotation grading "
             "(requires ANTHROPIC_API_KEY env var or --api-key)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use for LLM grading (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)",
    )
    args = parser.parse_args()

    submissions_dir = os.path.abspath(args.submissions_dir)
    if not os.path.isdir(submissions_dir):
        print(f"Error: {submissions_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # --- Initialize LLM client if requested ---
    llm_client = None
    if args.llm:
        if not HAS_ANTHROPIC:
            print(
                "Error: --llm requires the anthropic package. "
                "Install it with: pip install anthropic",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: --llm requires an API key. Set ANTHROPIC_API_KEY env var "
                "or pass --api-key.",
                file=sys.stderr,
            )
            sys.exit(1)
        llm_client = anthropic.Anthropic(api_key=api_key)
        print(f"LLM grading enabled (model: {args.model})")

    # Collect student folders (skip hidden dirs)
    student_dirs = sorted(
        d
        for d in os.listdir(submissions_dir)
        if os.path.isdir(os.path.join(submissions_dir, d)) and not d.startswith(".")
    )

    if not student_dirs:
        print(f"No student submission folders found in {submissions_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(student_dirs)} submission(s) in {submissions_dir}\n")
    print("=" * 70)

    csv_rows = []

    for student in student_dirs:
        student_path = os.path.join(submissions_dir, student)
        print(f"\nGrading: {student}")
        print("-" * 50)

        try:
            result = grade_submission(
                student_path,
                timeout=args.timeout,
                verbose=args.verbose,
                llm_client=llm_client,
                llm_model=args.model,
            )
        except Exception:
            tb = traceback.format_exc()
            result = {
                "score": 0.0,
                "tests": [
                    {
                        "score": 0.0,
                        "max_score": 20.0,
                        "name": "Autograder Error",
                        "output": f"Unexpected error grading this submission:\n{tb}",
                    }
                ],
            }

        # Print summary
        for test in result["tests"]:
            status = (
                "FULL"
                if test["score"] == test["max_score"]
                else ("ZERO" if test["score"] <= 0 else "PARTIAL")
            )
            print(
                f"  {test['name']}: {test['score']}/{test['max_score']} [{status}]"
            )
        print(f"  TOTAL: {result['score']}/20.0")

        # Write per-student Gradescope JSON
        results_path = os.path.join(student_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)

        # CSV row
        row = {"student": student, "total": result["score"]}
        for test in result["tests"]:
            row[test["name"]] = test["score"]
        csv_rows.append(row)

    # Write summary CSV
    csv_path = os.path.join(submissions_dir, "grades_summary.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n{'=' * 70}")
        print(f"Summary CSV written to: {csv_path}")

    print(f"Per-student results.json written to each submission folder.")
    print("Done.")


if __name__ == "__main__":
    main()
