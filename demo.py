#!/usr/bin/env python3
"""
demo.py — CrewAI multi-agent news analysis workflow.

Workflow (Hybrid execution):
    Phase 1 (Parallel):  Agent 1 (Keyword Extraction) and Agent 2 (Sentiment
                          Analysis) analyse articles concurrently.
    Phase 2 (Sequential): Agent 3 (Entity & Stats Reporter) receives the
                          combined output and produces the final report.

Usage:
    export DEEPSEEK_API_KEY=your_key_here
    python demo.py
"""

from __future__ import annotations

import json
import os
import sys
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Crew, Task, Process, LLM

from tool import (
    keyword_extractor_tool,
    sentiment_scorer_tool,
    entity_stats_tool,
    extract_keywords,
    score_sentiment,
    extract_entities_and_stats,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("ERROR: DEEPSEEK_API_KEY environment variable is not set.")
    print("Please run:  export DEEPSEEK_API_KEY=your_key_here")
    print("Or create a .env file with DEEPSEEK_API_KEY=your_key_here")
    sys.exit(1)

llm = LLM(
    model="openai/deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.3,
)

DEFAULT_DATA = Path(__file__).parent / "data" / "articles.json"
ALT_DATA     = Path(__file__).parent / "data" / "cleaned_output.json"
DEMO_ARTICLE_COUNT = 3


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def resolve_data_path() -> Path:
    """Pick data file: --original flag uses cleaned_output.json, default uses articles.json."""
    if "--original" in sys.argv:
        return ALT_DATA
    return DEFAULT_DATA


def load_articles(path: Path, limit: int = DEMO_ARTICLE_COUNT) -> list[dict]:
    """Load articles from a JSON file.  Returns a list of article dicts."""
    if not path.exists():
        print(f"ERROR: Data file not found at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    return articles[:limit]


# ------------------------------------------------------------------
# Agent definitions
# ------------------------------------------------------------------

data_analyst = Agent(
    role="News Keyword Analyst",
    goal="Extract keywords from each article by calling keyword_extractor_tool.",
    backstory=(
        "You are a keyword specialist. When given article text, you ALWAYS "
        "call keyword_extractor_tool to get data-driven results. You never "
        "guess — you always use the tool first, then summarise its output."
    ),
    tools=[keyword_extractor_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)

sentiment_analyst = Agent(
    role="News Sentiment Analyst",
    goal="Score the sentiment of each article by calling sentiment_scorer_tool.",
    backstory=(
        "You are a sentiment expert. When given article text, you ALWAYS "
        "call sentiment_scorer_tool to get quantitative results. You never "
        "guess — you always use the tool first, then summarise its output."
    ),
    tools=[sentiment_scorer_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)

intelligence_reporter = Agent(
    role="Intelligence Report Compiler",
    goal=(
        "Extract entities from each article by calling entity_stats_tool, "
        "then combine ALL analysis into a final intelligence report."
    ),
    backstory=(
        "You are a senior analyst who compiles intelligence reports. You "
        "ALWAYS call entity_stats_tool on each article, then combine the "
        "entity data with upstream keyword and sentiment analysis."
    ),
    tools=[entity_stats_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=20,
)


# ------------------------------------------------------------------
# Phase-1 helper
# ------------------------------------------------------------------

def _run_single_crew(agent: Agent, task_desc: str, expected: str) -> str:
    """Build a Crew for *agent*, kick it off, return result as str."""
    task = Task(description=task_desc, expected_output=expected, agent=agent)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


# ------------------------------------------------------------------
# Workflow execution
# ------------------------------------------------------------------

def run_workflow(articles: list[dict]) -> str:
    """Execute the hybrid multi-agent workflow and return the final report."""

    article_texts = []
    for i, a in enumerate(articles, 1):
        article_texts.append(
            f"ARTICLE {i} — \"{a['title']}\"\n{a['content']}"
        )
    articles_block = "\n\n".join(article_texts)

    # ── Phase 1: Parallel ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1  —  Parallel: Keyword Extraction + Sentiment Analysis")
    print("=" * 70)

    kw_desc = (
        "Below are news articles. For EACH article, you MUST call "
        "keyword_extractor_tool with the full article content as the "
        "'text' argument. After getting results from the tool, list "
        "the top keywords for each article.\n\n" + articles_block
    )
    sent_desc = (
        "Below are news articles. For EACH article, you MUST call "
        "sentiment_scorer_tool with the full article content as the "
        "'text' argument. After getting results from the tool, list "
        "the sentiment label, score, and key words for each article.\n\n"
        + articles_block
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_kw = pool.submit(
            _run_single_crew, data_analyst, kw_desc,
            "For each article: title, top keywords (word + frequency), total words."
        )
        future_sent = pool.submit(
            _run_single_crew, sentiment_analyst, sent_desc,
            "For each article: title, sentiment label, score, positive/negative words."
        )
        result_kw = future_kw.result()
        result_sent = future_sent.result()

    print("\n" + "-" * 40)
    print("Phase 1 — Keyword results:")
    print("-" * 40)
    print(result_kw[:1000])
    print("\n" + "-" * 40)
    print("Phase 1 — Sentiment results:")
    print("-" * 40)
    print(result_sent[:1000])

    # ── Phase 2: Sequential ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2  —  Sequential: Entity Extraction + Final Report")
    print("=" * 70)

    report_desc = (
        "Below are news articles and upstream results.\n\n"
        "== ARTICLES ==\n" + articles_block + "\n\n"
        "== KEYWORD ANALYSIS ==\n" + result_kw + "\n\n"
        "== SENTIMENT ANALYSIS ==\n" + result_sent + "\n\n"
        "YOUR TASK:\n"
        "Step 1: Call entity_stats_tool on each article's full content.\n"
        "Step 2: Write a final intelligence report with these sections:\n"
        "  1. Executive Summary\n"
        "  2. Key Themes & Keywords\n"
        "  3. Sentiment Overview\n"
        "  4. Major Entities & Figures\n"
        "  5. Recommendations\n"
    )

    report_task = Task(
        description=report_desc,
        expected_output=(
            "A structured intelligence report with 5 sections: "
            "Executive Summary, Key Themes, Sentiment Overview, "
            "Major Entities, and Recommendations."
        ),
        agent=intelligence_reporter,
    )
    crew_report = Crew(
        agents=[intelligence_reporter],
        tasks=[report_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew_report.kickoff())


# ------------------------------------------------------------------
# Error handling demonstration
# ------------------------------------------------------------------

def demo_error_handling() -> None:
    """Demonstrate structured error returns from each tool."""
    print("\n" + "=" * 70)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 70)

    test_cases = [
        ("Keyword Extractor — empty string",     extract_keywords, {"text": ""}),
        ("Keyword Extractor — non-string input",  extract_keywords, {"text": 42}),
        ("Keyword Extractor — invalid top_n",     extract_keywords, {"text": "hello world test", "top_n": -3}),
        ("Sentiment Scorer  — None input",        score_sentiment,  {"text": None}),
        ("Sentiment Scorer  — whitespace only",   score_sentiment,  {"text": "   \t\n  "}),
        ("Entity Extractor  — list input",        extract_entities_and_stats, {"text": ["not", "a", "string"]}),
        ("Entity Extractor  — empty string",      extract_entities_and_stats, {"text": ""}),
    ]

    for label, fn, kwargs in test_cases:
        result = fn(**kwargs)
        print(f"\n  [{label}]")
        print(f"    Input:  {kwargs}")
        print(f"    Output: {json.dumps(result, ensure_ascii=False)}")

    print("\n  All error cases returned structured error dicts — no exceptions raised.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  CrewAI News Analysis Workflow  —  3 Tools × 3 Agents")
    print("=" * 70)

    data_path = resolve_data_path()
    articles = load_articles(data_path)
    print(f"\nLoaded {len(articles)} articles from {data_path.name}\n")
    for a in articles:
        print(f"  • {a['title']}")

    # --- Part 1: Error handling demo (no API calls) ---
    demo_error_handling()

    # --- Part 2: Full workflow (requires DeepSeek API) ---
    print("\n" + "=" * 70)
    print("  STARTING FULL CREW WORKFLOW")
    print("=" * 70)

    report = run_workflow(articles)

    print("\n" + "=" * 70)
    print("  FINAL INTELLIGENCE REPORT")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
