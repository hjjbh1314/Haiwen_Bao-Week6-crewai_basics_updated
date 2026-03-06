# Mini-Assignment 2 — News Article Analysis Tools + CrewAI Workflow

## What the Tools Do

Three custom tools for news/business text analysis, each implemented as a pure-Python function (no external NLP libraries):

| Tool | Purpose | Key Output |
|---|---|---|
| **Keyword Extractor** | TF-based keyword ranking with stopword filtering | Top-N keywords, frequencies, relevance scores |
| **Sentiment Scorer** | Weighted lexicon-based sentiment analysis | Label (positive/negative/neutral), score [-1,1], matched words |
| **Entity & Stats Extractor** | Regex entity recognition + readability metrics | Companies, monetary values, percentages, dates, Flesch score |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set the DeepSeek API key (pick one method)
export DEEPSEEK_API_KEY=your_key_here        # shell variable
# — or —
cp .env.example .env                          # then edit .env

# 3. Run the demo (uses sample data by default)
python demo.py

# 3a. Or use the original dataset (26 NYT articles)
python demo.py --original
```

### Data Sources

The repo includes two datasets in `data/`. Pass `--original` to switch:

| Flag | File | Description |
|---|---|---|
| *(default)* | `data/articles.json` | 8 curated tech/business articles — diverse topics, varied sentiment |
| `--original` | `data/cleaned_output.json` | 26 real NYT articles — original raw dataset for broader testing |

### What the Demo Shows

`demo.py` runs **two parts** in sequence to demonstrate both success and failure:

1. **Error Handling Demo** (no API calls needed) — 7 test cases with invalid inputs (empty string, wrong type, bad parameters). Each tool returns a structured `{"status": "error", ...}` dict instead of crashing.
2. **Full Crew Workflow** (requires DeepSeek API key) — 3 agents collaborate in a hybrid parallel+sequential pipeline, producing a 5-section intelligence report. This demonstrates the successful end-to-end execution.

## Realistic Use Case

**Scenario:** A financial analyst at an investment firm monitors 50+ news sources daily. Instead of reading every article, the analyst runs the pipeline on the morning's article batch:

```
$ python demo.py
```

**What happens:**

1. **Keyword Extraction** instantly surfaces the day's dominant topics (e.g. "nvidia", "tariff", "cloud outage") — the analyst can skip articles that fall outside their coverage.
2. **Sentiment Scoring** flags articles with strong negative scores (e.g. -0.63 for the AWS outage piece), prioritising them for risk assessment.
3. **Entity & Stats Extraction** pulls out all monetary figures ($4tn, $38bn), percentages (2.4%, 69%), and company names — no manual scanning needed.
4. The **Intelligence Reporter** agent synthesises everything into a structured 5-section briefing with actionable recommendations.

**Sample output excerpt (from actual demo run):**

> **Sentiment Overview**
> - Article 1 (Nvidia): **Positive** (0.62) — celebratory tone, revenue growth highlighted
> - Article 3 (AWS Outage): **Negative** (-0.63) — critical tone, systemic risk concerns
>
> **Recommendation:** Stress-test critical cloud dependencies; explore multi-cloud strategies.

**Error handling in action:** If an article's content field is empty or corrupted, the tool returns `{"status": "error", "error": "text must be a non-empty string"}` — the agent receives a clear message instead of crashing, and continues processing the remaining articles. The demo script includes 7 such error test cases that run before the main workflow.

## Design Decisions

- **Pure Python, no heavy NLP deps** — keeps install light and the logic transparent; suitable for a business-analysis audience that values reproducibility over model sophistication.
- **Three separate tools** — each has a single clear responsibility (SRP), making them independently testable and reusable.
- **Hybrid workflow** — Phase 1 runs keyword extraction and sentiment analysis in parallel (`ThreadPoolExecutor`); Phase 2 feeds both outputs to the reporter sequentially. This demonstrates both parallel and sequential collaboration patterns from Week 6 lecture.
- **Structured error returns** — every tool returns `{"status": "error", "error": "..."}` instead of raising exceptions, so CrewAI agents receive informative messages rather than stack traces.
- **DeepSeek API via environment variable** — the key is never hardcoded; loaded through `python-dotenv`.

## Limitations

- Keyword extraction uses raw term frequency, not TF-IDF across a corpus — relevance is per-article only.
- Sentiment lexicon is hand-curated (~120 words); domain-specific or sarcastic language may be misclassified.
- Entity regex targets common tech/finance names; lesser-known companies will not be detected.
- Flesch readability assumes English prose; heavily numeric or jargon-laden text may yield unusual scores.
