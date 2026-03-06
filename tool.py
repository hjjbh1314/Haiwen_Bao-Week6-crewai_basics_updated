"""
tool.py — Three custom tools for news article analysis.

Tools:
    1. Keyword Extractor: TF-based keyword extraction with stopword filtering.
    2. Sentiment Scorer: Lexicon-based heuristic sentiment analysis.
    3. Entity & Statistics Extractor: Regex entity recognition + readability metrics.

Each tool is exposed via:
    - A pure function with full type hints, docstring, validation, and error handling.
    - A lightweight Tool wrapper class (as suggested by the assignment spec).
    - A CrewAI @tool decorated version for agent integration.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from crewai.tools import tool as crewai_tool

# ---------------------------------------------------------------------------
# Generic Tool wrapper (assignment-suggested structure)
# ---------------------------------------------------------------------------

class Tool:
    """Lightweight wrapper that pairs a callable with metadata."""

    def __init__(self, name: str, description: str, fn: Callable[..., Dict[str, Any]]):
        self.name = name
        self.description = description
        self.fn = fn

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the wrapped function, catching unexpected exceptions."""
        try:
            return self.fn(**kwargs)
        except Exception as exc:
            return {"status": "error", "error": f"Unexpected error: {exc}"}

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


# ===================================================================
# Shared helpers
# ===================================================================

STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "just", "let", "me", "might", "more", "most", "much",
    "must", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "per", "s", "same", "she", "should", "shouldn't", "so", "some",
    "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won't", "would",
    "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "also", "been", "being", "can", "come", "could", "even", "first",
    "get", "got", "may", "new", "one", "said", "say", "says", "since",
    "still", "take", "two", "well", "would", "year", "years",
}


def _validate_text(text: Any) -> Optional[Dict[str, Any]]:
    """Return an error dict if *text* is not a non-empty string, else None."""
    if not isinstance(text, str):
        return {
            "status": "error",
            "error": f"Expected a string, got {type(text).__name__}.",
        }
    if not text.strip():
        return {"status": "error", "error": "Text is empty or whitespace-only."}
    return None


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenisation keeping only alphabetic tokens >= 3 chars."""
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) >= 3]


def _count_syllables(word: str) -> int:
    """Rough syllable count for Flesch-Kincaid formula."""
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


# ===================================================================
# Tool 1 — Keyword Extractor
# ===================================================================

def extract_keywords(text: str, top_n: int = 10) -> Dict[str, Any]:
    """Extract the top-N keywords from *text* using term-frequency analysis.

    Parameters
    ----------
    text : str
        The article body to analyse.
    top_n : int, optional
        Number of keywords to return (default 10).  Must be a positive integer.

    Returns
    -------
    dict
        On success::

            {
                "status": "success",
                "keywords": [{"word": str, "frequency": int, "relevance": float}, ...],
                "total_words": int,
                "unique_words": int,
            }

        On failure::

            {"status": "error", "error": str}
    """
    err = _validate_text(text)
    if err:
        return err

    if not isinstance(top_n, int) or top_n <= 0:
        return {"status": "error", "error": f"top_n must be a positive integer, got {top_n!r}."}

    tokens = _tokenize(text)
    if not tokens:
        return {"status": "error", "error": "No valid tokens found after tokenisation."}

    filtered = [t for t in tokens if t not in STOPWORDS]
    if not filtered:
        return {"status": "error", "error": "All tokens were stopwords; no keywords extracted."}

    freq = Counter(filtered)
    max_freq = freq.most_common(1)[0][1]
    ranked = [
        {
            "word": word,
            "frequency": count,
            "relevance": round(count / max_freq, 4),
        }
        for word, count in freq.most_common(top_n)
    ]

    return {
        "status": "success",
        "keywords": ranked,
        "total_words": len(tokens),
        "unique_words": len(set(tokens)),
    }


# ===================================================================
# Tool 2 — Sentiment Scorer
# ===================================================================

POSITIVE_LEXICON: Dict[str, float] = {
    "good": 1.0, "great": 1.5, "excellent": 2.0, "amazing": 2.0,
    "wonderful": 2.0, "fantastic": 2.0, "outstanding": 2.0, "superb": 2.0,
    "brilliant": 1.8, "remarkable": 1.5, "impressive": 1.5, "incredible": 1.8,
    "positive": 1.0, "benefit": 1.2, "benefits": 1.2, "success": 1.5,
    "successful": 1.5, "growth": 1.2, "grow": 1.0, "growing": 1.0,
    "gain": 1.0, "gains": 1.0, "profit": 1.2, "profits": 1.2,
    "profitable": 1.3, "innovation": 1.5, "innovative": 1.5,
    "opportunity": 1.2, "opportunities": 1.2, "boost": 1.2, "boosted": 1.2,
    "surge": 1.3, "surged": 1.3, "soar": 1.3, "soared": 1.3,
    "record": 1.0, "high": 0.8, "historic": 1.2, "landmark": 1.2,
    "breakthrough": 1.5, "advance": 1.2, "advanced": 1.2, "leadership": 1.0,
    "leader": 1.0, "leading": 1.0, "best": 1.2, "top": 0.8,
    "strengthen": 1.2, "strengthens": 1.2, "improve": 1.0, "improved": 1.0,
    "rising": 1.0, "rise": 1.0, "risen": 1.0, "climbed": 1.0,
    "upgrade": 1.0, "upgraded": 1.0, "prosper": 1.3, "prosperity": 1.3,
    "invest": 1.0, "investment": 1.0, "invested": 1.0, "investing": 1.0,
    "revenue": 0.8, "demand": 0.8, "expand": 1.0, "expansion": 1.0,
    "powerful": 1.0, "superpower": 1.5, "accelerate": 1.2,
}

NEGATIVE_LEXICON: Dict[str, float] = {
    "bad": -1.0, "terrible": -2.0, "horrible": -2.0, "awful": -2.0,
    "worst": -2.0, "poor": -1.5, "negative": -1.0, "failure": -1.5,
    "fail": -1.5, "failed": -1.5, "crisis": -1.5, "crash": -1.8,
    "crashed": -1.8, "collapse": -1.8, "collapsed": -1.8,
    "loss": -1.2, "losses": -1.2, "lost": -1.0, "lose": -1.0,
    "decline": -1.2, "declined": -1.2, "declining": -1.2,
    "drop": -1.0, "dropped": -1.0, "fall": -1.0, "fell": -1.0,
    "fallen": -1.0, "plunge": -1.5, "plunged": -1.5,
    "risk": -0.8, "risks": -0.8, "risky": -1.0, "threat": -1.2,
    "threats": -1.2, "threaten": -1.2, "threatened": -1.2,
    "concern": -0.8, "concerns": -0.8, "worried": -1.0, "worry": -1.0,
    "fear": -1.2, "fears": -1.2, "warning": -1.0, "warnings": -1.0,
    "warn": -1.0, "warned": -1.0, "danger": -1.2, "dangerous": -1.2,
    "problem": -1.0, "problems": -1.0, "issue": -0.6, "issues": -0.6,
    "controversy": -1.0, "controversial": -1.0, "scandal": -1.5,
    "violated": -1.5, "violation": -1.5, "ban": -1.0, "banned": -1.0,
    "restrict": -1.0, "restricted": -1.0, "restriction": -1.0,
    "restrictions": -1.0, "sanctions": -1.2, "penalty": -1.2,
    "fine": -0.8, "fines": -0.8, "unprofitable": -1.5, "debt": -1.0,
    "bubble": -1.2, "burst": -1.3, "outage": -1.3, "disruption": -1.0,
    "struggle": -1.0, "struggled": -1.0, "struggling": -1.0,
    "disappointed": -1.2, "disappointing": -1.2, "scrutiny": -1.0,
    "tension": -1.0, "tensions": -1.0, "conflict": -1.2, "war": -1.0,
}


def score_sentiment(text: str) -> Dict[str, Any]:
    """Score the sentiment of *text* using a weighted lexicon approach.

    The scorer scans every token against built-in positive and negative
    lexicons.  It returns a normalised score in [-1, 1], a label
    (``positive`` / ``negative`` / ``neutral``), and the matched words.

    Parameters
    ----------
    text : str
        The article body to analyse.

    Returns
    -------
    dict
        On success::

            {
                "status": "success",
                "sentiment": "positive" | "negative" | "neutral",
                "score": float,        # normalised in [-1, 1]
                "confidence": float,   # 0-1 based on evidence strength
                "positive_words": [{"word": str, "weight": float}, ...],
                "negative_words": [{"word": str, "weight": float}, ...],
            }

        On failure::

            {"status": "error", "error": str}
    """
    err = _validate_text(text)
    if err:
        return err

    tokens = _tokenize(text)
    if not tokens:
        return {"status": "error", "error": "No valid tokens found after tokenisation."}

    pos_matches: List[Dict[str, Any]] = []
    neg_matches: List[Dict[str, Any]] = []
    pos_score = 0.0
    neg_score = 0.0

    for token in tokens:
        if token in POSITIVE_LEXICON:
            w = POSITIVE_LEXICON[token]
            pos_score += w
            pos_matches.append({"word": token, "weight": w})
        if token in NEGATIVE_LEXICON:
            w = NEGATIVE_LEXICON[token]
            neg_score += abs(w)
            neg_matches.append({"word": token, "weight": w})

    total_weight = pos_score + neg_score
    if total_weight == 0:
        normalised = 0.0
        confidence = 0.0
    else:
        normalised = round((pos_score - neg_score) / total_weight, 4)
        evidence_ratio = total_weight / len(tokens)
        confidence = round(min(evidence_ratio * 5, 1.0), 4)

    if normalised > 0.05:
        label = "positive"
    elif normalised < -0.05:
        label = "negative"
    else:
        label = "neutral"

    seen_pos = {}
    for m in pos_matches:
        seen_pos[m["word"]] = m["weight"]
    seen_neg = {}
    for m in neg_matches:
        seen_neg[m["word"]] = m["weight"]

    return {
        "status": "success",
        "sentiment": label,
        "score": normalised,
        "confidence": confidence,
        "positive_words": [{"word": w, "weight": v} for w, v in seen_pos.items()],
        "negative_words": [{"word": w, "weight": v} for w, v in seen_neg.items()],
    }


# ===================================================================
# Tool 3 — Entity & Statistics Extractor
# ===================================================================

_COMPANY_PATTERNS = re.compile(
    r"\b(?:"
    r"Nvidia|NVIDIA|Intel|AMD|TSMC|Microsoft|Google|Apple|Amazon|Meta|"
    r"OpenAI|Samsung|Huawei|Alibaba|Oracle|SoftBank|Broadcom|"
    r"Qualcomm|Tesla|Uber|Nokia|Stellantis|Arm|SK Hynix|"
    r"DeepSeek|Nscale|AWS|BBC|IBM|Sony|Netflix|Alphabet|YouTube|"
    r"LG|Hyundai|Kakao|Naver|JP Morgan|Bank of America"
    r")\b"
)
_MONETARY_PATTERN = re.compile(
    r"(?:\$[\d,.]+\s*(?:bn|billion|tn|trillion|m|million|k|thousand)?)"
    r"|(?:£[\d,.]+\s*(?:bn|billion|tn|trillion|m|million|k|thousand)?)"
    r"|(?:€[\d,.]+\s*(?:bn|billion|tn|trillion|m|million|k|thousand)?)",
    re.IGNORECASE,
)
_PERCENTAGE_PATTERN = re.compile(r"\d+(?:\.\d+)?%")
_DATE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"(?:Q[1-4]\s+\d{4})"
    r")\b"
)


def extract_entities_and_stats(text: str) -> Dict[str, Any]:
    """Extract named entities and compute readability statistics for *text*.

    Entities detected (via regex):
        - Companies / organisations
        - Monetary values ($, £, €)
        - Percentages
        - Dates

    Statistics computed:
        - word_count, sentence_count, avg_sentence_length
        - readability_score (Flesch Reading Ease)
        - reading_time_minutes (assuming 200 wpm)

    Parameters
    ----------
    text : str
        The article body to analyse.

    Returns
    -------
    dict
        On success::

            {
                "status": "success",
                "entities": {
                    "companies": [...],
                    "monetary": [...],
                    "percentages": [...],
                    "dates": [...],
                },
                "statistics": {
                    "word_count": int,
                    "sentence_count": int,
                    "avg_sentence_length": float,
                    "readability_score": float,
                    "reading_time_minutes": float,
                },
            }

        On failure::

            {"status": "error", "error": str}
    """
    err = _validate_text(text)
    if err:
        return err

    companies = sorted(set(_COMPANY_PATTERNS.findall(text)))
    monetary = _MONETARY_PATTERN.findall(text)
    percentages = _PERCENTAGE_PATTERN.findall(text)
    dates = _DATE_PATTERN.findall(text)

    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_length = round(word_count / sentence_count, 2)

    total_syllables = sum(_count_syllables(w) for w in re.findall(r"[a-zA-Z]+", text))
    if word_count > 0:
        flesch = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (total_syllables / word_count)
        )
        flesch = round(flesch, 2)
    else:
        flesch = 0.0

    reading_time = round(word_count / 200, 2)

    return {
        "status": "success",
        "entities": {
            "companies": companies,
            "monetary": monetary,
            "percentages": percentages,
            "dates": dates,
        },
        "statistics": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "readability_score": flesch,
            "reading_time_minutes": reading_time,
        },
    }


# ===================================================================
# Tool wrapper instances
# ===================================================================

keyword_extractor = Tool(
    name="keyword_extractor",
    description="Extract top keywords from article text using term-frequency analysis.",
    fn=extract_keywords,
)

sentiment_scorer = Tool(
    name="sentiment_scorer",
    description="Score article sentiment using a weighted positive/negative lexicon.",
    fn=score_sentiment,
)

entity_stats_extractor = Tool(
    name="entity_stats_extractor",
    description="Extract named entities (companies, monetary, %, dates) and compute text statistics.",
    fn=extract_entities_and_stats,
)

# ===================================================================
# CrewAI @tool versions (for agent integration in demo.py)
# ===================================================================

@crewai_tool
def keyword_extractor_tool(text: str) -> str:
    """Extract top-10 keywords from the given article text.

    Returns a JSON string with ranked keywords, frequencies, and relevance
    scores.  Useful for identifying the core topics of a news article.
    """
    result = extract_keywords(text, top_n=10)
    return json.dumps(result, ensure_ascii=False, indent=2)


@crewai_tool
def sentiment_scorer_tool(text: str) -> str:
    """Analyse the sentiment of the given article text.

    Returns a JSON string with a sentiment label (positive / negative /
    neutral), a normalised score in [-1, 1], confidence, and the matched
    positive and negative words.
    """
    result = score_sentiment(text)
    return json.dumps(result, ensure_ascii=False, indent=2)


@crewai_tool
def entity_stats_tool(text: str) -> str:
    """Extract named entities and readability statistics from the given text.

    Returns a JSON string containing detected companies, monetary values,
    percentages, dates, plus word count, sentence count, average sentence
    length, Flesch readability score, and estimated reading time.
    """
    result = extract_entities_and_stats(text)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===================================================================
# Quick self-test when run directly
# ===================================================================

if __name__ == "__main__":
    sample = (
        "Nvidia has become the first company to reach a market value of $4tn. "
        "Shares rose by 2.4% on Wednesday. CEO Jensen Huang said this was a "
        "historic moment. The company reported revenue of $44.1bn in Q1 2025, "
        "marking a 69% jump from a year ago."
    )
    print("=== Keyword Extractor ===")
    print(json.dumps(keyword_extractor.execute(text=sample, top_n=5), indent=2))

    print("\n=== Sentiment Scorer ===")
    print(json.dumps(sentiment_scorer.execute(text=sample), indent=2))

    print("\n=== Entity & Stats Extractor ===")
    print(json.dumps(entity_stats_extractor.execute(text=sample), indent=2))

    print("\n=== Error handling demos ===")
    print(keyword_extractor.execute(text=""))
    print(keyword_extractor.execute(text=12345))
    print(sentiment_scorer.execute(text=None))
    print(entity_stats_extractor.execute(text="   "))
