"""Grounded answer generation for the HR Policy Assistant.

Flow: retrieved chunks -> Groq structured response -> evidence validation
-> citations generated from trusted ChromaDB metadata.
"""

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from retrieval import retrieve_chunks


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Override this in .env if you switch to another Groq model.
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEBUG = os.getenv("RAG_DEBUG", "0") == "1"
TOP_K = 3

REFUSAL_MESSAGE = (
    "I don't have enough information in the uploaded policies. "
    "Please contact HR."
)

logger = logging.getLogger(__name__)


# Phrases indicating that the policy is silent or delegates the topic
# elsewhere. These are used only as a safeguard when the model labels
# evidence as an explicit negative; they are not a topic-specific list.
UNDEFINED_RE = re.compile(
    r"\b(?:does not|do not|is not|are not|not)\s+"
    r"(?:define|defined|specify|specified|address|addressed)\b"
    r"|\bseparate\s+(?:[\w-]+\s+){0,3}(?:policy|circular|document)\b"
    r"|\boutside the scope of (?:this|the) policy\b"
    r"|\bconsult (?:the )?(?:separate )?[\w-]*(?:policy|circular|document)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an internal HR Policy Assistant.
Answer employee questions using ONLY the retrieved policy documents.

STEP 1 - CLASSIFY THE EVIDENCE

Choose exactly one quote_type:

- gives_value_or_rule: the text gives an amount, limit, eligibility
  rule, condition, or procedure.
- prohibits_or_excludes: the text explicitly says something is
  prohibited, not covered, not reimbursable, not permitted, or not eligible.
- says_undefined_or_elsewhere: the text says something is not defined,
  not specified, outside this policy, or covered in another document
  such as a circular or separate policy.
- none: there is no relevant text.

Choose exactly one support_type:

- supported: a policy value or rule directly answers the question.
- explicit_negative: the policy explicitly prohibits or excludes
  the item asked about.
- not_specified: the policy explicitly says the requested information
  is undefined or handled in another document.
- not_found: the retrieved context does not contain enough relevant
  information to answer the question.

CLASSIFICATION RULES

1. Silence is not a prohibition. Never answer "No" merely because
   the policy does not mention something.
2. A statement such as "not reimbursable" or "not covered" is an
   explicit negative only when it directly applies to the requested item.
3. A statement such as "does not define", "not specified", or
   "refer to a separate policy/circular" means not_specified, not
   explicit_negative.
4. If the question asks about information that is absent from the
   retrieved context, use not_found.
5. A related section alone is not enough. The evidence must support
   every essential part of the question.
6. Use no outside knowledge and make no assumptions about common
   company practices.

GENERAL POLICY REASONING

7. You may combine information from multiple retrieved chunks and
   perform simple arithmetic or comparisons only when every required
   value and rule is explicitly present in the retrieved context.
8. If any essential value, condition, or rule is missing, refuse.
9. If retrieved documents conflict and the context does not establish
   which policy takes precedence, do not choose arbitrarily; refuse
   rather than inventing a precedence rule.

TABLE INTERPRETATION

10. Markdown tables are structured policy data. Do not ignore data
    because it appears in a table.
11. Identify the requested row (for example employee band or health tier)
    and the requested column (for example allowance or coverage), then
    read the cell at their intersection.
12. Interpret ranges and inclusive categories exactly as written, such
    as "Band A-B" or "Band E and above".
13. For table-based answers, provide separate evidence items for the
    relevant header row and the complete matching data row.
14. If a table value is combined with another rule or condition, provide
    a separate evidence item for that rule or condition.

EVIDENCE EXTRACTION - STRICT

15. Every evidence item must be one contiguous piece of text copied
    verbatim from a single retrieved chunk: one sentence or one table row.
16. Never paraphrase, summarize, reconstruct, merge, shorten, reorder,
    or rewrite a quotation.
17. Do not combine words from adjacent sentences into one quotation.
18. Preserve the original wording, numbers, symbols, and Markdown content.
19. For each evidence item, use the exact chunk ID belonging to its source.
20. If you cannot provide exact supporting evidence, do not answer.

OUTPUT CLASSIFICATION

- supported: status="answered", support_type="supported".
- explicit_negative: status="answered", support_type="explicit_negative".
- not_specified: status="insufficient_information",
  support_type="not_specified", evidence=[].
- not_found: status="insufficient_information",
  support_type="not_found", evidence=[].

When refusing, use exactly this answer:
"I don't have enough information in the uploaded policies. Please contact HR."

Return only the JSON object matching the supplied schema.
"""


# Groq strict JSON Schema response format.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_policy_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "quote_type": {
                    "type": "string",
                    "enum": [
                        "gives_value_or_rule",
                        "prohibits_or_excludes",
                        "says_undefined_or_elsewhere",
                        "none",
                    ],
                },
                "status": {
                    "type": "string",
                    "enum": ["answered", "insufficient_information"],
                },
                "support_type": {
                    "type": "string",
                    "enum": [
                        "supported",
                        "explicit_negative",
                        "not_specified",
                        "not_found",
                    ],
                },
                "answer": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["chunk_id", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "quote_type",
                "status",
                "support_type",
                "answer",
                "evidence",
            ],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------
# Text normalization and evidence verification
# ---------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize harmless formatting differences for quote comparison."""
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible formatting characters.
    for char in ("\u200b", "\ufeff", "\u00ad"):
        text = text.replace(char, "")

    # Markdown emphasis/code delimiters do not change the words.
    text = text.replace("**", "").replace("__", "").replace("`", "")

    # Normalize whitespace and line breaks.
    text = re.sub(r"\s+", " ", text)

    # Ignore spaces around Markdown table separators.
    text = re.sub(r"\s*\|\s*", "|", text)

    return text.strip().casefold()


def quote_is_supported(quote: str, source_text: str) -> bool:
    """Accept only a continuous source quotation after safe normalization.

    No fuzzy token-overlap fallback is used: a quote must remain a
    contiguous substring of its original retrieved chunk.
    """
    if not isinstance(quote, str) or not quote.strip():
        return False

    if not isinstance(source_text, str) or not source_text:
        return False

    return normalize_text(quote) in normalize_text(source_text)


# ---------------------------------------------------------------------
# Groq client and response helpers
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    return Groq(api_key=api_key)


def refusal_response() -> dict[str, Any]:
    return {
        "status": "insufficient_information",
        "answer": REFUSAL_MESSAGE,
        "citations": [],
    }


def build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks with source metadata for the LLM."""
    parts = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") or {}
        parts.append(
            f"CHUNK ID: {chunk.get('id', '')}\n"
            f"DOCUMENT: {metadata.get('document_name', '')}\n"
            f"SECTION: {metadata.get('section', '')}\n"
            f"CONTENT:\n{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(parts)


def validate_llm_result(result: Any) -> bool:
    """Defensively validate the returned JSON structure and field types."""
    if not isinstance(result, dict):
        return False

    required_fields = {
        "quote_type",
        "status",
        "support_type",
        "answer",
        "evidence",
    }
    if set(result.keys()) != required_fields:
        return False

    if result["quote_type"] not in {
        "gives_value_or_rule",
        "prohibits_or_excludes",
        "says_undefined_or_elsewhere",
        "none",
    }:
        return False

    if result["status"] not in {"answered", "insufficient_information"}:
        return False

    if result["support_type"] not in {
        "supported",
        "explicit_negative",
        "not_specified",
        "not_found",
    }:
        return False

    if not isinstance(result["answer"], str):
        return False

    if not isinstance(result["evidence"], list):
        return False

    for item in result["evidence"]:
        if not isinstance(item, dict):
            return False
        if set(item.keys()) != {"chunk_id", "quote"}:
            return False
        if not isinstance(item["chunk_id"], str):
            return False
        if not isinstance(item["quote"], str):
            return False

    return True


# ---------------------------------------------------------------------
# Main generation and validation
# ---------------------------------------------------------------------


def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate an answer and validate every supporting evidence item."""
    if not retrieved_chunks:
        return refusal_response()

    user_prompt = (
        f"EMPLOYEE QUESTION:\n{question}\n\n"
        f"RETRIEVED POLICY CONTEXT:\n{build_context(retrieved_chunks)}\n\n"
        "Examine all retrieved chunks and apply the system rules. "
        "Return the JSON object matching the schema. "
        "Copy each evidence quote verbatim as one contiguous sentence "
        "or one complete table row. Do not merge separate source passages."
    )

    client = get_groq_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt + """

    Return a JSON object with exactly these fields:

    {
        "quote_type": "...",
        "status": "...",
        "support_type": "...",
        "answer": "...",
        "evidence": [
            {
                "chunk_id": "...",
                "quote": "..."
            }
        ]
    }

    Do not add additional fields.
    For refusals, evidence must be an empty array.
    """
        },
    ]

    try:
        # Primary approach: Strict JSON Schema
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            response_format=RESPONSE_SCHEMA,
        )

    except Exception as exc:

        # Retry only for Groq's structured-output validation error.
        # Do not retry rate limits or unrelated API errors.
        if (
            getattr(exc, "status_code", None) != 400
            or "json_validate_failed" not in str(exc)
        ):
            raise

        logger.warning(
            "Strict JSON generation failed. Retrying with JSON Object Mode."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

    raw_response = response.choices[0].message.content
    if not raw_response:
        raise ValueError("Groq returned an empty response.")

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("Groq returned invalid JSON.") from exc

    if DEBUG:
        logger.debug(
            "Groq response: %s",
            json.dumps(result, indent=2, ensure_ascii=False),
        )

    if not validate_llm_result(result):
        logger.warning("Groq response did not match the expected schema.")
        return refusal_response()

    quote_type = result["quote_type"]
    support_type = result["support_type"]

    # Refuse classifications that do not represent a supported answer.
    if support_type in {"not_specified", "not_found"}:
        return refusal_response()

    if support_type not in {"supported", "explicit_negative"}:
        return refusal_response()

    if result["status"] != "answered" or not result["answer"].strip():
        return refusal_response()

    if not result["evidence"]:
        return refusal_response()

    if quote_type in {"says_undefined_or_elsewhere", "none"}:
        return refusal_response()

    # The classification labels must agree with the quote type.
    if support_type == "supported" and quote_type != "gives_value_or_rule":
        return refusal_response()

    if support_type == "explicit_negative":
        if quote_type != "prohibits_or_excludes":
            return refusal_response()

        # A negative answer cannot be based on text that merely says
        # the topic is undefined or handled by another source.
        evidence_and_answer = [result["answer"]] + [
            item["quote"] for item in result["evidence"]
        ]
        if any(UNDEFINED_RE.search(text) for text in evidence_and_answer):
            if DEBUG:
                logger.debug(
                    "Refusing explicit_negative classification based on undefined wording."
                )
            return refusal_response()

    # Build trusted citations from the actual retrieved chunks.
    chunks_by_id = {
        chunk.get("id"): chunk
        for chunk in retrieved_chunks
        if isinstance(chunk, dict) and chunk.get("id")
    }

    citations = []
    seen = set()

    # IMPORTANT: all submitted evidence items must be valid.
    # Do not silently skip invalid evidence and accept a partial set.
    for item in result["evidence"]:
        chunk_id = item["chunk_id"]
        quote = item["quote"]

        source_chunk = chunks_by_id.get(chunk_id)
        if source_chunk is None:
            if DEBUG:
                logger.debug("Evidence references an unknown chunk ID: %s", chunk_id)
            return refusal_response()

        if not quote_is_supported(quote, source_chunk.get("text", "")):
            if DEBUG:
                logger.debug("Rejected non-verbatim evidence quote: %r", quote)
            return refusal_response()

        metadata = source_chunk.get("metadata") or {}
        document_name = metadata.get("document_name")
        section = metadata.get("section")

        if not document_name or not section:
            if DEBUG:
                logger.debug("Evidence chunk is missing citation metadata: %s", chunk_id)
            return refusal_response()

        citation_key = (document_name, section)
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(
                {
                    "document_name": document_name,
                    "section": section,
                }
            )

    if not citations:
        return refusal_response()

    return {
        "status": "answered",
        "answer": result["answer"].strip(),
        "citations": citations,
    }


def answer_question(question: str) -> dict[str, Any]:
    """Complete flow: question -> retrieval -> grounded answer."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved_chunks = retrieve_chunks(
        question=question,
        top_k=TOP_K,
    )

    return generate_grounded_answer(question.strip(), retrieved_chunks)
