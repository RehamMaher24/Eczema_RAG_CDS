from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from .models import RetrievalHit
from .retriever import citation_for_hit


JUDGE_SYSTEM_PROMPT = """You are a strict clinical RAG output reviewer. Your ONLY job is to output valid JSON.

CRITICAL: Output ONLY the JSON object. No text before or after.

Given:
- Question: clinical query
- Answer: proposed response  
- Evidence: supplied chunks

Task: Verify if the answer is grounded in the evidence.

Output this exact JSON (no markdown, no explanation):

{
  "decision": "approved",
  "grounded": true,
  "citation_valid": true,
  "unsupported_claims": [],
  "citation_errors": [],
  "reason": "All claims are supported by evidence."
}

Decisions:
- "approved": All claims supported, citations valid
- "revise": Some claims unsupported or citation errors  
- "refuse": Insufficient evidence
""".strip()


def extract_json_from_response(response_text: str) -> dict[str, Any]:
    """Extract JSON from response, handling markdown code blocks and variations."""
    response_text = response_text.strip()
    
    if not response_text:
        raise ValueError("Empty response from judge")
    
    # Try to extract JSON from markdown code block
    for pattern in [r"```(?:json)?\s*({.*?})\s*```", r"```({.*?})```"]:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    # Try parsing the whole response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract just the first { } block
    brace_start = response_text.find('{')
    if brace_start >= 0:
        brace_count = 0
        for i in range(brace_start, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(response_text[brace_start:i+1])
                    except json.JSONDecodeError:
                        pass
                    break
    
    raise ValueError(f"Could not extract valid JSON from response: {response_text[:300]}")


@dataclass(slots=True)
class JudgeReview:
    decision: str
    grounded: bool
    citation_valid: bool
    unsupported_claims: list[str]
    citation_errors: list[str]
    reason: str


class GroundedAnswerJudge:
    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv()

        api_key = os.getenv("JUDGE_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing JUDGE_API_KEY or GROQ_API_KEY in environment")

        self.client = Groq(api_key=api_key, timeout=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "15")))
        self.model = str(config["model"])
        self.temperature = float(config["temperature"])
        self.max_tokens = int(config["max_tokens"])

    def review(
        self,
        question: str,
        answer: str,
        hits: list[RetrievalHit],
    ) -> JudgeReview:
        evidence = self._format_evidence(hits)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": JUDGE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Proposed answer:\n{answer}\n\n"
                        f"Allowed evidence and citations:\n{evidence}"
                    ),
                },
            ],
        )

        raw_response = (response.choices[0].message.content or "").strip()

        try:
            payload = extract_json_from_response(raw_response)
        except (json.JSONDecodeError, ValueError) as e:
            return JudgeReview(
                decision="revise",
                grounded=False,
                citation_valid=False,
                unsupported_claims=[],
                citation_errors=[],
                reason=f"Judge response parsing failed: {str(e)[:100]}",
            )

        return JudgeReview(
            decision=str(payload.get("decision", "revise")).lower(),
            grounded=bool(payload.get("grounded", False)),
            citation_valid=bool(payload.get("citation_valid", False)),
            unsupported_claims=[
                str(item) for item in payload.get("unsupported_claims", [])
            ],
            citation_errors=[
                str(item) for item in payload.get("citation_errors", [])
            ],
            reason=str(payload.get("reason", "")),
        )

    @staticmethod
    def _format_evidence(hits: list[RetrievalHit]) -> str:
        return "\n\n".join(
            (
                f"[EVIDENCE {index}]\n"
                f"Allowed citation: {citation_for_hit(hit)}\n"
                f"Text:\n{hit.chunk.text}"
            )
            for index, hit in enumerate(hits, start=1)
        )
