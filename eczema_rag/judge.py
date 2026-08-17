from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from .models import RetrievalHit
from .retriever import citation_for_hit


JUDGE_SYSTEM_PROMPT = """
You are a strict clinical RAG output reviewer.

You do NOT provide medical advice.
You only review whether a proposed answer is supported by the supplied evidence.

Check:
1. Every clinical claim is supported by the evidence.
2. The answer adds no outside medical knowledge.
3. Every citation in the answer matches one of the supplied citations.
4. The answer does not claim diagnosis or certainty beyond the evidence.

Return ONLY valid JSON using this exact schema:

{
  "decision": "approved" | "revise" | "refuse",
  "grounded": true | false,
  "citation_valid": true | false,
  "unsupported_claims": ["..."],
  "citation_errors": ["..."],
  "reason": "..."
}

Decision rules:
- approved: all important claims are supported and citations are valid.
- revise: evidence exists, but the answer has unsupported claims or citation errors.
- refuse: evidence is insufficient for the question.
""".strip()


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

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY in environment")

        self.client = Groq(api_key=api_key)
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
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return JudgeReview(
                decision="revise",
                grounded=False,
                citation_valid=False,
                unsupported_claims=[],
                citation_errors=[],
                reason="Judge returned invalid JSON; answer must be reviewed.",
            )

        return JudgeReview(
            decision=str(payload.get("decision", "revise")),
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