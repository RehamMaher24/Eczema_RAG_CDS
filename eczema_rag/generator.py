from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from .models import RetrievalHit
from .retriever import citation_for_hit

SYSTEM_PROMPT = """
You are a clinical guideline evidence assistant.

Your response must use ONLY the supplied EVIDENCE blocks.
Do not use medical knowledge outside the evidence.
Do not diagnose a patient.
Do not invent recommendations, contraindications, doses, or citations.
If the evidence does not directly support an answer, state:
"Insufficient guideline evidence retrieved."
For an out-of-domain question, return exactly:
"This assistant is designed for eczema and dermatitis questions. I cannot
answer this question from the current clinical corpus. Please ask an
eczema- or dermatitis-related question."
Write in this exact structure:

Recommendation:
<brief evidence-grounded answer>

Evidence summary:
<faithful summary of the supplied evidence only>

Citations:
- <citation copied exactly from supplied evidence>

Safety note:
This is guideline evidence retrieval, not a diagnosis or substitute for clinical judgment.
""".strip()



@dataclass(slots=True)
class GeneratedAnswer:
    status: str  # "answered" or "insufficient_evidence"
    answer: str
    citations: list[str]
    retrieval_scores: list[float]
    refusal_reason: str | None = None


class GroundedAnswerGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv()

        api_key = os.getenv("GENERATOR_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GENERATOR_API_KEY or GROQ_API_KEY in environment")

        self.client = Groq(api_key=api_key, timeout=float(os.getenv("GENERATOR_TIMEOUT_SECONDS", "30")))
        self.model = str(config["model"])
        self.temperature = float(config["temperature"])
        self.max_tokens = int(config["max_tokens"])
        self.evidence_top_k = int(config["evidence_top_k"])
        self.minimum_retrieval_score = float(
            config["minimum_retrieval_score"]
        )
    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
    ) -> GeneratedAnswer:
        selected_hits = hits[: self.evidence_top_k]

        if (
            not selected_hits
            or selected_hits[0].score < self.minimum_retrieval_score
        ):
            return GeneratedAnswer(
                status="insufficient_evidence",
                answer=(
                    "I cannot provide a guideline-grounded answer because "
                    "the retrieved evidence is insufficient for this question."
                ),
                citations=[],
                retrieval_scores=[
                    hit.score for hit in selected_hits
                ],
                refusal_reason=(
                    "No retrieved evidence met the configured confidence threshold."
                ),
            )

        evidence_context = self._format_evidence(selected_hits)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question.strip()}\n\n"
                        f"Retrieved evidence:\n{evidence_context}"
                    ),
                },
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Groq returned an empty answer")

        return GeneratedAnswer(
            status="answered",
            answer=answer,
            citations=[
                citation_for_hit(hit) for hit in selected_hits
            ],
            retrieval_scores=[
                hit.score for hit in selected_hits
            ],
        )

    @staticmethod
    def _format_evidence(hits: list[RetrievalHit]) -> str:
        blocks: list[str] = []

        for index, hit in enumerate(hits, start=1):
            blocks.append(
                f"[EVIDENCE {index}]\n"
                f"Citation: {citation_for_hit(hit)}\n"
                f"Retrieval score: {hit.score}\n"
                f"Text:\n{hit.chunk.text}"
            )

        return "\n\n".join(blocks)
