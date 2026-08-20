from __future__ import annotations
from pydantic import BaseModel, Field
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from .models import RetrievalHit
from .retriever import citation_for_hit

SYSTEM_PROMPT = """
You are a clinical guideline evidence assistant.

Use ONLY the supplied evidence. Never use external knowledge.

Return valid JSON only:

{
  "recommendation": "brief answer or Insufficient guideline evidence retrieved.",
  "evidence_summary": "summary based only on evidence",
  "claims": [
    {
      "claim": "one factual or clinical claim",
      "evidence": [
        {
          "chunk_id": "ID copied from an evidence block",
          "quote": "short exact quote copied from that chunk"
        }
      ]
    }
  ]
}

Rules:
- Every factual/clinical claim needs at least one evidence item.
- Each quote must be copied exactly from the matching evidence block.
- Do not create a claim if no supplied evidence supports it.
- Do not diagnose or prescribe.
""".strip()



class ClaimEvidence(BaseModel):
    chunk_id: str
    quote: str = Field(min_length=8, max_length=500)


class GeneratedClaim(BaseModel):
    claim: str = Field(min_length=1, max_length=1000)
    evidence: list[ClaimEvidence] = Field(min_length=1)


class GeneratedPayload(BaseModel):
    recommendation: str
    evidence_summary: str
    claims: list[GeneratedClaim] = Field(default_factory=list)


@dataclass(slots=True)
class GeneratedAnswer:
    status: str  # "answered" or "insufficient_evidence"
    answer: str
    citations: list[str]
    retrieval_scores: list[float]
    claims: list[GeneratedClaim]
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
                claims=[],
                refusal_reason=(
                    "No retrieved evidence met the configured confidence threshold."
                ),
            )

        evidence_context = self._format_evidence(selected_hits)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
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

        raw_answer = (response.choices[0].message.content or "").strip()
        if not raw_answer:
            raise RuntimeError("Groq returned an empty answer")

        payload = GeneratedPayload.model_validate_json(raw_answer)

        citations = self._citations_for_claims(payload.claims, selected_hits,)

        answer = f"""Recommendation:
        {payload.recommendation}

        Evidence summary:
        {payload.evidence_summary}

        Citations:
        {chr(10).join(f"- {citation}" for citation in citations)}

        Safety note:
        This is guideline evidence retrieval, not a diagnosis or substitute for clinical judgment.
        """

        return GeneratedAnswer(
            status="answered",
            answer=answer,
            citations=citations,
            retrieval_scores=[hit.score for hit in selected_hits],
            claims=payload.claims,
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
    @staticmethod
    def _citations_for_claims(
        claims: list[GeneratedClaim],
        hits: list[RetrievalHit],
    ) -> list[str]:
        hit_by_id = {
            hit.chunk.chunk_id: hit
            for hit in hits
        }

        citations: list[str] = []
        used_chunk_ids: set[str] = set()

        for claim in claims:
            for evidence in claim.evidence:
                chunk_id = evidence.chunk_id

                if chunk_id in used_chunk_ids:
                    continue

                hit = hit_by_id.get(chunk_id)
                if hit is None:
                    continue

                citations.append(citation_for_hit(hit))
                used_chunk_ids.add(chunk_id)

        return citations