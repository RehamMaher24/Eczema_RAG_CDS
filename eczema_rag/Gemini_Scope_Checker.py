from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# FIX: Add a fail-closed Gemini scope gate that classifies the user question before any expensive RAG step.
SCOPE_SYSTEM_PROMPT = """
You are a strict scope classifier for an eczema and dermatitis clinical RAG system.

Return JSON only, with exactly these keys:
{
  "in_scope": true or false,
  "confidence": number between 0 and 1,
  "reason": "short explanation"
}

IN SCOPE means the user question is directly about:
- atopic dermatitis or atopic eczema;
- allergic or irritant contact dermatitis;
- eczema or dermatitis symptoms, diagnosis, clinical history, or severity;
- eczema-related itch, rash, triggers, flares, or maintenance;
- patch testing or contact-allergen evaluation;
- topical therapy, emollients, phototherapy, systemic therapy, or other eczema treatment;
- prevention or management of eczema and dermatitis.

OUT OF SCOPE means the question is mainly about cancer, heart disease,
politics, finance, unrelated diseases, or any subject not directly related to
eczema or dermatitis.

Important rules:
1. Classify only the USER QUESTION.
2. Do not inspect or infer from retrieved evidence; no evidence is provided here.
3. A word such as "cancer" appearing in an eczema guideline does not make a
   general cancer question relevant to this assistant.
4. If the question is ambiguous and cannot clearly be connected to eczema or
dermatitis, return in_scope=false.
5. Return JSON only. Do not add Markdown, commentary, or code fences.
""".strip()



@dataclass(frozen=True, slots=True)
class ScopeDecision:
    in_scope: bool
    confidence: float
    reason: str
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeminiScopeChecker:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        confidence_threshold: float = 0.70,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv(
            "GEMINI_SCOPE_MODEL",
            os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )
        self.confidence_threshold = confidence_threshold

    def check(self, question: str) -> ScopeDecision:
        question = " ".join((question or "").split())
        if not question:
            return ScopeDecision(
                in_scope=False,
                confidence=1.0,
                reason="The question is empty.",
                status="invalid_question",
            )

        prompt = f"{SCOPE_SYSTEM_PROMPT}\n\nUSER QUESTION:\n{question}"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw_text = (getattr(response, "text", "") or "").strip()
            payload = json.loads(raw_text)
            in_scope = bool(payload["in_scope"])
            confidence = max(0.0, min(1.0, float(payload["confidence"])))
            reason = str(payload.get("reason", "")).strip()

            # Treat low-confidence decisions as out of scope so uncertain queries
            # cannot trigger expensive retrieval or unsupported clinical answers.
            if confidence < self.confidence_threshold:
                return ScopeDecision(
                    in_scope=False,
                    confidence=confidence,
                    reason=reason or "The scope decision was uncertain.",
                    status="uncertain",
                )

            return ScopeDecision(
                in_scope=in_scope,
                confidence=confidence,
                reason=reason,
                status="ok",
            )
        except Exception as exc:
            # Fail closed for a clinical assistant: do not continue when scope
            # verification is unavailable or the model returns malformed JSON.
            return ScopeDecision(
                in_scope=False,
                confidence=0.0,
                reason="Scope verification was unavailable; the request was stopped safely.",
                status=f"scope_check_error:{type(exc).__name__}",
            )
