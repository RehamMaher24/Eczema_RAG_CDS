from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable
from .embedder import embedder_from_state
from .router import route_question_with_scores, EXPERT_BY_DOC
from .models import RetrievalHit
from .text_utils import tokenize
from .vector_store import SQLiteVectorStore


class GuidelineRetriever:
    def __init__(
        self,
        vector_store_path: Path,
        collection_name: str,
        dimension: int,
        top_k: int = 5,
        minimum_score: float = 0.05,
    ) -> None:
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.dimension = dimension
        self.top_k = top_k
        self.minimum_score = minimum_score

    def _search_single_expert(
        self,
        query: str,
        expert: str,
        *,
        top_k: int | None = None,
        minimum_score: float | None = None,
        include_reference_sections: bool = False,
    ) -> list[RetrievalHit]:
        expert_doc_ids = [
            doc_id for doc_id, assigned_expert in EXPERT_BY_DOC.items()
            if assigned_expert == expert
        ]

        return self.search(
            query,
            top_k=top_k,
            minimum_score=minimum_score,
            doc_ids=expert_doc_ids,
            include_reference_sections=include_reference_sections,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        minimum_score: float | None = None,
        doc_ids: Iterable[str] | None = None,
        include_reference_sections: bool = False,
    ) -> list[RetrievalHit]:
        query = query.strip()
        if not query:
            return []
        # #MoE
        # if doc_ids is None:
        #     experts = route_question(query)
        #     all_hits: list[RetrievalHit] = []

        #     for expert in experts:
        #         hits = self._search_single_expert(
        #             query,
        #             expert,
        #             top_k=top_k,
        #             minimum_score=minimum_score,
        #             include_reference_sections=include_reference_sections,
        #         )
        #         all_hits.extend(hits)

        #     seen: set[str] = set()
        #     merged: list[RetrievalHit] = []
        #     for hit in all_hits:
        #         if hit.chunk.chunk_id in seen:
        #             continue
        #         seen.add(hit.chunk.chunk_id)
        #         merged.append(hit)

        #     merged.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        #     return merged[: (top_k if top_k is not None else self.top_k)]
        # #end of routing, rest of logic is same as before
        requested_top_k = top_k if top_k is not None else self.top_k
        selected_doc_ids = sorted(set(doc_ids or []))
        routing_scores = (
            route_question_with_scores(query)
            if not selected_doc_ids
            else {}
        )


        with SQLiteVectorStore(self.vector_store_path, self.dimension) as store:
            state = store.get_model_state(self.collection_name)
            embedder = embedder_from_state(state)
            vector = embedder.embed_query(query)
            candidates = store.similarity_search(
                self.collection_name,
                vector,
                top_k=max(requested_top_k * 8, requested_top_k),
                minimum_score=(
                    minimum_score
                    if minimum_score is not None
                    else self.minimum_score
                ),
                doc_ids=doc_ids,
            )
        if not include_reference_sections:
            candidates = [hit for hit in candidates if not is_reference_chunk(hit)]
        for hit in candidates:
            hit.raw_score = hit.score
            hit.score = round(
                max(-1.0, min(1.0, hit.score + retrieval_adjustment(query, hit)+ routing_adjustment(hit, routing_scores))),
                6,
            )
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        selected = candidates[:requested_top_k]
        for rank, hit in enumerate(selected, start=1):
            hit.rank = rank
        return selected


def citation_for_hit(hit: RetrievalHit) -> str:
    chunk = hit.chunk
    page_range = (
        str(chunk.page_start)
        if chunk.page_start == chunk.page_end
        else f"{chunk.page_start}-{chunk.page_end}"
    )
    printed = ""
    if chunk.printed_page_start:
        printed_end = chunk.printed_page_end or chunk.printed_page_start
        printed_range = (
            chunk.printed_page_start
            if printed_end == chunk.printed_page_start
            else f"{chunk.printed_page_start}-{printed_end}"
        )
        printed = f"; printed page {printed_range}"
    section_path = " > ".join(clean_section_path(chunk.section_path))
    return (
        f"{chunk.document_name} — {section_path}; PDF page {page_range}{printed}; "
        f"chunk {chunk.chunk_id}; source {chunk.source_reference}"
    )


def routing_adjustment(
    hit: RetrievalHit,
    routing_scores: dict[str, float],
) -> float:
    """Apply a small soft-routing bonus without excluding other experts."""
    if not routing_scores:
        return 0.0
    expert = EXPERT_BY_DOC.get(hit.chunk.doc_id)
    return 0.08 * routing_scores.get(expert, 0.0)


def retrieval_adjustment(query: str, hit: RetrievalHit) -> float:
    chunk = hit.chunk
    query_lower = query.casefold()
    document_context = (
        f"{chunk.document_name} {chunk.topic} {chunk.scope}"
    ).casefold()
    evidence_context = (
        f"{' '.join(chunk.section_path)} {chunk.text}"
    ).casefold()
    section_context = " ".join(chunk.section_path).casefold()
    adjustment = 0.0
    clinical_evidence_request = any(
        marker in query_lower
        for marker in (
            "diagnos", "assess", "treat", "therapy", "management", "recommend",
            "should", "phototherapy", "patch test",
        )
    )
    if clinical_evidence_request and any(
        marker in section_context
        for marker in (
            "references", "bibliography", "acknowledg", "disclosure",
            "conflict of interest", "grant", "gaps in research", "research gaps",
        )
    ):
        adjustment -= 0.20
     # ================= START EVIDENCE RERANK FIX =================
    # Boost intent-matching clinical sections and demote background, disclosure, grant, and research-noise sections.
    if "patch test" in query_lower or "patch testing" in query_lower:
        if any(
            marker in section_context
            for marker in ("patch testing", "patch-test", "diagnostic tests", "diagnostic test")
        ):
            adjustment += 0.18
        if any(
            marker in section_context
            for marker in ("background", "introduction", "gaps in research", "disclosure", "conflict of interest", "acknowledg")
        ):
            adjustment -= 0.16

    if "phototherapy" in query_lower:
        if any(
            marker in section_context
            for marker in ("phototherapy", "uvb", "uva", "systemic agents", "treatment")
        ):
            adjustment += 0.18
        if any(
            marker in section_context
            for marker in ("gaps in research", "disclosure", "conflict of interest", "acknowledg")
        ) or "grant" in evidence_context:
            adjustment -= 0.20
    # ================== END EVIDENCE RERANK FIX ==================
    atopic_query = "atopic" in query_lower
    contact_query = "contact" in query_lower or "patch test" in query_lower
    if atopic_query:
        adjustment += 0.08 if "atopic" in document_context else -0.06
    if contact_query:
        adjustment += 0.08 if "contact" in document_context else -0.05

    query_terms = {
        canonical_term(term)
        for term in tokenize(query_lower)
        if len(term) >= 4 and term not in {"what", "when", "should", "included"}
    }
    evidence_terms = {canonical_term(term) for term in tokenize(evidence_context)}
    if query_terms:
        overlap = len(query_terms & evidence_terms) / len(query_terms)
        adjustment += 0.05 * overlap

    for phrase in (
        "diagnostic criteria",
        "clinical history",
        "quality of life",
        "trigger factors",
        "patch testing",
        "topical corticosteroids",
        "phototherapy",
        "disease flares",
    ):
        if phrase in query_lower and phrase in evidence_context:
            adjustment += 0.04

    if "child" in query_terms:
        if "under 12" in document_context or "child" in evidence_terms:
            adjustment += 0.07
        else:
            adjustment -= 0.03
    if "refer" in query_terms:
        adjustment += 0.11 if "refer" in evidence_terms else -0.04
    if "patch" in query_terms and "test" in query_terms:
        adjustment += 0.10 if "patch" in evidence_terms and "test" in evidence_terms else -0.04
    if "phototherapy" in query_terms:
        adjustment += 0.08 if "phototherapy" in evidence_terms else -0.03
    if "systemic" in query_terms:
        adjustment += 0.06 if "systemic" in evidence_terms else -0.02
    if "flare" in query_terms:
        adjustment += 0.07 if "flare" in evidence_terms else -0.03

    if chunk.word_count < 25:
        adjustment -= 0.08
    elif chunk.word_count < 60:
        adjustment -= 0.03
    if any(marker in evidence_context for marker in ("clinical questions used to structure", "section: disclaimer")):
        adjustment -= 0.05
    return adjustment



def canonical_term(term: str) -> str:
    mappings = {
        "children": "child",
        "childhood": "child",
        "referred": "refer",
        "referral": "refer",
        "referrals": "refer",
        "referring": "refer",
        "recommended": "recommend",
        "recommendation": "recommend",
        "recommendations": "recommend",
        "testing": "test",
        "tests": "test",
        "flares": "flare",
        "flaring": "flare",
        "treatments": "treatment",
    }
    return mappings.get(term, term)


def is_reference_chunk(hit: RetrievalHit) -> bool:
    chunk = hit.chunk
    path = " ".join(chunk.section_path).casefold()
    if any(
        marker in path
        for marker in (
            "references",
            "bibliography",
            "acknowledgments",
            "acknowledgements",
            "conflict of interest",
            "disclosures",
            "disclaimer",
        )
    ):
        return True
    body = chunk.text.casefold()
    section_leaf = chunk.section_path[-1] if chunk.section_path else ""
    if re.match(r"^\s*\d{2,4}\s+[A-Z]", section_leaf):
        return True
    if re.match(r"^\s*\d{1,4}\s+[A-Z]", section_leaf) and section_leaf.count(",") >= 2:
        return True
    if re.search(r"\b\d{1,4}\s+[A-Z][A-Za-z'-]+\s+[A-Z]", section_leaf) and re.search(
        r"\b(?:19|20)\d{2}\b", body
    ):
        return True
    citation_signals = 0
    citation_signals += body.count(" et al")
    citation_signals += len(re.findall(r"\b(?:19|20)\d{2};\d+", body))
    citation_signals += len(re.findall(r"\bdoi\s*:", body))
    citation_signals += len(re.findall(r"\b(?:j|br j|am j)\s+[a-z ]{2,20}\s+(?:19|20)\d{2}", body))
    return citation_signals >= 3 and chunk.word_count < 450


def clean_section_path(section_path: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in section_path:
        value = " ".join(item.split()).strip(" >")
        if not value or value.casefold() == "front matter":
            continue
        if "..." in value or re.search(r"\.{5,}\s*\d+$", value):
            continue
        if re.match(r"^(?:volume\s+\d+|j\s+am\s+acad\s+dermatol)$", value, re.I):
            continue
        if cleaned and cleaned[-1].casefold() == value.casefold():
            continue
        cleaned.append(value[:180])
    return cleaned or ["Section not reliably detected"]

