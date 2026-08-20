from dataclasses import dataclass
from .models import RetrievalHit


@dataclass
class GroundingValidation:
    valid: bool
    unsupported_claims: list[str]
    citation_errors: list[str]


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_claims(claims, hits: list[RetrievalHit]) -> GroundingValidation:
    chunks = {hit.chunk.chunk_id: hit.chunk for hit in hits}
    unsupported_claims: list[str] = []
    citation_errors: list[str] = []

    for claim in claims:
        if not claim.evidence:
            unsupported_claims.append(claim.claim)
            continue

        supported = False

        for reference in claim.evidence:
            chunk = chunks.get(reference.chunk_id)

            if chunk is None:
                citation_errors.append(
                    f"Claim cites a chunk that was not retrieved: {reference.chunk_id}"
                )
                continue

            # Exact source-quote validation.
            if normalise(reference.quote) not in normalise(chunk.text):
                citation_errors.append(
                    f"Quote is not present in chunk {reference.chunk_id}"
                )
                continue

            supported = True

        if not supported:
            unsupported_claims.append(claim.claim)

    return GroundingValidation(
        valid=not unsupported_claims and not citation_errors,
        unsupported_claims=unsupported_claims,
        citation_errors=citation_errors,
    )