from __future__ import annotations

from collections.abc import Iterable

EXPERTS = (
    "eczema_diagnosis_assessment",
    "contact_dermatitis_management",
    "eczema_treatment_topical_systemic",
    "eczema_prevention_flare_control",
)

EXPERT_BY_DOC = {
    "nice_cg57_atopic_eczema_under_12": "eczema_diagnosis_assessment",
    "aad_ad_section1_diagnosis_2014": "eczema_diagnosis_assessment",
    "jtf_contact_dermatitis_2015": "contact_dermatitis_management",
    "bad_contact_dermatitis_2017": "contact_dermatitis_management",
    "aad_ad_section2_topical_2014": "eczema_treatment_topical_systemic",
    "aad_ad_section3_phototherapy_systemic_2014": "eczema_treatment_topical_systemic",
    "aad_ad_section4_flare_prevention_2014": "eczema_prevention_flare_control",
}
#ur query got a keyword that matches the expert rules, then we route to that expert. If not, we route to all experts in order of priority.
EXPERT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "contact_dermatitis_management": (
        "contact dermatitis",
        "allergic contact",
        "irritant contact",
        "patch test",
        "patch testing",
        "allergy testing",
        "allergen",
        "sensitization",
        "eczema trigger",
    ),
    "eczema_diagnosis_assessment": (
        "eczema diagnosis",
        "atopic eczema",
        "diagnosis of eczema",
        "severity assessment",
        "eczema severity",
        "severity",
        "classify",
        "classification",
        "assess severity",
        "clinical history",
        "assess eczema",
        "diagnostic criteria",
        "quality of life",
        "clinical features",
    ),
    "eczema_treatment_topical_systemic": (
        "topical therapy",
        "topical treatment",
        "phototherapy",
        "systemic treatment",
        "systemic therapy",
        "treatment options",
        "moderate to severe",
        "therapy",
        "treatment",
    ),
    "eczema_prevention_flare_control": (
        "flare prevention",
        "prevent flares",
        "preventing flares",
        "flare control",
        "flare trigger",
        "flare triggers",
        "eczema flare",
        "reduce flare",
        "maintenance therapy",
        "adjunctive therapy",
        "trigger factors",
        "prevention",
        "preventive",
        "maintenance",
    ),
}


def normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def expert_for_doc(doc_id: str | None) -> str:
    if not doc_id:
        return "eczema_diagnosis_assessment"
    return EXPERT_BY_DOC.get(doc_id, "eczema_diagnosis_assessment")


def route_question(question: str) -> list[str]:
    '''return a list of experts to route the question to, based on the user query.'''
    text = normalize_question(question)
    scores = {expert: 0.0 for expert in EXPERTS}

    for expert_name, keywords in EXPERT_KEYWORDS.items():
        scores[expert_name] = sum(
            1.0 for keyword in keywords if keyword in text
        )

    ranked = [
        expert for expert, score in sorted(scores.items(), key=lambda x: (-x[1], EXPERTS.index(x[0])))
        if score > 0
    ]

    if not ranked:
        return list(EXPERTS[:2])  # or list(EXPERTS)

    return ranked[:2]


def route_question_with_scores(question: str) -> dict[str, float]:
    """Return a weight map for each expert, (soft routing) based on the user query."""
    text = normalize_question(question)
    scores = {expert: 0.0 for expert in EXPERTS}

    for expert_name, keywords in EXPERT_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if keyword in text:
                score += 1.0
        scores[expert_name] = round(score, 4)

    total = sum(scores.values())
    if total == 0:
        return {expert: 1.0 / len(EXPERTS) for expert in EXPERTS}

    return {expert: round(scores[expert] / total, 4) for expert in EXPERTS}

#in case of hard routing, we can use this
def top_expert(question: str) -> str:
    scores = route_question_with_scores(question)
    return max(scores, key=scores.get)

def experts_for_docs(doc_ids: Iterable[str | None]) -> set[str]:
    return {expert_for_doc(doc_id) for doc_id in doc_ids if doc_id}
