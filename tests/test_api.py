from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_service
from api.schemas import EvidenceItem, GroundingReview, ImagePrediction, RoutingResponse, ScopeCheckResponse, Timings


class FakeService:
    def health(self):
        return {"status": "ok", "collection": "fixed", "embedding_model": "test", "image_classifier_available": False}

    def retrieve(self, question, top_k=None, document_filters=None):
        if not question.strip():
            from api.services import ServiceError
            raise ServiceError(422, "Question must not be empty.")
        evidence = [EvidenceItem(rank=1, score=0.8, raw_score=0.7, text="Guideline text", citation="Fixed citation", document="Guideline", section="Treatment", pdf_page_start=2, pdf_page_end=2, chunk_id="chunk-1")]
        return ScopeCheckResponse(in_scope=True, confidence=0.99, reason="Eczema question.", status="ok"), RoutingResponse(experts=["eczema_treatment_topical_systemic"], weights={"eczema_treatment_topical_systemic": 1.0}), evidence, Timings(scope_check=1, retrieval=2, total=3)

    def chat(self, question, image_bytes=None, filename="", mime_type=None, top_k=None):
        scope, routing, evidence, timings = self.retrieve(question, top_k)
        if image_bytes and mime_type not in {"image/jpeg", "image/png"}:
            from api.services import ServiceError
            raise ServiceError(415, "Unsupported image type. Upload a JPG, JPEG, or PNG image.")
        return {"request_id": "test-request", "question": question, "scope_check": scope, "image_prediction": ImagePrediction(), "routing": routing, "evidence": evidence, "answer": "Evidence-grounded answer", "answer_status": "answered", "grounding_review": GroundingReview(status="judge_error", reason="Automated verification unavailable."), "warnings": ["Automated grounding verification was unavailable; retrieved evidence is still shown."], "timings_ms": timings}


app.dependency_overrides[get_service] = FakeService
client = TestClient(app)


def test_health_returns_metadata_without_index_rebuild():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["collection"] == "fixed"


def test_text_only_chat_works_without_image():
    response = client.post("/chat", data={"question": "How is eczema treated?"})
    assert response.status_code == 200
    assert response.json()["image_prediction"]["status"] == "not_available"
    assert response.json()["scope_check"]["in_scope"] is True
    assert response.json()["evidence"][0]["citation"] == "Fixed citation"


def test_valid_image_does_not_need_classifier():
    response = client.post("/chat", data={"question": "How is eczema treated?"}, files={"image": ("rash.png", b"png", "image/png")})
    assert response.status_code == 200
    assert response.json()["image_prediction"]["status"] == "not_available"


def test_unsupported_image_is_rejected():
    response = client.post("/chat", data={"question": "How is eczema treated?"}, files={"image": ("rash.gif", b"gif", "image/gif")})
    assert response.status_code == 415


def test_retrieve_returns_raw_and_final_scores_with_citations():
    response = client.post("/retrieve", json={"question": "How is eczema treated?"})
    assert response.status_code == 200
    evidence = response.json()["evidence"][0]
    assert evidence["raw_score"] == 0.7
    assert evidence["score"] == 0.8
    assert evidence["citation"]
