from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000").rstrip("/")
SAFETY_NOTICE = "This tool provides evidence-based information from the supplied eczema and dermatitis guidelines. It is not a diagnosis or a substitute for a qualified clinician. Confirm consequential decisions with a healthcare professional."

st.set_page_config(page_title="Eczema Clinical RAG Prototype", page_icon="🩺", layout="wide")
st.title("Eczema Clinical RAG Prototype")
st.warning(SAFETY_NOTICE)

with st.form("clinical_question"):
    question = st.text_area("Clinical question", placeholder="Ask a guideline evidence question…", height=100)
    image = st.file_uploader("Optional image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if image:
        st.image(image, caption="Uploaded image — not a confirmed diagnosis", width=260)
    submitted = st.form_submit_button("Retrieve evidence and answer", type="primary")

if submitted:
    if not question.strip():
        st.error("Enter a clinical question before submitting.")
    else:
        data = {"question": question.strip()}
        files = None
        if image:
            files = {"image": (image.name, image.getvalue(), image.type)}
        try:
            response = requests.post(f"{API_BASE_URL}/chat", data=data, files=files, timeout=75)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            st.error("The RAG API could not be reached. Start FastAPI and check RAG_API_BASE_URL.")
            st.caption(str(exc))
        else:
            prediction = payload["image_prediction"]
            scope = payload["scope_check"]
            st.caption(f"Scope check: {'in scope' if scope['in_scope'] else 'out of scope'} ({scope['confidence']:.0%}) — {scope['reason']}")
            prediction_detail = prediction["status"]
            if prediction.get("predicted_type"):
                prediction_detail += f" — {prediction['predicted_type']} ({prediction.get('confidence', 0):.1%})"
            elif prediction.get("confidence") is not None:
                prediction_detail += f" — confidence {prediction['confidence']:.1%}"
            st.caption(f"Image classifier: {prediction_detail}. Any model output is only a retrieval hint, never a diagnosis.")
            routing, timings = st.columns(2)
            with routing:
                st.subheader("Routing")
                st.json(payload["routing"])
            with timings:
                st.subheader("Timings (ms)")
                st.json(payload["timings_ms"])

            st.subheader("Retrieved evidence")
            if not payload["evidence"]:
                st.info("No evidence met the current retrieval threshold.")
            for item in payload["evidence"]:
                with st.expander(f"#{item['rank']} · {item['document']} · score {item['score']:.3f}", expanded=item["rank"] == 1):
                    st.caption(f"{item['section']} · PDF pages {item['pdf_page_start']}–{item['pdf_page_end']} · chunk {item['chunk_id']}")
                    st.write(item["text"])
                    st.code(item["citation"], language=None)

            st.subheader("Grounded answer")
            st.write(payload["answer"])
            review = payload["grounding_review"]
            st.subheader("Grounding review")
            st.info(f"{review['status']}: {review['reason']}")
            for warning in payload["warnings"]:
                st.warning(warning)
