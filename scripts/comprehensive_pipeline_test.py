#!/usr/bin/env python3
"""
Comprehensive end-to-end RAG pipeline test.

Covers:
- MoE routing (router.py)
- Evidence retrieval with expert filtering (retriever.py)
- Citation formatting
- Answer generation (generator.py)
- Answer judging (judge.py)

Output: Detailed results with citations, generated responses, judge reviews, and validation checklist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eczema_rag.config import PipelineConfig
from eczema_rag.router import route_question_with_scores
from eczema_rag.retriever import GuidelineRetriever, citation_for_hit
from eczema_rag.generator import GroundedAnswerGenerator
from eczema_rag.judge import GroundedAnswerJudge


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def format_checklist_item(status: bool, label: str) -> str:
    """Format a checklist item."""
    symbol = "[OK]" if status else "[FAIL]"
    return f"{symbol} {label}"


def test_pipeline() -> None:
    """Run comprehensive pipeline tests."""
    
    # Load configuration - use Gemini embedder
    print_section("LOADING CONFIGURATION")
    config = PipelineConfig.load(ROOT, ROOT / "config" / "pipeline_gemini.json")
    print(f"[OK] Configuration loaded (Gemini embedder)")
    print(f"  - Embedding model: {config.embedding.get('model', 'unknown')}")
    print(f"  - Vector store: {config.vector_store.get('path', 'unknown')}")
    print(f"  - Collection: {config.collection_name}")
    
    # Extract config values
    vector_store_path = Path(config.vector_store.get("path"))
    embedding_dimension = config.embedding.get("dimension", 768)
    retrieval_top_k = config.retrieval.get("top_k", 5)
    retrieval_minimum_score = config.retrieval.get("minimum_score", 0.0)
    generation_config = config.generation
    judge_config = config.judge
    
    # Initialize components
    print_section("INITIALIZING PIPELINE COMPONENTS")
    
    retriever = GuidelineRetriever(
        vector_store_path=vector_store_path,
        collection_name=config.collection_name,
        dimension=embedding_dimension,
        top_k=retrieval_top_k,
        minimum_score=retrieval_minimum_score,
    )
    print(f"[OK] Retriever initialized")
    
    generator = GroundedAnswerGenerator(generation_config)
    print(f"[OK] Generator initialized (model: {generator.model})")
    
    judge_instance = GroundedAnswerJudge(judge_config)
    print(f"[OK] Judge initialized (model: {judge_instance.model})")
    
    # Test questions covering all 4 expert classes
    test_questions = [
        "How should eczema severity be assessed and classified?",
        "When is patch testing recommended for contact dermatitis diagnosis?",
        "What are the phototherapy options available for treating moderate to severe eczema?",
        "What preventive strategies help reduce eczema flare triggers?",
    ]
    
    # Validation checklist
    validation_results = {
        "total_questions": len(test_questions),
        "routing_success": 0,
        "retrieval_success": 0,
        "citations_formatted": 0,
        "generation_success": 0,
        "judge_review_success": 0,
        "answer_approved": 0,
        "answer_grounded": 0,
    }
    
    all_results = []
    
    # Process each question
    for question_index, question in enumerate(test_questions, start=1):
        print_section(f"QUESTION {question_index}/{len(test_questions)}: {question}")
        
        result = {
            "question_number": question_index,
            "question": question,
            "routing": None,
            "retrieval": None,
            "generation": None,
            "judge_review": None,
            "validation": {},
        }
        
        # ============================================
        # 1. ROUTING
        # ============================================
        print_subsection("1. EXPERT ROUTING (MoE)")
        try:
            routed_experts_dict = route_question_with_scores(question)
            # Convert dict to list of tuples for consistent handling
            routed_experts = list(routed_experts_dict.items())
            result["routing"] = [
                {"expert": exp, "score": round(score, 4)}
                for exp, score in routed_experts
            ]
            print(f"[OK] Question routed to {len(routed_experts)} expert(s)")
            for expert, score in routed_experts:
                print(f"  - {expert}: {score:.4f}")
            result["validation"]["routing"] = True
            validation_results["routing_success"] += 1
        except Exception as e:
            print(f"[FAIL] Routing failed: {e}")
            result["validation"]["routing"] = False
            all_results.append(result)
            continue
        
        # ============================================
        # 2. RETRIEVAL WITH EXPERT FILTERING
        # ============================================
        print_subsection("2. EVIDENCE RETRIEVAL (with Expert Filtering)")
        try:
            hits = retriever.search(question, top_k=retrieval_top_k)
            result["retrieval"] = {
                "count": len(hits),
                "hits": [],
            }
            
            if not hits:
                print(f"[WARN] No evidence retrieved for this question")
                result["validation"]["retrieval"] = False
            else:
                print(f"[OK] Retrieved {len(hits)} evidence chunk(s)")
                result["validation"]["retrieval"] = True
                validation_results["retrieval_success"] += 1
                
                # Display retrieved chunks with citations
                for rank, hit in enumerate(hits, start=1):
                    print(f"\n  [{rank}] EVIDENCE CHUNK (Score: {hit.score:.6f})")
                    
                    # Format citation
                    citation = citation_for_hit(hit)
                    print(f"      Citation: {citation}")
                    
                    # Add to result
                    hit_data = {
                        "rank": rank,
                        "score": round(hit.score, 6),
                        "chunk_id": hit.chunk.chunk_id,
                        "document": hit.chunk.document_name,
                        "section": " > ".join(hit.chunk.section_path),
                        "page_start": hit.chunk.page_start,
                        "page_end": hit.chunk.page_end,
                        "citation": citation,
                        "text": hit.chunk.text[:200] + "..." if len(hit.chunk.text) > 200 else hit.chunk.text,
                    }
                    result["retrieval"]["hits"].append(hit_data)
                    
                    # Print text preview
                    text_preview = hit.chunk.text[:150].replace("\n", " ")
                    print(f"      Text: {text_preview}...")
                
                result["validation"]["citations_formatted"] = True
                validation_results["citations_formatted"] += 1
        
        except Exception as e:
            print(f"[FAIL] Retrieval failed: {e}")
            result["validation"]["retrieval"] = False
            all_results.append(result)
            continue
        
        # Skip generation/judge if no evidence
        if not hits:
            print_subsection("3. ANSWER GENERATION - SKIPPED (insufficient evidence)")
            print_subsection("4. ANSWER JUDGMENT - SKIPPED (insufficient evidence)")
            all_results.append(result)
            continue
        
        # ============================================
        # 3. ANSWER GENERATION
        # ============================================
        print_subsection("3. ANSWER GENERATION (Grounded)")
        try:
            generated = generator.generate(question, hits)
            result["generation"] = {
                "status": generated.status,
                "answer": generated.answer,
                "citations": generated.citations,
                "retrieval_scores": [round(s, 6) for s in generated.retrieval_scores],
                "refusal_reason": generated.refusal_reason,
            }
            
            if generated.status == "insufficient_evidence":
                print(f"[WARN] Generation refused: {generated.refusal_reason}")
                result["validation"]["generation"] = False
            else:
                print(f"[OK] Answer generated successfully")
                try:
                    print(f"\n  ANSWER:\n{generated.answer[:500]}...")
                except UnicodeEncodeError:
                    print(f"\n  ANSWER: [Unicode content - see JSON output for full text]")
                print(f"\n  CITATIONS:")
                for i, citation in enumerate(generated.citations, start=1):
                    try:
                        print(f"    {i}. {citation[:100]}...")
                    except UnicodeEncodeError:
                        print(f"    {i}. [Citation - see JSON output for full text]")
                result["validation"]["generation"] = True
                validation_results["generation_success"] += 1
        
        except Exception as e:
            print(f"[FAIL] Generation failed: {e}")
            result["validation"]["generation"] = False
            all_results.append(result)
            continue
        
        # Skip judge if generation was refused
        if generated.status != "answered":
            print_subsection("4. ANSWER JUDGMENT - SKIPPED (answer refused)")
            all_results.append(result)
            continue
        
        # ============================================
        # 4. ANSWER JUDGMENT
        # ============================================
        print_subsection("4. ANSWER JUDGMENT (Evidence Validation)")
        try:
            review = judge_instance.review(question, generated.answer, hits)
            result["judge_review"] = {
                "decision": review.decision,
                "grounded": review.grounded,
                "citation_valid": review.citation_valid,
                "unsupported_claims": review.unsupported_claims,
                "citation_errors": review.citation_errors,
                "reason": review.reason,
            }
            
            print(f"[OK] Judge review completed")
            print(f"  - Decision: {review.decision.upper()}")
            print(f"  - Grounded: {review.grounded}")
            print(f"  - Citations valid: {review.citation_valid}")
            if review.unsupported_claims:
                print(f"  - Unsupported claims:")
                for claim in review.unsupported_claims:
                    print(f"    • {claim}")
            if review.citation_errors:
                print(f"  - Citation errors:")
                for error in review.citation_errors:
                    print(f"    • {error}")
            print(f"  - Reason: {review.reason}")
            
            result["validation"]["judge_review"] = True
            validation_results["judge_review_success"] += 1
            
            if review.decision == "approved":
                validation_results["answer_approved"] += 1
            if review.grounded:
                validation_results["answer_grounded"] += 1
        
        except Exception as e:
            print(f"[FAIL] Judge review failed: {e}")
            result["validation"]["judge_review"] = False
        
        all_results.append(result)
    
    # ============================================
    # FINAL VALIDATION CHECKLIST
    # ============================================
    print_section("FINAL VALIDATION CHECKLIST")
    
    print(format_checklist_item(
        validation_results["routing_success"] == validation_results["total_questions"],
        f"Expert routing: {validation_results['routing_success']}/{validation_results['total_questions']} questions routed"
    ))
    
    print(format_checklist_item(
        validation_results["retrieval_success"] == validation_results["total_questions"],
        f"Evidence retrieval: {validation_results['retrieval_success']}/{validation_results['total_questions']} questions retrieved chunks"
    ))
    
    print(format_checklist_item(
        validation_results["citations_formatted"] == validation_results["total_questions"],
        f"Citation formatting: {validation_results['citations_formatted']}/{validation_results['total_questions']} questions have formatted citations"
    ))
    
    print(format_checklist_item(
        validation_results["generation_success"] > 0,
        f"Answer generation: {validation_results['generation_success']}/{validation_results['total_questions']} questions answered"
    ))
    
    print(format_checklist_item(
        validation_results["judge_review_success"] > 0,
        f"Judge review: {validation_results['judge_review_success']}/{validation_results['total_questions']} answers reviewed"
    ))
    
    if validation_results["judge_review_success"] > 0:
        print(format_checklist_item(
            validation_results["answer_approved"] > 0,
            f"Approved answers: {validation_results['answer_approved']}/{validation_results['judge_review_success']} reviews approved"
        ))
        
        print(format_checklist_item(
            validation_results["answer_grounded"] > 0,
            f"Grounded answers: {validation_results['answer_grounded']}/{validation_results['judge_review_success']} reviews grounded"
        ))
    
    # ============================================
    # SAVE DETAILED RESULTS
    # ============================================
    print_section("SAVING RESULTS")
    output_dir = ROOT / "outputs" / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "comprehensive_pipeline_test.json"
    
    output_data = {
        "test_date": str(Path(__file__).stat().st_mtime),
        "validation_summary": validation_results,
        "detailed_results": all_results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Detailed results saved to: {output_file}")
    
    # Print summary
    print_section("TEST SUMMARY")
    print(f"Total questions tested: {validation_results['total_questions']}")
    print(f"Routing success: {validation_results['routing_success']}/{validation_results['total_questions']}")
    print(f"Retrieval success: {validation_results['retrieval_success']}/{validation_results['total_questions']}")
    print(f"Generation success: {validation_results['generation_success']}/{validation_results['total_questions']}")
    print(f"Judge reviews: {validation_results['judge_review_success']}/{validation_results['total_questions']}")
    print(f"Approved answers: {validation_results['answer_approved']}")
    print(f"Grounded answers: {validation_results['answer_grounded']}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n[FAIL] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
