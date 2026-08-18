# Ingestion Validation Report

Collection: `eczema_guidelines_gemini_embedding_001_768`

| Metric | Value |
| --- | ---: |
| Documents processed | 7 |
| Pages processed | 153 |
| Sections detected | 324 |
| Chunks created | 510 |
| Vectors indexed | 510 |
| Tables extracted | 35 |
| Figure references recorded | 3 |
| Pages flagged for OCR | 0 |

## Per-document results

| Document ID | Pages | Sections | Chunks | Vectors | Tables | Figures | OCR candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nice_cg57_atopic_eczema_under_12` | 31 | 95 | 96 | 96 | 2 | 0 | 0 |
| `jtf_contact_dermatitis_2015` | 39 | 80 | 136 | 136 | 4 | 2 | 0 |
| `aad_ad_section1_diagnosis_2014` | 14 | 23 | 47 | 47 | 6 | 0 | 0 |
| `bad_contact_dermatitis_2017` | 13 | 43 | 55 | 55 | 0 | 1 | 0 |
| `aad_ad_section2_topical_2014` | 17 | 23 | 51 | 51 | 5 | 0 | 0 |
| `aad_ad_section4_flare_prevention_2014` | 16 | 18 | 42 | 42 | 3 | 0 | 0 |
| `aad_ad_section3_phototherapy_systemic_2014` | 23 | 42 | 83 | 83 | 15 | 0 | 0 |

## Known Day 1 limitations

The parser detects pages that may need OCR but does not perform OCR. Image objects are recorded as page-level references but are not semantically interpreted. Tables are extracted heuristically and preserved as Markdown when detection succeeds; complex multi-column or spanning tables may need manual review. The local hashing embedding model is deterministic and dependency-light, but a production deployment may replace it through the embedding abstraction.
