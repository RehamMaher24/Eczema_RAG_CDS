# Fixed-Corpus Audit

**Author:** Manus AI  
**Scope:** Seven user-supplied PDFs only  
**Corpus policy:** No additional data search, replacement, or source-document modification

## Audit conclusion

All seven supplied PDFs are digitally extractable and contain internal provenance indicators such as an issuing organization, journal title, guideline identifier, DOI, or official source URL. The implementation therefore includes every document exactly as supplied and verifies its SHA-256 hash before every ingestion run. This audit records provenance visible in the files; it does not independently adjudicate clinical currency or grant redistribution rights.

| Document ID | PDF pages | Stated source | Internal provenance | Layout | Ingestion decision |
| --- | ---: | --- | --- | --- | --- |
| `nice_cg57_atopic_eczema_under_12` | 31 | NICE | Guideline identifier CG57 and NICE URL [1] | Predominantly single-column | Included unchanged; numbered recommendation hierarchy preserved. |
| `jtf_contact_dermatitis_2015` | 39 | AAAAI/ACAAI Joint Task Force | Journal DOI [2] | Two-column, tables, appendices | Included unchanged; repeated journal margins removed and figure references recorded. |
| `aad_ad_section1_diagnosis_2014` | 14 | American Academy of Dermatology | JAAD citation and DOI [3] | Two-column, tables and boxes | Included unchanged; references are filtered from default retrieval ranking. |
| `bad_contact_dermatitis_2017` | 13 | British Association of Dermatologists | BJD citation and DOI [4] | Two-column with numbered sections | Included unchanged; table captions that cannot be reconstructed are explicitly flagged. |
| `aad_ad_section2_topical_2014` | 17 | American Academy of Dermatology | JAAD citation and DOI [5] | Two-column with clinically relevant tables | Included unchanged; table content is retained where structural extraction succeeds. |
| `aad_ad_section4_flare_prevention_2014` | 16 | American Academy of Dermatology | JAAD citation and DOI [6] | Two-column with tables | Included unchanged; flare-prevention sections remain citation-addressable. |
| `aad_ad_section3_phototherapy_systemic_2014` | 23 | American Academy of Dermatology | JAAD citation and DOI [7] | Two-column with multiple treatment tables | Included unchanged; phototherapy and systemic-treatment evidence remains separately retrievable. |

## Immutable file register

| File | SHA-256 |
| --- | --- |
| `atopic-eczema-under-12.pdf` | `3b9071a77faf78ac5354fd90ec8173255fa92c496b5ccdcf64861cbfe9d826a5` |
| `Contact Dermatitis.pdf` | `92ec71e02ccebebdd3bf6a9f8b2732c5340c4ed91cb093e996ec6dd05c3baade` |
| `Diagnosis and assessment of atopic dermatitis.pdf` | `c21f67b524116018c267163ac6da85b7e24f1936a9b680628ef8d2b4c0222e0a` |
| `guidelines for the management of contact dermatitis.pdf` | `8aa45661456425430fff5908e42221e92697eb905763bd99b885a25a4d04bc1b` |
| `Management and treatment of atopic dermatitis.pdf` | `a8deff3c5533895676d201bd6aaa38ffc863989c77e0d4c81473fed098fbe95e` |
| `Prevention of disease flares atopic dermatitis.pdf` | `5aa4a7e4fe7845de39f788b28714a3c3c4e201106a1b0f2ee58cf780dfc08ec3` |
| `treatment with phototherapy atopic dermatitis.pdf` | `d8e6906d263f532af26dfcb334972fee7a8953dc9433d4c4f4c96cfc32e506d4` |

The machine-readable source of truth is `config/resources.json`; `config/corpus_sha256.txt` provides a compact checksum register. A mismatch causes ingestion to fail before parsing, preventing silent source drift.

## Parsing and retrieval risks

The journal PDFs use multi-column layouts, repeated journal headers, references, tables, and occasional image objects. The parser uses layout-aware extraction, repeated-margin detection, bibliography-resistant heading detection, caption-gated table extraction, and page-level warnings. Figure objects are recorded with page and bounding-box metadata but are not semantically interpreted.

The pipeline detected native text on all **153 pages**, so no OCR was required for this release. It produced **510 chunks** and **510 vectors**, extracted **35 structured tables**, and recorded **3 figure references**. Some table captions cannot be converted reliably into rows and columns; those pages retain their linearized text and receive `table_caption_detected_but_structure_not_extracted` warnings rather than being silently discarded.

## Rights and use boundary

The supplied PDFs retain their original notices. Repository metadata records a rights note for every source, and the pipeline does not alter the PDFs. Users are responsible for confirming that their intended storage, redistribution, and deployment comply with the applicable publisher or issuing-body terms.

## References

[1]: https://www.nice.org.uk/guidance/cg57 "NICE CG57"
[2]: https://doi.org/10.1016/j.jaip.2015.02.009 "Contact Dermatitis: A Practice Parameter-Update 2015"
[3]: https://doi.org/10.1016/j.jaad.2013.10.010 "Atopic dermatitis diagnosis and assessment"
[4]: https://doi.org/10.1111/bjd.15239 "BAD contact dermatitis guideline 2017"
[5]: https://doi.org/10.1016/j.jaad.2014.03.023 "Atopic dermatitis topical therapies"
[6]: https://doi.org/10.1016/j.jaad.2014.08.038 "Atopic dermatitis flare prevention"
[7]: https://doi.org/10.1016/j.jaad.2014.03.030 "Atopic dermatitis phototherapy and systemic agents"
