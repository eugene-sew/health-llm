# Health-domain LLM (GP-style general consultation): Prosit 1, Section C

Fine-tune an existing open LLM (**Qwen3.5-2B-Base**) on health text, and show it is effective **as a language model**.

## Pipeline
1. `collect.py` downloads open health text: PubMed Central (per-article licence kept), MedlinePlus, Ghana STG/EML, and Hugging Face datasets.
2. `prep.py` splits **held-out sets first**, builds continued-pre-training text, instruction data, and evaluation sets, and writes `colab_health_bundle.zip`.
3. Colab (T4). Put the zip in Drive under `MyDrive/health-llm/`, then run in order:
   - `notebooks/health_llm.ipynb`: one notebook, organised by the six Section C questions. Data → approaches → Stage 1 continued pre-training (LoRA) → Stage 2 instruction tuning → evaluation (base vs +CPT vs +CPT+SFT: perplexity on held-out PMC / MedlinePlus / Ghana STG / WikiText-2 forgetting control, zero-shot MCQ on MedQA, MedMCQA, PubMedQA, GP vignettes) → results → caveats.

`hl_lib.py` = shared helpers (prompt format, packing, perplexity, MCQ scoring). Self-tests: `python hl_lib.py`, `.venv/bin/python prep.py --selftest`, `.venv/bin/python collect.py --selftest`.

## Data and licences (checked 2026-09-21)
| Source | Use | Licence |
|---|---|---|
| PubMed Central OA (AWS open-data bucket), about 2,575 articles sampled | continued pre-training | per article: CC BY, CC0, CC BY-NC, CC BY-NC-SA (ND excluded); stored per document |
| MedlinePlus health topics (1,014) | continued pre-training | public domain (NLM), credit required |
| Ghana Standard Treatment Guidelines 2017, Essential Medicines List 2017 | continued pre-training (repeated x3) and held-out perplexity | Ghana MoH publications; check reuse terms before redistributing |
| medical-o1-reasoning-SFT (3,000 used) | instruction tuning | Apache-2.0 |
| MedText (1,300), MedQuAD (2,500), MedMCQA train (1,500), MedQA train (1,000) | instruction tuning | CC BY 4.0, CC BY 4.0 (verify, card has no field), Apache-2.0, CC BY 4.0 |
| MedQA test, MedMCQA validation, PubMedQA labeled | evaluation only (never trained on) | CC BY 4.0, Apache-2.0, MIT |
| WikiText-2 test | general-English perplexity control | CC BY-SA 3.0 |
| Deliberately NOT used | scraped patient-doctor chats (ChatDoctor / ai-medical-chatbot), StatPearls (CC BY-NC-ND), USMLE textbooks (copyright), GPL-licensed term lists | unclear or restrictive |

## Caveats to state in the report
- Educational use only. The model is not a clinical tool and its outputs are not validated by clinicians.
- MCQ benchmarks measure exam knowledge, not consultation quality. The GP vignettes are read by hand.
- The base model already saw much biomedical text, so gains may be modest. That is a legitimate finding.
- MedQA/MedMCQA *train* splits are in the instruction data; evaluation uses their *test/validation* splits.
