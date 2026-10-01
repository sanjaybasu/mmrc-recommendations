# State maternal mortality review committee recommendations: content, evidence base, and downstream policy

Analysis code for "Content, Evidence Base, and Downstream Policy Uptake of State Maternal Mortality Review Committee Recommendations: A Cross-Sectional Study and Decision Analysis" (Basu S, et al.; manuscript under review).

The study classifies 3,351 prevention recommendations from 54 published state maternal mortality review committee reports, grades what each recommendation requests against the effectiveness literature, tests whether recommendations are associated with Medicaid managed care contracts, legislation, perinatal quality collaborative initiatives, and task force priorities, and projects the consequences of implementing them in a microsimulation of the Medicaid-financed US birth cohort.

This repository contains code only. It contains no data, no model outputs, and no manuscript files.

## Data

All inputs are published public documents.

- **Recommendation and findings corpus.** Extracted from state review committee reports with document- and page-level provenance by the open pipeline at [sanjaybasu/dark-health-data](https://github.com/sanjaybasu/dark-health-data). The scripts expect its processed output at `~/waymark-local/notebooks/dark-health-data/data/processed/mmrc/`.
- **Medicaid managed care procurement and contract documents.** Assembled for Basu S, et al. *Inquiry*. 2026;63:469580261444608; expected as plain text at `~/waymark-local/notebooks/rfp_analysis/processed_text/<State>/`. Available from the corresponding author.
- **Model parameters.** Every effect estimate, cost, prevalence, and utility is taken from a published source, with the citation stored alongside the value; values without a published source are labeled as assumptions and varied in sensitivity analysis.

Paths are set at the top of each script and can be edited for a different layout.

## Pipeline

Run from the project root in order. Scripts that call language models read `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from the environment.

| Step | Script | Purpose |
|---|---|---|
| Classification | `07_llm_classify.py`, `21_intervention_map.py`, `07b_llm_causes.py`, `25_dual_model_classify.py` | Classify recommendations (cause, policy lever, actor, coverage period, requested intervention) and committee cause labels with two models from different developers |
| Adjudication and validation | `26_adopt_dual_labels.py`, `27_adjudication_sheet.py`, `28_apply_adjudication.py`, `24_validation_packet.py`, `30_score_validation.py`, `32_test_retest.py`, `33_validation_stats.py`, `34_second_coder_packet.py`, `37_supplementary_packet.py` | Build the physician adjudication and blinded coding workbooks, apply adjudicated labels, score agreement with blinded physician coding, finalize evidence grades |
| Design-based correction | `36_dsl_inference.py` | Design-based supervised learning (prediction-powered difference) estimate of the share naming an intervention with evidence of benefit, with inclusion probabilities from the sampling design |
| Evidence grading | `29_evidence_grading.py` | Independent grading by the investigators and two models, with a physician adjudication sheet |
| Alignment | `08_classification_analysis.py`, `18_committee_composition.py`, `20_alignment_moderators.py` | Silence, concordance, and their correlates |
| State policy | `04_accountability_chain.py`, `06_withhold_proximity.py`, `19_propagation.py`, `35_contract_validation_packet.py`, `38_score_contract_validation.py` | Contract, legislation, quality collaborative, and task force linkage, with state and cause fixed effects; accuracy of the contract text rules; window widths set with `MMRC_CTX_WINDOW`, `MMRC_WH_WINDOW`, `MMRC_WINDOW_TAG` |
| Model | `09_build_params.py`, `10_model_inputs.py`, `11_microsim.py`, `12_interventions.py`, `13_portfolio_cea.py`, `14_uncertainty.py`, `31_oneway.py` | Microsimulation, portfolios, probabilistic, one-way, and structural sensitivity analyses |
| Exhibits and manuscript | `15_exhibits.py`, `16_tables.py`, `17_canonical.py`, `23_exhibits_md.py`, `22_render.py`, `39_redline.py` | Figures and tables from result files; every number in the manuscript is rendered from `canonical_numbers.json`; tracked-changes copy against an earlier render |

Scripts `00`-`06` are the preliminary rule-based analyses reported in the study protocol; the manuscript uses the language-model classification.

## Requirements

Python 3.11; see `requirements.txt`. `pdftotext` (Poppler) is required for `18_committee_composition.py`, and pandoc for `22_render.py`.

## License

Apache License 2.0.

## Contact

Sanjay Basu, MD, PhD, Department of Medicine, University of California, San Francisco (sanjay.basu@ucsf.edu).
