# CLAUDE.md — LR Analysis Project

## What this project is

This is a research project analyzing how large language models (LLMs) approach **linguistic reasoning (LR) tasks** from the **WMT25 MIST shared task** (Multilingual Instruction Shared Task). The project is supervised by **Julia Kreutzer**, Senior Research Scientist at Cohere Labs, and conducted by **Rita Berrada**.

The MIST paper (Kocmi et al., 2025) benchmarked ~20 LLMs on five sub-tasks across 30 languages and published all prompts, outputs, and scores. This project focuses exclusively on the **LR sub-task** and asks a question the paper left open: the paper reports *what* scores models got, but not *how* they got there — what language they reasoned in, how long their responses were, how structured their reasoning was, and how any of that connects to performance.

**Paper:** https://aclanthology.org/2025.wmt-1.23.pdf  
**Data:** https://github.com/wmt-conference/wmt-mist/tree/main/data  
**Key reference for analysis methodology:** Yong et al. (2025) "Crosslingual Reasoning through Test-Time Scaling" https://arxiv.org/abs/2505.05408 — Julia is a co-author on this paper

---

## The LR sub-task: what the data looks like

- **15 prompt languages** (called `problem_language` in the data): Chinese, Czech, Dutch, English, Estonian, French, German, Japanese, Korean, Persian, Portuguese, Russian, Spanish, Swedish, Ukrainian
- **5 puzzle source languages** (called `instruction_language`): Koryak, Hadza, Komnzo, Dâw, Yanyuwa — these are the exotic IOL languages the puzzles are *about*, not the language the model is prompted in
- **90 prompts per metalanguage** = 1,350 total
- **5 task types**: classification (4), editing (1), fill-in-blanks (20), mapping (24), translation (41)
- **Scoring**: exact match [0–1] for classification, editing, fill-in-blanks, mapping — ChrF [0–1] for translation — each item has a point weight (0.5–2.5) — max attainable per language = 100

---

## Data files and what they contain

**`data/humeval/lr.json`** — gold metadata for every item. Columns: `id`, `points`, `meta`, `instruction_language`, `problem_language`, `type`, `eval_type`, `answer`, `prompt`. This is the ground truth file — 1,350 entries.

**`data/submissions/{Model}.json`** — raw model outputs. Each entry has exactly three fields:
- `taskid` — e.g. `linguistic_reasoning_1:a1_0`
- `answer` — the full raw model response text
- `tokens` — dict with `input_tokens`, `output_tokens`, `thinking_tokens` (for most models)

**`data/humeval_aggregated/lr.json`** — pre-computed language-level score totals (same as Table 6 in the paper). Rows = languages, columns = models. These are the scores out of 100 per language. Use these for language-level score correlations — do not recompute scores from scratch.

**Taskid join key:** to join submissions with humeval/lr.json, strip `linguistic_reasoning_` from the front and `_{N}` from the end of the taskid: `taskid.replace('linguistic_reasoning_', '').rsplit('_', 1)[0]` gives the `id` field (e.g. `1:a1`). The `N` suffix is the integer row index into df_lr.

---

## Existing notebook: `analysis_lr.ipynb`

A notebook already exists that has done the data loading and joining. **Use this notebook — do not rebuild from scratch.** Add new analysis cells after cell 25.

### What is already built

The notebook builds a master dataframe `df` with **1,350 rows × 50 columns**:

**Base columns** (from humeval/lr.json):
`N`, `id`, `points`, `instruction_language`, `problem_language`, `type`, `eval_type`, `answer`

**Per-model columns** (for each of 7 models, 6 columns each):

| Column | Description | Source |
|---|---|---|
| `score_{stem}` | Recomputed weighted score | Computed from raw answer + gold |
| `tokens_out_{stem}` | Output token count | Read directly from submission file |
| `tokens_think_{stem}` | Thinking token count | Read directly from submission file |
| `bracket_{stem}` | Bool: response contains `[...]` | Parsed from raw response |
| `pred_{stem}` | Extracted answer text | Parsed from raw response |
| `raw_{stem}` | Full model response text | Read directly from submission file |

Token counts are **read directly from the submission files**, not recalculated.

### Validation: confirmed and passed

The recomputed `score_` columns were validated against `humeval_aggregated/lr.json`:

| Model | Max diff | Mean diff | Status |
|---|---|---|---|
| Gemini 2.5 Pro | 7.000 | 1.006 | ✓ close enough |
| Claude 4 | 1.500 | 0.100 | ✓ close enough |
| DeepSeek V3 | 1.000 | 0.276 | ✓ close enough |
| GPT 4.1 | 3.500 | 1.341 | ✓ close enough |
| Llama 4 Maverick | 2.500 | 0.400 | ✓ close enough |
| CommandA | 1.500 | 0.155 | ✓ close enough |
| Mistral Medium | 4.073 | 0.511 | ✓ close enough |

Rankings and relative gaps are preserved. The `score_` columns are reliable for correlation analysis.

### Known issues — read before using

**⚠️ For language-level score correlations, prefer `humeval_aggregated/lr.json`** over the recomputed `score_` columns. For item-level correlations, the recomputed scores are fine.

---

## Models to use

Use all **7 models** including Mistral Medium.

| Model | Stem in notebook | Avg LR score |
|---|---|---|
| Gemini 2.5 Pro | `Gemini-2.5-Pro` | 36.3 |
| Claude 4 | `Claude-4` | 29.7 |
| DeepSeek V3 | `DeepSeek-V3` | 23.6 |
| GPT 4.1 | `GPT-4.1` | 23.4 |
| Llama 4 Maverick | `Llama-4-Maverick` | 22.9 |
| CommandA | `CommandA` | 19.8 |
| Mistral Medium | `Mistral-Medium` | — |

Define this at the top of any new cell:
```python
STEMS_USE = ['Gemini-2.5-Pro', 'Claude-4', 'DeepSeek-V3', 'GPT-4.1', 'Llama-4-Maverick', 'CommandA', 'Mistral-Medium']
```

---

## Key score patterns to explain (from Table 6 of the paper)

These are the patterns the analyses below are trying to explain:

- **Max score is only 40.9/100** (Gemini 2.5 Pro in Korean) — all models fail the majority of tasks. The best human at IOL 2024 scored 79, in their own language.
- **English is not the best language for top models:** Claude 4 scores 33.8 in German/Spanish but only 24.6 in English — its lowest across all 15 languages. GPT-4.1 scores 29.0 in Spanish but only 20.5 in English.
- **Gemini 2.5 Pro peaks in Korean (40.9)** — the single best score in the dataset.
- **GPT-4.1 underperforms on LR relative to other sub-tasks** — it's 2nd best on MT, OEG, XLSum, and LLM-as-a-judge, but 4th on LR.

---

## Analyses

The workflow for each analysis is:
**Step 1 — Measure** the property on every response in `df`
**Step 2 — Correlate** with score
**Step 3 — Ask why** the correlation exists (which model? which language? which task type?)

Do the analyses in order — findings from earlier ones inform later ones. Add each as new cells after cell 25 in the existing notebook.

---

### Analysis 1 — Language identification
*Julia's step: "Run language identification on the model's responses"*

**Step 1 — Measure:** Run a language ID tool on the `raw_{stem}` column for each of the 7 models. Recommended: `lingua-language-detector` (Python). Segment each response into two parts before running lang-ID:
- reasoning part: everything before the last `[...]`
- final answer part: the content inside the last `[...]`

New columns to add to `df`:
- `lang_full_{stem}` — detected language of the full response
- `lang_reasoning_{stem}` — detected language of the reasoning part
- `lang_answer_{stem}` — detected language of the final answer
- `lang_match_{stem}` — bool: does `lang_full` match `problem_language`?
- `quote_and_think_{stem}` — bool: model quotes non-English input, reasons in English, answers in target language (the pattern from Yong et al. 2505.05408)

**Step 2 — Correlate with score:** does reasoning in English (even when prompted in another language) correlate with higher `score_{stem}`? Does the quote-and-think pattern predict higher scores?

**Step 3 — Ask why:** if a correlation exists, is it driven by certain models? Certain prompt languages (does switching to English matter more for low-resource languages like Persian or Estonian)? Certain task types?

**Output to produce:**
- Table: for each (model, problem_language) — % of responses in English / prompt language / mixed
- Correlation table: mean score grouped by `lang_reasoning` (English vs other)
- Prevalence of quote-and-think pattern across models and languages

---

### Analysis 2 — Length
*Julia's step: "Analyze them in length"*

**Step 1 — Measure:** Compute character length of `raw_{stem}` for every response. Do NOT use `tokens_out_{stem}` — Gemini's token count is inflated and not comparable. Use character count consistently across all models.

New columns to add to `df`:
- `len_chars_{stem}` — character length of full `raw_{stem}`
- `len_chars_reasoning_{stem}` — character length of everything before the last `[...]`
- `len_chars_answer_{stem}` — character length of the content inside the last `[...]`

**Step 2 — Correlate with score:** compute Pearson and Spearman correlation between `len_chars_{stem}` and `score_{stem}` across all 1,350 rows. Do longer responses score higher? Does overthinking sometimes hurt?

**Step 3 — Ask why:** if length correlates with score, what drives it? Certain models? Certain prompt languages (are low-resource language responses shorter, and does that track with lower scores)? Certain task types?

**Cross-link with Analysis 1:** do responses that switch to English (`lang_reasoning == 'English'`) tend to be longer? Run mean `len_chars` grouped by `lang_reasoning`.

**Output to produce:**
- Table: mean `len_chars` per model (sorted by avg LR score)
- Table: mean `len_chars` per `problem_language`
- Correlation table: `len_chars` vs `score` overall and broken down by task type
- Scatter plot: `len_chars` vs `score`, colored by model

---

### Analysis 3 — Structure of reasoning traces
*Julia's step: "Analyze their structure (e.g. like in arxiv 2505.05408)"*

**Step 1 — Measure:** For each response, code whether it contains the following reasoning moves (use an LLM classifier or manual coding on a sample):
- `has_pattern_id` — model explicitly identifies a linguistic pattern or rule
- `has_rule_testing` — model tests the pattern against examples in the prompt
- `has_intermediate_steps` — model shows working between examples and final answer
- `has_self_correction` — model notices an error and revises
- `direct_answer` — model gives final answer with no visible reasoning

**Step 2 — Correlate with score:** which moves predict higher scores? Is structure a better predictor than raw length?

**Step 3 — Ask why:** is it certain models that reason more structurally? Certain prompt languages where structure degrades? Certain task types?

**Cross-links:** does reasoning in English (Analysis 1) produce more structured traces? Does a structured response tend to also be longer (Analysis 2)?

---

## Contextualizing pieces (after analyses 1–3)

### A — Single-model deep dive
Use **Claude 4** as the case study — starkest score gap: 33.8 in German/Spanish vs 24.6 in English. Use all three analyses together to explain why: different reasoning language? shorter responses? less structured reasoning? more formatting failures?

### B — Human comparison (qualitative)
Find the IOL 2024 gold-medal answer sheet on ioling.org/results/2024 (the participant who scored 79 in their own language). Apply the Analysis 3 coding scheme to their response and compare with the best-performing model on the same problem. Why do all models score below 41 when a human scored 79?

### C — Cross-task consistency
GPT-4.1 is 2nd best on MT, OEG, XLSum, LLM-as-a-judge — but 4th on LR. Use findings from analyses 1–3 to explain why.

---

## Optional (requires API access)

### D — Prompt sensitivity
The LR prompts instruct: "the last line of your response should only contain the solution within square brackets." Test: remove this constraint and re-run via Cohere's API. Does the score change? Separates instruction-following failures from reasoning failures.

### E — Test-time scaling
Current data is at temperature=0. Try Best-of-N and higher temperatures on Command A Reasoning via Cohere's API (API credit grant available from Julia).
