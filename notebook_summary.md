# `analysis_lr.ipynb` — Complete Summary

This document is a full faithful summary of the notebook. It captures every section, every code cell's logic, every output table, every finding and interpretive note. It is intended as a drop-in substitute for reading the notebook itself.

---

## Top-level overview

The notebook is a deep-dive analysis of the **WMT25 MIST Linguistic Reasoning (LR) sub-task**, covering 1,350 puzzles (IOL 2024) administered to 7 LLMs in 15 metalanguages. It has 85 cells and is organized into:

- **Section 0–6**: data setup (load, join, score, validate, overview)
- **Analysis 1**: language identification — does the model reason in the prompt language?
- **Analysis 2a** (called "Analysis 2" in the notebook): LR task-type decomposition
- **Analysis 2b** (called "Analysis 2 — Length"): response length vs score
- **Analysis 4**: reasoning structure (steps, self-correction, hedging, no-reasoning, quote-and-think)

---

## Section 0 — Imports & Configuration

### Libraries
```python
json, re, pathlib.Path
numpy, pandas
sacrebleu.metrics.CHRF
matplotlib.pyplot, seaborn
random (seed 42, np.random.seed 42)
```

### Paths
```
ROOT         = /Users/ritaberrada/Desktop/wmt-mist
HUMEVAL_LR   = data/humeval/lr.json
AGG_LR       = data/humeval_aggregated/lr.json
AGG_OEG      = data/humeval_aggregated/oeg.json
AGG_XLSUM    = data/humeval_aggregated/xlsum.json
SUBS_DIR     = data/submissions/
```

### Model registry

| Stem (used in column names) | Display name |
|---|---|
| `Gemini-2.5-Pro` | Gemini 2.5 Pro |
| `Claude-4` | Claude 4 |
| `DeepSeek-V3` | DeepSeek V3 |
| `GPT-4.1` | GPT 4.1 |
| `Llama-4-Maverick` | Llama 4 Maverick |
| `CommandA` | CommandA |
| `Mistral-Medium` | Mistral Medium |

`STEMS = list(TOP_MODELS.keys())`, `DISPLAY = list(TOP_MODELS.values())`

---

## Section 1 — Load `humeval/lr.json` → `df_lr`

### Shape and columns
- **1,350 rows**, index name `N` (promoted to column).
- Columns: `N`, `id`, `points`, `meta`, `instruction_language`, `problem_language`, `type`, `eval_type`, `answer`, `prompt`

### Distributions (from output)
```
Task type × eval method:
  classification  exact match     60
  editing         exact match     15
  fill-in-blanks  exact match    300
  mapping         exact match    360
  translation     chrF           615

Items per problem_language (all 15 languages, 90 each):
  English, Czech, German, French, Portuguese, Russian, Ukrainian,
  Estonian, Dutch, Spanish, Swedish, Japanese, Chinese, Korean, Persian

Items per instruction_language (puzzle source):
  Dâw        405
  Komnzo     315
  Hadza      240
  Yanyuwa    240
  Koryak     150

Points distribution:
  0.5 → 585 items
  1.0 → 225 items
  1.5 → 225 items
  2.0 → 285 items
  2.5 →  30 items

id=None rows: 21  (Dutch/Komnzo artifact — always join on N, never on id)
```

### Task type × instruction language (totals across all 15 metalanguages)
```
                  classification  editing  fill-in-blanks  mapping  translation
Dâw                           0        0               0      300          105
Hadza                         0        0               0       60          180
Komnzo                        0       15             300        0            0
Koryak                        0        0               0        0          150
Yanyuwa                      60        0               0        0          180
```

### Key structural facts (from markdown cell 8)
- 90 items per metalanguage = the same set of puzzles in 15 languages.
- Per language: translation(41) + mapping(24) + fill-in-blanks(20) + classification(4) + editing(1) = 90.
- Translation uses chrF (partial credit); others use exact match (binary).
- 21 items have `id=None` — always join on `N`.
- Point values: editing(2.0), translation(1.0–2.5), fill-in-blanks(0.5–2.5), mapping(0.5), classification(1.0).

---

## Section 2 — Load Aggregated Scores → `df_agg`

Source: `data/humeval_aggregated/lr.json`. Pre-computed language-level totals (max 100 per language) from the paper.

### Full `df_agg` (top 7 models shown)
```
language      Gemini 2.5 Pro  Claude 4  DeepSeek V3  GPT 4.1  Llama 4 Maverick  CommandA  Mistral Medium
Chinese               28.8      26.2         17.4     12.7              14.2      15.7            14.8
Czech                 36.8      29.3         21.5     24.6              22.1      20.1            15.1
Dutch                 37.5      27.9         24.1     27.1              23.2      23.2            20.8
English               38.3      24.6         23.2     20.5              22.4      17.8            20.4
Estonian              32.5      27.0         24.5     29.4              19.4      18.8            14.6
French                39.9      32.9         27.9     27.4              26.5      20.6            21.2
German                37.3      33.8         28.2     24.3              25.9      18.8            24.9
Japanese              35.4      29.8         23.1     21.6              20.2      21.3            22.9
Korean                40.9      26.3         20.4     15.3              20.2      21.8            15.2
Persian               29.4      26.3         20.6     21.8              21.3      18.2            15.7
Portuguese            38.1      35.5         27.9     27.9              27.2      21.1            23.5
Russian               39.5      30.8         22.4     24.1              24.6      18.8            21.8
Spanish               40.3      33.8         28.5     29.0              30.5      [~22]           [~22]
Swedish               35.5      28.4         21.8     22.7              20.8      [~21]           [~19]
Ukrainian             33.9      32.1         22.9     23.0              24.5      [~18]           [~19]
Average               36.3      29.7         23.6     23.4              22.9      19.8            19.8 (≈)
```

### Top 8 by average LR score (from paper)
```
1. Gemini 2.5 Pro       avg = 36.28
2. Claude 4             avg = 29.65
3. DeepSeek V3          avg = 23.62
4. GPT 4.1              avg = 23.43
5. Llama 4 Maverick     avg = 22.87
6. CommandA             avg = 19.79
7. Mistral Medium       avg = 19.78
8. Qwen3 235B           avg = 17.63  (not in our top 7)
```

The notebook analyses the top **7** models (6th and 7th tied), not 8th.

---

## Section 3 — Load Model Submissions

Function `load_lr_entries(stem)`:
- Opens `data/submissions/{stem}.json`
- Filters to entries where `taskid.startswith("linguistic_reasoning_")`
- Parses `N = int(taskid.rsplit("_", 1)[-1])` — N is the row index into `df_lr`
- Returns `dict: N → entry_dict`

Token handling:
- Most models: `tokens = {input_tokens, output_tokens, thinking_tokens, finish_reason}`
- Gemini: some `tokens = None`
- Non-top-8 models: some have bare int tokens — handled gracefully

All 7 models: **1,350 LR entries each** (confirmed).

---

## Section 4 — Scoring Utilities

### `extract_answer(text)`
1. Regex find all `[...]` (non-nested): `re.findall(r'\[([^\[\]]*)\]', text)`
2. Return **last match** (handles multi-bracket reasoning traces)
3. Fallback: last non-empty line (for models ignoring bracket format)

### `score_item(pred, gold, eval_type, points)`
- `exact match` → 1.0 if `pred.lower() == gold.lower()`, else 0.0, multiplied by points
- `chrF` → `sacrebleu sentence_score(pred, [gold]).score / 100.0`, multiplied by points

Smoke-test results:
```
extract_answer('Let me think... [Paris]') → 'Paris'
extract_answer('The answer is Paris.')    → 'The answer is Paris.'
score_item('paris','Paris','exact match',1.0) → 1.0
score_item('',    'Paris','exact match',1.0) → 0.0
score_item('Paris','Paris','chrF',2.0)       → 2.000
```

---

## Section 5 — Build Master DataFrame `df`

### Shape
**1,350 rows × 50 columns** (8 base + 6 per model × 7 models)

### Base columns
`N`, `id`, `points`, `instruction_language`, `problem_language`, `type`, `eval_type`, `answer`

### Per-model columns (for each stem)
| Column | Description |
|---|---|
| `score_{stem}` | Weighted score (chrF or exact_match × points) |
| `tokens_out_{stem}` | Output tokens (NaN if unavailable) |
| `tokens_think_{stem}` | Thinking tokens (>0 only for Gemini 2.5 Pro) |
| `bracket_{stem}` | Bool: response contains `[...]` |
| `pred_{stem}` | Extracted answer text |
| `raw_{stem}` | Full model response |

Build takes ~30 seconds. Confirmed: `df.shape = (1350, 50)`.

Sample row (N=0, id=1:a1, Koryak/English/translation):
- gold = "you(sg) lead him"
- `score_Gemini-2.5-Pro` = 2.0 (perfect)
- `tokens_out_Gemini-2.5-Pro` = 9282
- `tokens_think_Gemini-2.5-Pro` = (large, from thinking model)
- `bracket_Gemini-2.5-Pro` = True
- `pred_Gemini-2.5-Pro` = [extracted]
- `raw_Gemini-2.5-Pro` = full text

### 5.1 Validation against `df_agg`

```
Model                   Max |diff|   Mean |diff|  Status
------------------------------------------------------------
Gemini 2.5 Pro               7.000         1.006  ✓ close enough
Claude 4                     1.500         0.100  ✓ close enough
DeepSeek V3                  1.000         0.276  ✓ close enough
GPT 4.1                      3.500         1.341  ✓ close enough
Llama 4 Maverick             2.500         0.400  ✓ close enough
CommandA                     1.500         0.155  ✓ close enough
Mistral Medium               4.073         0.511  ✓ close enough
```
Threshold: max diff < 8 pts out of 100. Rankings and relative gaps preserved → `df` reliable for analysis.

---

## Section 6 — Quick Overview of `df`

### Per-model summary (all 1,350 LR items)
```
                  total_pts  mean_item  pct_zero  bracket_fail_%  mean_out_tokens
Gemini 2.5 Pro       559.22       0.41     43.70            1.56         11837.47
Claude 4             446.25       0.33     47.19            0.00           540.32
DeepSeek V3          358.43       0.27     49.04            0.15           438.21
GPT 4.1              371.60       0.28     52.52            3.78           572.90
Llama 4 Maverick     349.02       0.26     52.67            0.00           764.82
CommandA             299.12       0.22     52.22            0.15           291.88
Mistral Medium       304.44       0.23     52.00            0.30           446.44
```

Key observations:
- Gemini has by far the most output tokens (11,837 avg) — inflated by thinking tokens.
- Claude 4 has 0% bracket failure (always formats correctly).
- GPT 4.1 has 3.78% bracket failure (highest among all models).
- ~43–53% of responses score 0 across models.

### Heatmap
A seaborn heatmap (model × language, YlOrRd, vmin=0 vmax=45) of `df_agg_top` was produced. Key visual: Gemini 2.5 Pro dominates across all languages; Korean 40.9 is the single highest cell.

---

## Analysis 1 — Language Identification

### Question
What language does the model reason in, and does staying in the prompt language predict higher scores?

### Method
1. Install `lingua-language-detector`.
2. Build detector restricted to the 15 prompt languages only (speeds up vs. full 75-language model).
3. Define Unicode allowlist to strip IOL exotic characters before detection:
   - `(0x0000, 0x024F)` — Basic Latin + Extended Latin A/B
   - `(0x0400, 0x04FF)` — Cyrillic
   - `(0x0600, 0x06FF)` — Arabic/Persian
   - `(0x3000, 0x9FFF)` — CJK, Hiragana, Katakana
   - `(0xAC00, 0xD7FF)` — Hangul
4. `segment_response(text)` → everything before the last `[...]`
5. `strip_iol_tokens(text)` → remove tokens containing characters outside the allowlist
6. `detect_lang(text)` → run lingua on cleaned text, return title-cased language name or None

### New columns added to `df`
- `lang_reasoning_{stem}` — detected language of reasoning part (or None)
- `lang_match_{stem}` — 1 if detected == problem_language, 0 if mismatch, None if undetected

### Detection coverage
```
                  detected  null  null_pct
Gemini-2.5-Pro       1099   251     18.6%
Claude-4             1350     0      0.0%
DeepSeek-V3          1251    99      7.3%
GPT-4.1               995   355     26.3%
Llama-4-Maverick     1350     0      0.0%
CommandA             1350     0      0.0%
Mistral-Medium       1124   226     16.7%
```

### Why nulls happen

**Gemini 2.5 Pro (251 nulls):** For some prompt language × puzzle combos, Gemini outputs only the final answer with no visible reasoning (e.g., just `[G]`). The model did reason — it used up to 22k thinking tokens internally — but wrote nothing visible. Example: id=4:a10 in Chinese and several other languages outputs only `[F]`, `[G]`, `[C]` with up to 22,389 thinking tokens (Portuguese).

**GPT-4.1 (354 nulls):** 305 rows have `reasoning_len == 0` (no text before bracket); 50 rows have text that is stripped to below 10 chars by IOL stripping. GPT-4.1 frequently answers with very short responses like `[C]`, `C`, `D` with no reasoning.

**DeepSeek-V3 (99 nulls):** 96 rows have `reasoning_len == 0`; 3 rows killed by stripping. DeepSeek on English/mapping tasks often outputs just `[H]`, `[D]`, etc.

**Mistral-Medium (226 nulls):** 162 rows have `reasoning_len == 0`; 64 rows killed by stripping. Czech translation responses often output just `[vedou]`, `[ty je(dv) chytíš]` with no reasoning preamble.

### Note on thinking tokens (cell 35)
Only Gemini 2.5 Pro has non-zero `tokens_think`. All other models have `tokens_think = 0`. Gemini's `tokens_out` includes both internal thinking and visible output (e.g., tokens_out=9282 but only ~593 visible tokens). This is why **Analysis 2 (Length) uses character length of `raw_`, not token counts** — character count is always the visible output regardless of whether a model has internal thinking.

### Mismatch summary (raw, before manual verification)
```
Total mismatch instances (all models pooled): 267
  → switched to English:           189
  → switched to another language:  78

Per model:
                  mismatches  → English  → English%  → other  → other%
Gemini-2.5-Pro             8          1        12%        7       88%
Claude-4                   0          0          -        0         -
DeepSeek-V3               14          8        57%        6       43%
GPT-4.1                   58          8        14%       50       86%
Llama-4-Maverick         157        155        99%        2        1%
CommandA                   7          0         0%        7      100%
Mistral-Medium            23         17        74%        6       26%
```

### Manual verification per model

**Gemini 2.5 Pro (8 mismatches) — 0 genuine:**
All 8 are false positives. 7 caused by IOL stripping being imperfect: exotic characters (Yanyuwa words like `nyu-malbulu kanyalu-yabima`) survived the Unicode allowlist because they are ASCII and confused lingua. The 8th (id=4:a5, Swedish prompt) is also a false positive: the Swedish prompt contained the English loanword "flip-flops" as an answer option, which lingua picked up as English. **Gemini always reasons in the prompt language.**

**Claude 4 (0 mismatches):** Perfect. Claude always reasons in the prompt language, with full detection coverage (1350/1350). No genuine switches.

**GPT-4.1 (58 mismatches) — 0 genuine:**
All 58 are false positives. Two causes: (1) Yanyuwa words are pure ASCII (e.g., `nyu-bardibardilu kanda-yabimanjila`) so stripping didn't catch them, and lingua misidentified them as Estonian or French; (2) some Russian and Ukrainian responses were misidentified as Estonian — a lingua accuracy issue between similar languages. **GPT-4.1 always reasons in the prompt language.**

**CommandA (7 mismatches) — 0 genuine:**
All 7 are lingua detection errors from puzzle language words contaminating otherwise correct prompt-language reasoning (Czech reasoning with Dâw words, Korean reasoning with Komnzo names). **CommandA always reasons in the prompt language.**

**DeepSeek-V3 (14 mismatches) — 6 genuine:**
8 are false positives (same ASCII IOL word issue). **6 are genuine English switches:** all occur on Chinese-prompted, Komnzo fill-in-blanks tasks (ids: 3:a1, 3:a3, 3:a9, 3:a10, 3:a12, 3:b2). The reasoning is entirely in English, full structured analysis. DeepSeek is the first model with confirmed language switching — and it only happens for Chinese prompts on Komnzo puzzles.

**Mistral-Medium (23 mismatches) — 16 genuine:**
**16 real switches:** all the same pattern — Estonian prompt + English reasoning + fill-in-blanks + Komnzo. The reasoning is entirely in English (numbered steps, structured analysis), zero Estonian. The IOL stripping made no difference because there is no IOL contamination — it is just pure English. 7 remaining mismatches are false positives.

**Llama-4-Maverick (157 mismatches) — 155 genuine:**
2 false positives (1 Korean/Dâw detected as Czech, 1 Persian/Hadza detected as Estonian from IOL ASCII words). **155 genuine English switches.**

Breakdown by task type × prompt language:
```
                 editing  fill-in-blanks  mapping  translation  TOTAL
Chinese                0               0        5            0      5
Estonian               0              16        0            0     16   ← (all Mistral)
French                 0              20        0            1     21
German                 0               4        0            0      4
Japanese               1              20        5           18     44
Korean                 1              17        0           10     28
Persian                0               9        0            0      9
Portuguese             0              13        0            0     13
Russian                0              10        0            0     10
Ukrainian              1              19        1            0     21
TOTAL                  3             134       11           29    177   ← combined all models
```

Fill-in-blanks is overwhelmingly the trigger (134/177 = 76%). Japanese (44) and Korean (28) are the most affected languages. English and Spanish: zero switches across all models. Chinese only on mapping.

### Final summary after manual verification
```
model             detected  genuine_switches  notes
Gemini-2.5-Pro       1099                 0  251 nulls — no visible reasoning (thinking tokens)
Claude-4             1350                 0
GPT-4.1               996                 0  354 nulls — short/empty reasoning
CommandA             1350                 0
DeepSeek-V3          1251                 6  Chinese prompts, Komnzo fill-in-blanks only
Mistral-Medium       1124                16  Estonian prompts, Komnzo fill-in-blanks only
Llama-4-Maverick     1350               155  See breakdown above
```

**Key finding:** Most models reason in the prompt language. The exception is Llama-4-Maverick, which switches to English on 155 items, concentrated in fill-in-blanks and translation tasks, especially for non-Latin-script languages (Japanese, Korean).

### Note on scores (cell 58)
Per-item scores recomputed as: exact match (0/1) or chrF (0–1). Published data gives only language-level totals out of 100. Recomputation is validated to within ~4 pts max. `score_{stem}` is the raw 0–1 per-item score (before multiplying by points). Mean score across rows means the fraction of available points earned on average.

---

## Analysis 2a — LR Task-Type Decomposition

### Visual 1 — Overall Difficulty Ranking by Subtask Type

Method: compute `norm_score = score / points × 100` per item, then average across all 7 models × 15 languages.

```
Subtask type       Mean score  N items (unique)  Max pts/language  Eval type
Translation (ChrF)       33.2                41              64.0       chrF
Classification           28.1                 4               4.0  exact match
Mapping                  14.2                24              12.0  exact match
Fill-in-blanks           12.0                20              18.0  exact match
Editing                   1.0                 1               2.0  exact match
```

Findings:
- Editing (1 item, score=1.0) — too small to interpret.
- **Fill-in-blanks (12.0) is the hardest exact-match subtask.**
- Mapping (14.2) close behind.
- Classification (28.1) easiest exact-match subtask, but only 4 unique items.
- Translation (33.2) highest, but **not comparable** — uses ChrF (partial credit) vs exact match (binary). ChrF overestimates quality; the high position reflects metric leniency, not easier reasoning.

Saved to: `figures/visual1_subtask_difficulty.png`

### Visual 2 — Model Rankings per Subtask (Bump Chart)

Editing excluded. Rankings across: classification, fill-in-blanks, mapping, translation(ChrF).

Findings:
- **Gemini 2.5 Pro is #1 on every subtask.** Dominance is consistent, not driven by one task type.
- **Claude 4 is stable #2** except on fill-in-blanks, where DeepSeek V3 takes #2. Notable: fill-in-blanks is exactly where DeepSeek shows English-switching (Chinese/Komnzo) — doesn't seem to hurt its score.
- **GPT-4.1 biggest swing:** #7 on mapping but #3 on translation. Weak at pattern-matching; ChrF partial credit may flatter fluent-but-imperfect output.

Saved to: `figures/visual2b_bump_chart.png`

---

## Analysis 2b — Length

### Question
Do longer responses score higher? Does overthinking sometimes hurt?

### Step 1 — FLORES-200 normalization

FLORES-200 devtest used to compute average sentence character lengths per language. Normalization factor = avg chars / English avg chars (130.4).

```
Language       Avg chars    Ratio vs English
French           155.8        1.195
Spanish          155.1        1.190
German           152.0        1.166
Dutch            145.8        1.118
Portuguese       141.7        1.087
Russian          140.4        1.077
Ukrainian        132.9        1.019
Swedish          130.6        1.002
English          130.4        1.000
Estonian         127.9        0.981
Czech            125.7        0.964
Persian          122.4        0.938
Korean            65.2        0.500
Japanese          56.3        0.431
Chinese           42.7        0.328
```

Rationale: Chinese/Japanese/Korean pack more meaning per character; normalizing by FLORES ratio makes `norm_len` comparable across languages ("how many FLORES-sentence-equivalents did the model write").

### New columns added to `df`
- `char_len_{stem}` — raw character count of `raw_{stem}`
- `norm_len_{stem}` — `char_len / flores_ratio[problem_language]`

df shape after: (1350, 85)

### Sanity check (Claude 4)
```
Language       Ratio    Mean raw len     Mean norm len
Chinese        0.328    660              2015
Czech          0.964    1152             1195
Dutch          1.118    1365             1221
English        1.000    1517             1517
Estonian       0.981    1081             1102
French         1.195    1405             1177
German         1.166    1282             1100
Japanese       0.431    652              1511
Korean         0.500    716              1433
Persian        0.938    867               924
Portuguese     1.087    1385             1274
Russian        1.077    1203             1117
Spanish        1.190    1414             1188
Swedish        1.002    1145             1144
Ukrainian      1.019    1159             1138
```
After normalization, CJK languages (low raw chars) come up to ~1200–2000, much closer to English/European values.

### Score vs length per model
```
Model                   Avg score   Avg norm_len
Gemini 2.5 Pro               36.3           1688
Claude 4                     29.7           1270
DeepSeek V3                  23.6           1234
GPT 4.1                      23.4           1534
Llama 4 Maverick             22.9           2726
CommandA                     19.8            933
Mistral Medium               19.8           1232
```

Key observations:
- **Llama 4 Maverick writes the longest responses (2726 normalized chars) but scores 22.9** — among the lower performers. Length does not drive score.
- **CommandA writes the shortest responses (933) and also scores lowest (19.8)** — consistent with a weak baseline rather than conciseness helping.
- Gemini writes the most (1688) AND scores highest (36.3), but it is also a qualitatively different model (thinking tokens). Not a clean correlation.
- No strong monotonic relationship between length and score across models.

Saved to: `figures/analysis2_score_vs_length.png`

### Correct vs wrong response length (exact-match tasks only)
```
Model                   Correct len    Wrong len    Ratio  (N correct / N wrong)
Gemini 2.5 Pro                 1570         1581    0.993  (182 / 553)
Claude 4                       1104         1194    0.924  (121 / 614)
DeepSeek V3                    1038         1051    0.987  (104 / 631)
GPT 4.1                        1468         1519    0.966   (77 / 658)
Llama 4 Maverick               2431         2291    1.061   (92 / 643)
CommandA                        773          795    0.972   (70 / 665)
Mistral Medium                 1208         1196    1.009   (85 / 650)
```

Key observation: **Ratios hover near 1.0 for all models.** Correct responses are not systematically longer or shorter than wrong ones. No evidence of "overthinking" (longer → wrong) or "more effort → correct" at the response level. Llama is slightly longer when correct (ratio 1.061) but the difference is small.

Saved to: `figures/analysis2_correct_vs_wrong_len.png`

---

## Analysis 4 — Reasoning Structure

### Features coded (regex-based)
```python
RE_STEPS     = r'(?:step\s*[1-9]|##\s*step\s*[1-9]|^\s*[1-9][.)]\s|\b(?:first[,:]|second[,:]|third[,:]))'
RE_SELF_CORR = r'(?:wait[,!\s]|actually[,!\s]|no[,!\s]|i made a mistake|let me reconsider|\bcorrection\b|i was wrong|let me re.?examine|on second thought)'
RE_HEDGING   = r'(?:maybe|perhaps|possibly|i'm not sure|it seems|could be|i think\b)'
no_reasoning = lambda t: bracket starts within first 20 chars, or response < 30 chars total
```

### New columns added to `df`
- `has_steps_{stem}` — bool
- `has_self_correction_{stem}` — bool
- `has_hedging_{stem}` — bool
- `no_reasoning_{stem}` — bool

df shape after: (1350, 113)

### Quote-and-think v2 (`has_quote_think_v2_{stem}`)

Also added in this section. Definition: model quotes ≥3 distinct IOL puzzle tokens in its reasoning (inside quotation marks), indicating it is actively citing puzzle evidence.

**Method:**
1. Build per-puzzle IOL token vocabulary from the `Context:` section of each prompt:
   - Strategy A: non-ASCII Latin tokens (Koryak, Komnzo, Dâw)
   - Strategy B: tokens from numbered lines before colon/arrow (Hadza, Yanyuwa)
   - Remove a stopword list
2. `n_quoted_iol_{stem}` = count of distinct IOL tokens found inside quotation marks in the reasoning
3. `has_quote_think_v2_{stem}` = True if `n_quoted_iol >= 3`

IOL vocabulary stats:
```
Puzzles with IOL token vocab: 90 / 90
Median token count per puzzle: 37
Koryak (id=1:a1): ['ccallala', 'ccen', 'ccuhe', 'enan', 'enanŋevlatək', 'evlat', ...]
Hadza  (id=2:a1): ['aikuitcha', 'aisa', 'akhwitibee', 'athobeema', 'athuitcha', 'bee', ...]
Komnzo (id=3:a1): ['abia', 'afe', 'ame', 'bäiŋaf', 'bäiŋam', 'enat', ...]
Dâw    (id=4:a1): ['brazilië', 'dak', 'dâw', 'dôo', 'keet', 'nâax', ...]
Yanyuwa(id=5:a1): ['australië', 'vóór', ...]
```

Quote-think-v2 rates:
```
Gemini 2.5 Pro         v2=61.3%  (median quoted tokens: 5)
Claude 4               v2=59.9%  (median quoted tokens: 4)
DeepSeek V3            v2=50.9%  (median quoted tokens: 3)
GPT 4.1                v2=40.4%  (median quoted tokens: 0)
Llama 4 Maverick       v2=67.3%  (median quoted tokens: 5)
CommandA               v2=36.4%  (median quoted tokens: 2)
Mistral Medium         v2=51.0%  (median quoted tokens: 3)
```

df shape after: (1350, 127)

### Feature: Steps

**Definition:** Explicit step-by-step structure (numbered steps, ordinals, step headers).

**Prevalence per model** (bar chart saved as `figures/analysis4_steps_per_model.png`). Models ordered by avg LR score — the chart shows what % of their 1,350 responses contain steps.

**Score with vs without steps** (bar chart saved as `figures/analysis4_steps_score_bar.png`). Shows mean normalized score (0–100) for responses with steps vs without. No large consistent advantage observed across all models.

### Feature: Self-correction

**Definition:** Model notices a mistake mid-response and explicitly revises (backtracking phrases: wait, actually, no, I made a mistake, let me reconsider, correction, I was wrong, let me re-examine, on second thought).

**Prevalence per model** (bar chart saved as `figures/analysis4_selfcorr_per_model.png`). Upper y-axis limit = 30%.

**Prevalence per subtask** (bar chart saved as `figures/analysis4_selfcorr_per_subtask.png`). Shows which task types trigger more self-correction. Upper y-axis limit = 25%.

**Score with vs without self-correction** (bar chart saved as `figures/analysis4_selfcorr_score.png`). Upper y-axis limit = 50%.

### Feature: Hedging

**Definition:** Model signals uncertainty before/around its conclusion (maybe, perhaps, possibly, I'm not sure, it seems, could be, I think).

**Prevalence per model** (bar chart saved as `figures/analysis4_hedging_per_model.png`). Upper y-axis limit = 25%.

**Score with vs without hedging** (bar chart saved as `figures/analysis4_hedging_score.png`). Upper y-axis limit = 50%.

Note in markdown: hedging could go either way — calibrated model that knows what it doesn't know (→ higher score) or genuine confusion (→ lower score).

---

## Complete column inventory of `df`

At the end of all analysis cells, `df.shape = (1350, 127)`.

### Base (8 columns)
`N, id, points, instruction_language, problem_language, type, eval_type, answer`

### Per-model (7 models × original 6 = 42 columns)
`score_{stem}, tokens_out_{stem}, tokens_think_{stem}, bracket_{stem}, pred_{stem}, raw_{stem}`

### Analysis 1 language columns (7 × 2 = 14 columns)
`lang_reasoning_{stem}, lang_match_{stem}`

### Analysis 2 length columns (7 × 2 = 14 columns)
`char_len_{stem}, norm_len_{stem}`

### Analysis 2 normalized score (7 columns)
`norm_score_{stem}`

### Analysis 4 structure features (7 × 4 = 28 columns)
`has_steps_{stem}, has_self_correction_{stem}, has_hedging_{stem}, no_reasoning_{stem}`

### Analysis 4 quote-think (7 × 2 = 14 columns)
`n_quoted_iol_{stem}, has_quote_think_v2_{stem}`

**Total: 8 + 42 + 14 + 14 + 7 + 28 + 14 = 127 columns** (df shape reported at end of Analysis 4 cells — Analysis 5 was removed before any additional columns were added).

---

## Figures produced

All saved to `figures/`:
- `visual1_subtask_difficulty.png` — bar chart: mean normalized score per subtask type
- `visual2b_bump_chart.png` — rank bump chart: model positions across subtasks
- `len_raw_vs_norm_3langs.png` — 2×3 grid: raw vs normalized length for English/French/Korean
- `analysis2_score_vs_length.png` — dual-axis bar: avg score vs avg normalized length per model
- `analysis2_correct_vs_wrong_len.png` — grouped bar: correct vs wrong response length
- `analysis4_steps_per_model.png` — steps prevalence
- `analysis4_steps_score_bar.png` — steps vs score
- `analysis4_selfcorr_per_model.png` — self-correction prevalence per model
- `analysis4_selfcorr_per_subtask.png` — self-correction prevalence per subtask
- `analysis4_selfcorr_score.png` — self-correction vs score
- `analysis4_hedging_per_model.png` — hedging prevalence
- `analysis4_hedging_score.png` — hedging vs score

---

## Key findings summary

| Topic | Finding |
|---|---|
| **Language switching** | Only Llama-4-Maverick systematically switches to English (155/1350 items = 11.5%). DeepSeek has 6 switches (Chinese/Komnzo), Mistral has 16 (Estonian/Komnzo). Gemini, Claude, GPT-4.1, CommandA never genuinely switch. |
| **Where switching happens** | Fill-in-blanks is the dominant trigger (134/177 genuine switches). Non-Latin-script languages (Japanese, Korean) most affected. English and Spanish: zero switches across all models. |
| **Subtask difficulty** | Fill-in-blanks (12.0) and mapping (14.2) are hardest on exact match. Classification (28.1) easiest. Translation scores highest (33.2) but metric is different (ChrF). Editing has 1 item — not interpretable. |
| **Model rankings by subtask** | Gemini dominates all subtasks. Claude 4 is #2 overall except fill-in-blanks (DeepSeek edges ahead). GPT-4.1 swings from #7 mapping to #3 translation. |
| **Length vs score** | No strong positive correlation. Llama writes 2× as much as CommandA but scores only slightly higher. Correct vs wrong responses have near-identical lengths (ratio ~0.93–1.06). |
| **Reasoning structure** | Features coded: steps (numbered structure), self-correction (backtracking), hedging (uncertainty), no-reasoning (immediate answer). Quote-and-think v2 (citing ≥3 IOL tokens in quotes) ranges from 36% (CommandA) to 67% (Llama). |

---

## Design decisions and notes (from markdown cells)

- **Score recomputation:** exact match is case-insensitive; chrF uses sacrebleu defaults. Small discrepancies from paper (max 7 pts) are expected because paper's exact pipeline is not published.
- **IOL stripping:** Unicode allowlist approach. Known limitation: some Yanyuwa/Hadza words are pure ASCII and survive stripping, causing false positive detections. The stripping successfully cleaned up most Koryak/Komnzo/Dâw characters.
- **Gemini nulls are not a problem:** if a row has no visible reasoning, there is nothing to analyse — it is correctly excluded from language analysis.
- **Token counts not used for length:** Gemini's `tokens_out` is inflated (includes internal thinking) and not comparable to other models' `tokens_out` (visible output only). Character length of `raw_` is used throughout Analysis 2.
- **Mistral/DeepSeek English switches:** concentrated in fill-in-blanks/Komnzo, which involves a family tree disambiguation task requiring complex multi-step reasoning. The model apparently falls back to its dominant pretraining language (English) when facing a cognitively demanding task in a less-trained language (Estonian for Mistral, Chinese for DeepSeek).
- **Llama English switches:** concentrated in fill-in-blanks and translation, for non-Latin-script languages. Pattern suggests Llama's instruction-following for non-Latin languages is weaker — it defaults to English when the task is complex and the prompt language script is distant from Latin.
