# Repetitiveness Analysis — Implementation Spec (v2)

## Context

Working notebook: `analysis_lr.ipynb` (currently 127 columns, df shape (1350, 127)).

This is **Analysis 5** in the WMT25 MIST Linguistic Reasoning project. It extends the existing analysis pipeline (Sections 0–6 + Analyses 1–4) with a new measurement: **how much does each model repeat itself within a single response?**

The motivating question: Llama 4 Maverick writes ~2.7× longer responses than CommandA (2726 vs 933 normalized chars on average) but scores only marginally better (22.9 vs 19.8). Hypothesis: a substantial part of Llama's extra length is internal repetition / boilerplate, not useful reasoning. This analysis tests that hypothesis with quantitative metrics.

## Reference

Shaib et al. 2024, "Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores."

- Paper: https://arxiv.org/html/2403.00553v3
- Python package: https://pypi.org/project/diversity/
- GitHub: https://github.com/cshaib/diversity

Install: `pip install diversity`

## Scope of this analysis

This is **intra-response repetitiveness only** — we measure how much one response repeats itself internally. Inter-response diversity (do all of a model's responses look alike?) is a separate analysis that will come next.

The same 7 models as previous analyses: Gemini-2.5-Pro, Claude-4, DeepSeek-V3, GPT-4.1, Llama-4-Maverick, CommandA, Mistral-Medium.

## Metrics to compute

Only two metrics are appropriate for intra-response analysis. Other Shaib metrics (Self-BLEU, homogenization scores, NGD, MATTR, HD-D, CR:POS) are designed for *sets of texts* and don't apply here — keep them in mind for the upcoming diversity analysis instead.

### 1. Compression Ratio (CR)

For each row × model, compute:

```python
from diversity import compression_ratio
cr = compression_ratio([raw_text], method='gzip')
```

Pass the response as a single-element list. Higher CR = more internal redundancy.

New column: `compression_ratio_{stem}` (float, 7 new columns total).

### 2. Self-Repetition Score

For each row × model, split the response into sentences, then compute:

```python
from diversity import self_repetition_score

sentences = split_sentences(raw_text, language)
if len(sentences) >= 2:
    srs = self_repetition_score(sentences)
else:
    srs = None  # not interpretable on a single sentence
```

Self-Repetition Score counts how many 4-grams from a given sentence reappear in *other* sentences of the same response. Higher SRS = the model reuses long phrases across sentences within one answer.

New columns:
- `self_repetition_{stem}` (float, NaN-able, 7 columns)
- `n_sentences_{stem}` (int, 7 columns) — needed for the bucket analysis

## Sentence segmentation strategy

NLTK's `sent_tokenize` is English-trained but works reasonably on Latin-script languages. It fails on CJK (Japanese, Chinese, Korean) because those languages use different punctuation characters (`。`, `？`, `！` instead of `.`, `?`, `!`).

We use a hybrid approach:

```python
import re
import nltk
nltk.download('punkt_tab', quiet=True)

CJK_LANGS = {"Japanese", "Chinese", "Korean"}

def split_sentences(text, language):
    if text is None or not text.strip():
        return []
    if language in CJK_LANGS:
        # Regex split on both ASCII and CJK sentence-end punctuation
        sentences = re.split(r'[.。!?！？\n]+', text)
    else:
        sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]
```

### Validation step before running the full analysis (mandatory)

Before computing SRS over all 9450 responses, test `split_sentences` on a small sample to confirm it actually segments. Run this BEFORE Cell 4:

```python
# Validation cell — run before full SRS computation
test_languages = ["English", "French", "Chinese", "Japanese", "Korean", "Estonian", "Russian"]
for lang in test_languages:
    sample_rows = df[df["problem_language"] == lang].head(3)
    print(f"=== {lang} ===")
    for idx, row in sample_rows.iterrows():
        text = row["raw_Claude-4"]  # any model with full coverage
        if not text:
            continue
        sentences = split_sentences(text, lang)
        print(f"  Row {idx}: {len(text)} chars → {len(sentences)} sentences")
        for i, s in enumerate(sentences[:2]):
            print(f"    [{i}] {s[:80]}...")
    print()
```

**Pass criteria**: at least 2 sentences detected for non-trivial responses (>200 chars) in every language tested.

**If any language fails (returns only 1 sentence for clearly multi-sentence text):**

1. Inspect 2–3 sample responses from the failing language to see what punctuation or structure the model actually uses there.
2. Update the regex once — add the missing punctuation characters or adjust the split pattern based on what you observed.
3. Re-run the validation block with the updated regex.

**If the second attempt still fails for any language: stop and report back to the user.** Do not install additional libraries (spacy, etc.) and do not keep iterating the regex on your own. Report:

- Which languages still fail after the regex update
- 2–3 sample responses for each (first ~500 chars)
- Both the original regex and your updated version

Then wait for the user. Do not proceed to Cell 4 until segmentation is approved.

## Length is a confounder — handle it explicitly

Both CR and SRS depend on text length:

- **CR**: gzip needs material to detect redundancy. A 50-char response cannot have a high CR even if it is pure repetition — there's nothing for gzip to build a dictionary on. CR grows mechanically with text length.
- **SRS**: more sentences → more chances of mechanical 4-gram overlap. Bucket SRS by sentence count, not char count.

Use **raw character length** (`char_len_{stem}`), not the FLORES-normalized length (`norm_len_{stem}`). Gzip operates on bytes, so what matters is the actual size of the text fed to it. The FLORES normalization corrects for cross-language content density, which is a different question.

The composition of items per model is identical (same 1350 prompts), so comparing CR averages across models is internally consistent — every model has the same mix of languages. No need to stratify by language for cross-model comparison.

## Implementation steps

Place new cells **after cell 25** in `analysis_lr.ipynb`. Follow the existing notebook conventions (use `STEMS`, `TOP_MODELS`, `df` as defined earlier).

### Cell 1 — Install and import

```python
# pip install diversity (one-time; comment out after first run)
from diversity import compression_ratio, self_repetition_score
import nltk, re
nltk.download('punkt_tab', quiet=True)
```

### Cell 2 — Define `split_sentences` and validate

Implement `split_sentences` as defined above. Run the validation block on the 7 test languages. **Inspect the output manually** before proceeding. If any CJK language returns only 1 sentence for a clearly multi-sentence response, stop and refine.

### Cell 3 — Compute CR per item per model

For each stem in STEMS, compute `compression_ratio_{stem}` over all 1350 rows. Handle edge cases:

- Empty/None responses → CR = NaN
- Very short responses (<20 chars) → still compute, but flag in viz

Verify df.shape after: should be (1350, 134).

### Cell 4 — Compute SRS + sentence count per item per model

For each stem in STEMS, compute `self_repetition_{stem}` AND `n_sentences_{stem}`:

- Use `split_sentences(raw_{stem}, problem_language)` for segmentation
- Store the sentence count in `n_sentences_{stem}` (int)
- If `n_sentences < 2` → set `self_repetition_{stem}` to NaN
- Otherwise call `self_repetition_score(sentences)`

Verify df.shape after: should be (1350, 148) — that's 127 + 7 (CR) + 7 (SRS) + 7 (n_sentences).

### Cell 5 — Sanity check distributions

Print descriptive stats per model:

```
Model            n_items  CR_mean  CR_median  SRS_mean  SRS_median  n_sent_mean  char_len_mean  SRS_NaN_count
Gemini-2.5-Pro    1350     ?.??     ?.??       ?.??      ?.??        ?.??         ?              ?
Claude-4          1350     ?.??     ?.??       ?.??      ?.??        ?.??         ?              ?
...
```

Look for: do CR and char_len move together within a model? Does Llama have visibly higher CR than other models? How many items per model have NaN SRS (single-sentence)?

---

### CR visualizations (block 1)

### Cell 6 — Figure 1a: scatter CR vs char_len, grid by model

A 2×4 grid (7 models, last cell empty). For each model:

- x-axis: `char_len_{stem}` (clip at 6000 chars; let outliers stay but don't drive the axis)
- y-axis: `compression_ratio_{stem}`
- Alpha 0.3 for points
- Title: model display name + n items
- Same x and y axis range across all subplots so they're comparable at a glance

Diagnostic figure. Shows the length confounder shape per model. Save to `figures/analysis5_cr_scatter_grid.png`.

### Cell 7 — Figure 1b: smoothed curves CR vs char_len, overlaid

Single axis. For each model:

- Sort items by char_len
- Compute a rolling mean of CR over a window of ~100 items
- Plot the smoothed curve, one color per model, with a legend
- Same x-axis as Figure 1a (clip at 6000)
- y-axis auto-scaled

**This is the main comparison figure for CR.** A model whose curve sits above the others at equivalent length = more boilerplate. Save to `figures/analysis5_cr_curves.png`.

### Cell 8 — Figure 2: CR by length bucket, grouped bar

Define three buckets on `char_len`:

- short: <800 chars
- medium: 800–1500 chars
- long: >1500 chars

Build a table:

| Model | CR short | CR medium | CR long | n short | n medium | n long |

Plot as a grouped bar chart (3 bars per model, 7 model groups, error bars = SE).

Save to `figures/analysis5_cr_buckets.png`.

If a model has higher CR than peers in *all three buckets*, that's a clean signal of boilerplate independent of length.

### Cell 9 — Correlation: CR vs score, stratified

For each model, compute Spearman correlation between `compression_ratio_{stem}` and `score_{stem}`:

1. Overall (all 1350 items)
2. Within each length bucket (short / medium / long)

Output:

| Model | Spearman overall | Spearman short | Spearman medium | Spearman long |

Negative correlation across all three buckets = robust signal that internal repetition predicts lower score.

---

### SRS visualizations (block 2)

Mirror block 1 for SRS, with sentence count as the x-axis instead of char_len. SRS captures long-n-gram recycling between sentences — a different signal from CR (which is all redundancy).

### Cell 10 — Figure 3a: scatter SRS vs n_sentences, grid by model

Same 2×4 layout as Figure 1a. x = `n_sentences_{stem}`, y = `self_repetition_{stem}`. Exclude NaN SRS (single-sentence responses) by `.dropna()`.

Save to `figures/analysis5_srs_scatter_grid.png`.

### Cell 11 — Figure 3b: smoothed curves SRS vs n_sentences, overlaid

Same approach as Figure 1b. Rolling mean over n_sentences-sorted points, one curve per model.

Save to `figures/analysis5_srs_curves.png`.

### Cell 12 — Figure 4: SRS by sentence-count bucket, grouped bar

Buckets on `n_sentences`:

- short: <5 sentences
- medium: 5–15 sentences
- long: >15 sentences

Grouped bar chart, same structure as Figure 2.

Save to `figures/analysis5_srs_buckets.png`.

### Cell 13 — Correlation: SRS vs score, stratified

Same as Cell 9 but for SRS, stratified by sentence-count bucket.

---

### Test of the Llama hypothesis

### Cell 14 — Figure 5: useful_len vs score

Define `useful_len_{stem} = char_len_{stem} / compression_ratio_{stem}`. This approximates the size the response would have after gzip — a proxy for non-redundant content.

Replot the existing score vs length chart (from Analysis 2) using `useful_len` instead of `char_len`. One point per model, x = mean useful_len, y = mean score.

Compare to the original chart:
- If Llama's mean useful_len drops closer to CommandA's while its char_len stays much higher → confirmed: most of Llama's extra length is redundancy, not reasoning.
- If useful_len ranking matches char_len ranking → repetition is not the explanation.

Save to `figures/analysis5_useful_len_vs_score.png`.

---

### Cell 15 — Markdown summary cell

Write a concise summary block, Rita's preferred style (punchy bullets, first-person where appropriate):

- Which model has highest CR overall, and whether the bucket analysis confirms it
- Same for SRS — does it confirm or contradict CR?
- Whether the Llama length paradox is explained by repetition (Figure 5 result)
- Whether CR / SRS correlate negatively with score (and in how many buckets)
- Known limitations: sentence segmentation reliability (note the CJK fallback), NaN handling for single-sentence responses, length confounder always present

## Edge cases and gotchas

- **Empty responses**: some Gemini rows have no visible reasoning (251 nulls in Analysis 1). For these, both CR and SRS are NaN. Don't drop the rows from df — just exclude them from the relevant aggregations with `.dropna()` or `.notna()` masks at plot time.
- **Single-sentence responses**: many GPT-4.1 and Mistral responses are just `[answer]` with no preamble. SRS is NaN by design.
- **Don't compute CR on the bracket-only part**: feed `raw_{stem}` as-is, not the extracted answer. We want repetition in the reasoning, not in the final answer.
- **NLTK on bracket text**: if `nltk.sent_tokenize` returns the bracket as its own "sentence", consider stripping bracket content before tokenizing (or use `segment_response` from Analysis 1 which already does this).

## Deliverables

After running:

1. df shape = (1350, 148) — 21 new columns added (7 CR + 7 SRS + 7 n_sentences)
2. Seven new figures in `figures/` (1a, 1b, 2, 3a, 3b, 4, 5)
3. Multiple summary tables printed in the notebook (descriptives, bucket means for CR and SRS, correlation tables)
4. One markdown summary cell with findings

Stop after Cell 15. Do not start the diversity (inter-response) analysis in the same session — that is a separate spec.
