# Diversity Analysis — Implementation Spec

## Context

Working notebook: `analysis_lr.ipynb`.

This is **Analysis 6** in the WMT25 MIST Linguistic Reasoning project. It complements Analysis 5 (repetitiveness, which measured how much a model repeats itself *within* a single response). This analysis measures the **opposite axis**: how much a model varies its writing *between* responses.

## Research questions (from Julia)

1. **How uniform does a model approach a problem?** Globally, is its writing style consistent across all responses or highly varied?
2. **Is it able to shift its style, words, structure from one task to another or is it the same?** Does the model adapt between task types (translation, mapping, fill-in-blanks, classification), or apply the same template everywhere?
3. **How is it related to score: more diversity = better score?**

## Reference

Shaib et al. 2024, "Standardizing the Measurement of Text Diversity."

- Paper: https://arxiv.org/html/2403.00553v3
- Python package: https://pypi.org/project/diversity/

Already installed in Analysis 5.

## Scope

The same 7 models as previous analyses: Gemini-2.5-Pro, Claude-4, DeepSeek-V3, GPT-4.1, Llama-4-Maverick, CommandA, Mistral-Medium.

Three analyses, building from broad to fine:

| Analysis | Set analyzed | Question |
|---|---|---|
| **A. Global** | All ~1350 responses of one model | Who is globally most uniform in their writing? |
| **B. Per task type** | Responses of one model for one task type | In which task type is each model most uniform? |
| **C. Cross-type** | Pairwise comparison between task type A and B for one model | Between which task types does each model write the same vs differently? |

## Metrics used

Three intra-set metrics (used in Analyses A and B) + one cross-set metric (used in Analysis C only).

| Metric | What it captures | Used in |
|---|---|---|
| **CR (Compression Ratio)** | Global n-gram redundancy in the concatenated set | A, B |
| **SRS (Self-Repetition Score)** | 4-grams recycled between texts of the set | A, B |
| **Self-BLEU** | Pairwise similarity between texts of one set | A, B |
| **CR:POS (Compression Ratio on POS tags)** | Syntactic template redundancy (same grammatical skeletons across texts) | A only, English-prompt subset |
| **Cross-BLEU** | Similarity between texts of set A and texts of set B | C only |

**Note on CR:POS**: this metric detects *syntactic templates* — recurring grammatical skeletons even when the surface words vary. It's the metric the Shaib paper found most discriminating between human and machine text. We run it only on the English-prompt subset (~90 items per model where `problem_language == "English"`) because NLTK's POS tagger is English-only. This gives a complementary signal to CR/SRS/Self-BLEU that may reveal boilerplate hidden at the syntactic level.

### Why Cross-BLEU and how it's defined

Cross-BLEU is a straightforward extension of Self-BLEU. Self-BLEU takes each text of a set as a candidate and computes its BLEU against all other texts of the same set as references. Cross-BLEU(A, B) takes each text of set A as candidate and computes BLEU against all texts of set B as references:

```
Cross-BLEU(A, B) = mean over a in A of BLEU(candidate=a, references=B)
Cross-BLEU(B, A) = mean over b in B of BLEU(candidate=b, references=A)
Cross-BLEU_symmetric(A, B) = (Cross-BLEU(A, B) + Cross-BLEU(B, A)) / 2
```

This metric ignores intra-A and intra-B diversity entirely — it measures only similarity *between* the two sets. The interpretation:

- High Cross-BLEU(A, B) ≈ Self-BLEU(A) ≈ Self-BLEU(B): texts of A look like texts of B → the model writes the same way in both task types
- Low Cross-BLEU(A, B) compared to Self-BLEU(A) and Self-BLEU(B): texts of A and B share fewer n-grams → the model changes style between the two types

The `diversity` package does not have a built-in Cross-BLEU function. Implement it by reusing the BLEU computation from `sacrebleu` or `nltk.translate.bleu_score` (already in the env from sacrebleu).

## Technical choices

- **Text field**: use `segment_response_{stem}` (reasoning without the final bracket answer). Empty `segment_response` values (model responded with just `[D]`) are excluded — they have no material to analyze anyway.
- **Languages**: all 15 languages kept. The composition per language is identical across the 7 models, so cross-model comparisons remain valid even if the language mix introduces noise into absolute values.
- **Subsampling for BLEU metrics**: Self-BLEU and Cross-BLEU are pairwise — costly on large sets. Cap each set at **150 randomly sampled texts** for these two metrics. Set a fixed random seed (e.g., `random_state=42`) for reproducibility. CR and SRS are fast — run them on the full set.

## Implementation steps

Place new cells **after the repetitiveness analysis (Analysis 5) cells**. Follow existing notebook conventions (use `STEMS`, `TOP_MODELS`, `df` as defined earlier).

### Cell 1 — Imports and helpers

```python
from diversity import compression_ratio, self_repetition_score
import sacrebleu
import random
import numpy as np
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

random.seed(42)
np.random.seed(42)

SAMPLE_SIZE_BLEU = 150

def safe_texts(series):
    """Filter out None, empty, and whitespace-only strings."""
    return [t for t in series if t is not None and t.strip()]

def subsample(texts, n=SAMPLE_SIZE_BLEU, seed=42):
    """Reproducibly subsample texts. Returns texts unchanged if smaller than n."""
    if len(texts) <= n:
        return texts
    rng = random.Random(seed)
    return rng.sample(texts, n)

def text_to_pos_sequence(text):
    """Tokenize text and replace each token with its POS tag. Returns a string of tags."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return " ".join(tag for _, tag in tagged)

def cr_pos(texts):
    """Compression ratio on POS tag sequences. Captures syntactic template redundancy."""
    if not texts:
        return None
    pos_sequences = [text_to_pos_sequence(t) for t in texts if t.strip()]
    if not pos_sequences:
        return None
    return compression_ratio(pos_sequences, method='gzip')
```

### Cell 2 — Define Self-BLEU and Cross-BLEU wrappers

```python
def self_bleu(texts, sample_size=SAMPLE_SIZE_BLEU):
    """Mean of BLEU(candidate=t, references=others) over all t in texts."""
    texts = subsample(texts, sample_size)
    if len(texts) < 2:
        return None
    scores = []
    for i, candidate in enumerate(texts):
        references = [texts[j] for j in range(len(texts)) if j != i]
        # sacrebleu expects a list of reference strings per candidate
        bleu = sacrebleu.sentence_bleu(candidate, references)
        scores.append(bleu.score)
    return sum(scores) / len(scores)

def cross_bleu(texts_a, texts_b, sample_size=SAMPLE_SIZE_BLEU, symmetric=True):
    """
    Cross-BLEU(A, B): each candidate from A scored against references from B.
    If symmetric, return mean of Cross-BLEU(A, B) and Cross-BLEU(B, A).
    """
    a = subsample(texts_a, sample_size)
    b = subsample(texts_b, sample_size)
    if not a or not b:
        return None
    def directional(cands, refs):
        scores = [sacrebleu.sentence_bleu(c, refs).score for c in cands]
        return sum(scores) / len(scores)
    ab = directional(a, b)
    if not symmetric:
        return ab
    ba = directional(b, a)
    return (ab + ba) / 2
```

Note on speed: sacrebleu.sentence_bleu is reasonably fast but on 150 candidates × 150 refs per call, total per metric call is ~22,500 BLEU computations. For 7 models, Analysis B is 7×4 = 28 calls of Self-BLEU; Analysis C is 7×6 = 42 calls of Cross-BLEU. Plan for ~5–10 min total runtime for the BLEU metrics. Print progress.

### Cell 3 — Analysis A: Global diversity per model

For each model in TOP_MODELS:

- Get the list of `segment_response_{stem}` values, filter to non-empty
- Compute CR on the concatenated set: `compression_ratio(texts, method='gzip')`
- Compute SRS: `self_repetition_score(texts)`
- Compute Self-BLEU: `self_bleu(texts)` with subsampling
- **CR:POS (English-prompt subset only)**: filter `df` to rows where `problem_language == "English"`, get `segment_response_{stem}` values, filter non-empty, then call `cr_pos(english_texts)`. Store the number of English texts used as `n_english`.

Store in `df_div_global` (DataFrame):

| Model | n_texts | n_english | CR | SRS | Self_BLEU | CR_POS_en |
|---|---|---|---|---|---|---|
| Gemini-2.5-Pro | ? | ? | ?.?? | ?.?? | ?.?? | ?.?? |
| ... | | | | | | |

Print the table. Note for the user: **higher value = less diverse** in all 4 metrics. CR_POS_en is computed only on the ~90 English-prompt items per model (NLTK's POS tagger is English-only); the other 3 metrics use all available languages.

### Cell 4 — Figure 1: Global diversity bar chart

4-panel bar chart (2×2), one panel per metric (CR / SRS / Self-BLEU / CR:POS English-only). 7 bars per panel.

Title each panel with the metric name + a one-line reminder of interpretation ("higher = less diverse"). For the CR:POS panel, add "English-prompt subset only" in the title to make the subset explicit.

Save to `figures/analysis6_global_diversity.png`.

### Cell 5 — Analysis B: Per-task-type diversity

For each (model, task_type) pair:

- Get `segment_response_{stem}` values for items where `task_type == X`
- Filter to non-empty
- Compute CR, SRS, Self-BLEU on this subset

Task types to use: extract from `df["task_type"]` — should be `{translation, mapping, fill-in-blanks, classification}`. Skip `editing` (only 1 item, not enough).

Store results in `df_div_by_type` (long-form DataFrame):

| Model | task_type | n_texts | CR | SRS | Self_BLEU |
|---|---|---|---|---|---|
| Gemini-2.5-Pro | translation | ? | ?.?? | ?.?? | ?.?? |
| Gemini-2.5-Pro | mapping | ? | ?.?? | ?.?? | ?.?? |
| ... | | | | | |

7 models × 4 task types = 28 rows.

### Cell 6 — Figure 2: Per-task-type diversity heatmaps

3 heatmaps, one per metric. Each heatmap: rows = 7 models, columns = 4 task types, values = the metric.

- Annotate cells with values (2 decimals)
- Same colormap orientation across panels (higher = less diverse → use a sequential colormap where dark = high)
- Sort rows by global Self-BLEU (most uniform model at top) so reading order is consistent

Save to `figures/analysis6_by_task_type.png`.

**Reading guide for the markdown summary**: look at horizontal variance per row. A row with similar values across all 4 columns = a model that writes uniformly regardless of task type. A row with high variance = a model that adapts.

### Cell 7 — Analysis C: Cross-type pairwise Cross-BLEU

For each model:

- Build the 4 task-type subsets: `texts_translation`, `texts_mapping`, `texts_fillinblanks`, `texts_classification` (using `segment_response_{stem}`, non-empty filter)
- Compute Self-BLEU on each subset (diagonal of the matrix)
- For each pair of distinct types (6 pairs: trans-map, trans-fib, trans-class, map-fib, map-class, fib-class), compute symmetric Cross-BLEU

Store in `df_div_cross_type` (long-form):

| Model | type_A | type_B | metric | value |
|---|---|---|---|---|
| Gemini-2.5-Pro | translation | mapping | cross_bleu | ?.?? |
| Gemini-2.5-Pro | translation | translation | self_bleu | ?.?? |
| ... | | | | |

For each model, 4 self-BLEU values (diagonal) + 6 cross-BLEU values = 10 values. Total 70 rows.

### Cell 8 — Figure 3: Per-model 4×4 matrices

7 subplots (one per model), each is a 4×4 matrix.

- Rows and columns: 4 task types in fixed order (translation, mapping, fill-in-blanks, classification)
- Diagonal cells: Self-BLEU of that type (intra-type uniformity)
- Off-diagonal cells: Cross-BLEU between type A and type B
- Matrix is symmetric by construction
- Annotate each cell with the value
- Same color scale across all 7 matrices for visual comparability

Save to `figures/analysis6_cross_type_matrices.png`.

**Reading guide**: for a given model, compare a cell (A, B) to the diagonal cells (A, A) and (B, B). If (A, B) ≈ (A, A) ≈ (B, B), the model writes types A and B in the same style. If (A, B) is markedly lower, the model genuinely shifts style between A and B.

### Cell 9 — Figure 4: Diversity vs score (global level)

Scatter, 7 points (one per model):

- x: Self-BLEU from `df_div_global` (or use CR — make 2 versions)
- y: overall mean LR score for that model
- Annotate each point with the model name

Caveat in the title or annotation: n=7, low statistical power.

Save to `figures/analysis6_diversity_vs_score_global.png`.

### Cell 10 — Figure 5: Diversity vs score (per task type)

Scatter, 28 points (7 models × 4 task types):

- x: Self-BLEU from `df_div_by_type`
- y: mean LR score for that (model, task_type) pair (already computable from existing df)
- Color = model, marker shape = task type
- Legend showing both

Save to `figures/analysis6_diversity_vs_score_by_type.png`.

### Cell 11 — Markdown summary cell

Concise summary in Rita's preferred style (punchy bullets, first-person where appropriate). Cover:

- Globally (Analysis A), which model is most/least uniform across the 4 metrics. Specifically check whether CR:POS (English-only) reveals patterns invisible to CR/SRS — a model can have low CR (varied words) but high CR:POS (same syntactic templates), or vice versa.
- Per task type (Analysis B), which models show the strongest variance between task types (= biggest stylistic shift), which are most uniform across types
- Cross-type findings (Analysis C): which model × pair combinations show the largest gap between Self-BLEU diagonal and Cross-BLEU off-diagonal (= biggest style shift between those two specific types). Pick 2–3 most striking examples.
- Relationship with score: what the scatters show. Be cautious about claims given n=7 (global) and n=28 (per-type).
- Limitations: language mixing in sets (composition is identical across models, but absolute values may be inflated for non-Latin scripts), subsampling for BLEU metrics, n=7 statistical power, CR:POS computed only on the English-prompt subset (~90 items per model) so its sample is smaller and not directly comparable to CR/SRS/Self-BLEU which use the full 1350 items.

## Edge cases and gotchas

- **Empty `segment_response`**: many GPT-4.1 / Mistral rows have empty segment_response (response was just `[D]`). Filter these out before passing to any metric. Report the effective n_texts in each table so the user can see if some models have unusually small effective samples.
- **Task type with very few items**: classification has only 4 puzzles × 15 languages = 60 items per model max, possibly fewer after filtering empties. If a model has <20 effective texts for a given task type, set metrics to NaN for that cell rather than computing on too little data. Flag in the printed output.
- **BLEU on very long texts**: sacrebleu handles long inputs fine but truncate to first 2000 chars per text if any text exceeds that — this prevents pathological slowdowns on outlier-long responses without losing the main signal.
- **Reproducibility**: always set the random seed before any subsampling so re-runs produce the same numbers.

## Deliverables

After running:

1. Three new DataFrames: `df_div_global` (7 rows), `df_div_by_type` (28 rows), `df_div_cross_type` (70 rows)
2. Five new figures in `figures/`:
   - `analysis6_global_diversity.png`
   - `analysis6_by_task_type.png`
   - `analysis6_cross_type_matrices.png`
   - `analysis6_diversity_vs_score_global.png`
   - `analysis6_diversity_vs_score_by_type.png`
3. Printed tables for each analysis
4. One markdown summary cell with findings

Stop after Cell 11.
