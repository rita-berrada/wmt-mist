# Claude Code Prompt: Task-Type Decomposition Analysis

In `analysis_lr.ipynb`, add a new section after cell 60 (the last Analysis 1 cell) for **Research Direction 1: Task-Type Decomposition inside LR**.

---

## Notebook conventions (follow exactly)

- Section headers: markdown cell starting with `---` then `## Section Title`
- Code cell comments: `# ── Description ───────────────────` (wide dash line)
- One computation per code cell, markdown insight cells between code cells for me to fill in
- Use `from IPython.display import display` when showing DataFrames
- Heatmaps use `sns.heatmap` with `annot=True, fmt=".1f"`, `cmap="YlOrRd"`, `linewidths=0.4`
- Print-based exploration uses `print("=== Title ===")` style
- Figures use `fig, ax = plt.subplots(figsize=(W, H))` pattern
- `sns.set_theme(style="whitegrid", palette="colorblind")` and `plt.rcParams.update({"figure.dpi": 120})` are already set in cell 2

---

## Existing variables available (do NOT redefine)

```python
# From cell 3:
STEMS = ['Gemini-2.5-Pro', 'Claude-4', 'DeepSeek-V3', 'GPT-4.1', 'Llama-4-Maverick', 'CommandA', 'Mistral-Medium']
DISPLAY = ['Gemini 2.5 Pro', 'Claude 4', 'DeepSeek V3', 'GPT 4.1', 'Llama 4 Maverick', 'CommandA', 'Mistral Medium']
TOP_MODELS = {stem: display_name for stem, display_name in zip(STEMS, DISPLAY)}

# From cell 18 — master DataFrame `df` with 1,350 rows (90 puzzles × 15 languages)
# Base columns:
#   N, id, points, instruction_language, problem_language, type, eval_type, answer
# Per-model columns (for each stem in STEMS):
#   score_{stem}    — weighted score: (chrF or exact_match) × points
#   tokens_out_{stem}, tokens_think_{stem}
#   bracket_{stem}  — bool: did the response contain [...]?
#   pred_{stem}     — extracted answer text
#   raw_{stem}      — full raw response text

# Task types in df['type']: classification, editing, fill-in-blanks, mapping, translation
# df['eval_type']: 'chrF' for translation, 'exact_match' for the other 4
# df['problem_language']: Koryak, Hadza, Komnzo, Dâw, Yanyuwa (the IOL puzzle languages)
# df['instruction_language']: 15 prompt languages (English, French, German, Spanish, etc.)
# df['points']: weight per item (0.5, 1.0, 1.5, 2.0, 2.5) — sums to 20 per task, 100 per language
# df['id']: puzzle identifier like '4:a5' — same puzzle across 15 instruction languages

# Also available: df_lr (gold metadata), df_agg_top (pre-computed language-level scores)
```

---

## Cells to create (in this exact order)

### Cell A — Markdown: Section intro
```
---
## Research Direction 1 — Task-Type Decomposition

**Question:** Which LR task types are most difficult? Do model rankings change across task types? 

**Context:** The 90 LR items per language break down into 5 task types: classification (4), editing (1), fill-in-blanks (20), mapping (24), translation (41). Translation is scored with chrF (continuous 0–1, partial credit) while the other four use exact match (binary 0/1). This scoring difference means we need to compare rankings both with and without translation.

**Connection to Analysis 1:** Language-switching in Analysis 1 was heavily concentrated in fill-in-blanks (all 3 models that switched did so primarily on fill-in-blanks tasks about Komnzo). This decomposition will show whether fill-in-blanks is also the hardest task type by score.

**Caveat on sample sizes:** Classification has only 4 items per language (60 total per model) and editing has just 1 item per language (15 total per model). Findings for these two types should be interpreted cautiously.
```

### Cell B — Code: Task-type structure overview
Show for each task type:
- Number of unique puzzles (unique `id` values)
- Total items (rows in df)
- Sum of `points` per language (group by type, take one language, sum points)
- Eval method
- Proportion of total points

This establishes the structural facts before analysis.

### Cell C — Code: Raw accuracy vs weighted score (disentangling difficulty from point weighting)
For each model × task type, compute TWO metrics:
1. **Raw accuracy**: proportion of items where the model scored > 0 (i.e. got some credit). For exact_match tasks this is just mean accuracy. For chrF tasks use mean chrF score (the score before multiplying by points).
2. **Weighted mean score**: mean of `score_{stem}` (what we already have — includes point weighting).

Display both as side-by-side heatmaps (models as rows, task types as columns). This separates "this task type is hard" from "this task type is worth more/fewer points."

To compute raw accuracy for chrF items: `score_{stem} / points` gives the chrF score before weighting. For exact_match items: `score_{stem} / points` gives 0 or 1.

### Cell D — Markdown: Insight placeholder
```
### Insight — accuracy vs weighted score
**Key finding:** *(fill in after running)*
```

### Cell E — Code: Model rankings per task type
For each task type, rank the 7 models by their mean `score_{stem}`. Display as a DataFrame:
- Columns = task types + "Overall"
- Rows = models (use DISPLAY names)
- Values = rank (1 = best)
- Overall = rank by mean score across all items

After the table, print which models have a rank shift ≥ 2 positions between any task type and their overall rank. This identifies where models are disproportionately strong or weak.

### Cell F — Code: With vs without translation comparison
Compute two overall rankings:
1. Including all 5 task types (mean of `score_{stem}` across all rows)
2. Excluding translation (mean of `score_{stem}` where `type != 'translation'`)

Show side by side. Highlight any rank changes. This matters because translation uses chrF (partial credit) which could inflate or deflate certain models' standings.

### Cell G — Markdown: Insight placeholder
```
### Insight — ranking stability
**Key finding:** *(fill in after running)*
```

### Cell H — Code: Score distribution by task type (box plots)
Create a figure with one row of box plots per model (7 subplots stacked vertically or in a 2×4 grid). Each subplot shows box plots of item-level scores for the 5 task types. This shows variance, not just means — are some task types high-variance (some items easy, some impossible)?

Use `fig, axes = plt.subplots(2, 4, figsize=(20, 10))` — leave the 8th subplot empty or use it for a combined view.

### Cell I — Code: Problem language × task type interaction
The 90 puzzles come from 5 IOL source languages (Koryak, Hadza, Komnzo, Dâw, Yanyuwa). Different puzzle languages might be harder for different task types.

For each model, compute mean score grouped by `(problem_language, type)`. Display a heatmap for the best model (Gemini-2.5-Pro) showing problem_language (rows) × type (columns).

Then compute the same for all models averaged, to see if certain problem_language × task_type combinations are universally hard.

### Cell J — Markdown: Insight placeholder
```
### Insight — problem language effects
**Key finding:** *(fill in after running)*

**Connection to Analysis 1:** Komnzo fill-in-blanks was where language switching concentrated. Is Komnzo fill-in-blanks also the lowest-scoring combination here?
```

### Cell K — Code: Per-puzzle difficulty analysis
Group by `id` (puzzle identifier) and compute mean score across all 15 instruction languages, for each model. Then average across models to get a "universal difficulty" score per puzzle.

Show:
1. The 10 hardest puzzles (lowest mean score across all models) with their type and problem_language
2. The 10 easiest puzzles
3. A scatter plot: x = mean score across models, y = standard deviation across models. Color by task type. This shows which puzzles are universally hard vs which are "divisive" (some models solve them, others don't).

### Cell L — Code: Task type × instruction language interaction
For the top model (Gemini-2.5-Pro), create a heatmap: instruction_language (rows) × type (columns), values = mean score. This shows if certain prompt languages make certain task types harder.

Then do the same averaged across all 7 models.

### Cell M — Markdown: Section summary placeholder
```
### Task-Type Decomposition — Summary

**Key findings:** *(fill in after running)*

1. Hardest/easiest task types: ...
2. Ranking shifts: ...
3. With/without translation: ...
4. Problem language effects: ...
5. Connection to Analysis 1 (fill-in-blanks): ...
```

---

## Important constraints

- Do NOT modify or recompute any existing cells (cells 0–60).
- Do NOT redefine STEMS, DISPLAY, TOP_MODELS, df, or df_lr.
- All new cells go after cell 60.
- Each code cell should be independently runnable (given that all previous cells have been run).
- Use `display()` for DataFrames, `plt.show()` for figures.
- Add `plt.tight_layout()` before `plt.show()` for multi-panel figures.
