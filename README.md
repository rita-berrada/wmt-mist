# Multilingual Instruction Shared Task

This repository contains data and evaluation code for the [WMT25 Multilingual Instruction Shared Task](https://www2.statmt.org/wmt25/multilingual-instruction.html).

> **Title:** Findings of the WMT25 Multilingual Instruction Shared Task: Persistent Hurdles in Reasoning, Generation, and Evaluation
> 
> **Abstract:** The WMT25 Multilingual Instruction Shared Task (MIST) introduces a benchmark to evaluate large language models (LLMs) across 30 languages. The benchmark covers five types of problems: machine translation, linguistic reasoning, open-ended generation, cross-lingual summarization, and LLM-as-a-judge. We provide automatic evaluation and collect human annotations, which highlight the limitations of automatic evaluation and allow further research into metric meta-evaluation. We run on our benchmark a diverse set of open- and closed-weight LLMs, providing a broad assessment of the multilingual capabilities of current LLMs. Results highlight substantial variation across sub-tasks and languages, revealing persistent challenges in reasoning, cross-lingual generation, and evaluation reliability. This work establishes a standardized framework for measuring future progress in multilingual LLM development.

## Data

The shared task was organized in two rounds:
In the first round, participants processed outputs for 4 tasks (translation, cross-lingual summarization, open-ended generation, linguistic reasoning).
In the second round, the participants evaluated the outputs of all other models as LLM-as-a-judge.

See `data/mist_25.json` for input prompts (blind) used in the MIST25 dataset (almost 20k prompts across many languages).

See `data/submissions*/` for raw model outputs:
- `data/submissions/` for submissions of the four tasks,
- `data/submissions_judge_oeg` for open-ended generation judge,
- `data/submissions_judge_xlsum` for cross-lingual summarization judge.

See `data/humeval_aggregated/` for aggregated human evaluation results:
- `data/humeval_aggregated/xsum.json` for cross-lingual summarization evaluated on Likert-7 across multiple dimensions,
- `data/humeval_aggregated/judge_xlsum.json` for LLM-as-a-judge of cross-lingual summarization results,
- `data/humeval_aggregated/oeg.json` for open-ended generation evaluated on Likert-7 across multiple dimensions,
- `data/humeval_aggregated/judge_xlsum.json` for LLM-as-a-judge of open-ended generation results,
- `data/humeval_aggregated/mt.json` for machine translation (taken from [WMT25 General Machine Translation Shared Task](https://github.com/wmt-conference/wmt25-general-mt)),
- `data/humeval_aggregated/judge_mt.json` for LLM-as-a-judge for machine translation (taken from [WMT25 Automated Translation Quality Evaluation Systems](https://github.com/wmt-conference/wmt25-general-mt)),
- `data/humeval_aggregated/mt.json` for linguistic reasoning task.

See `data/humeval/` for per-item evaluation results:
- `data/mt.json` for translation evaluation results,
- `data/xlsum.json` for cross-lingual summarization evaluation results,
- `data/oeg.json` for open-ended generation evaluation results,
- `data/lr.json` for linguistic reasoning evaluation results.

## Other

Cite as:
```bibtex
@InProceedings{kocmi-etal-2025-mist,
  title     = {Findings of the WMT25 Multilingual Instruction Shared Task: Persistent Hurdles in Reasoning, Generation, and Evaluation},
  author    = {Kocmi, Tom and Agrawal, Sweta and Artemova, Ekaterina and Avramidis, Eleftherios and Briakou, Eleftheria and Chen, Pinzhen and Fadaee, Marzieh and Freitag, Markus and Grundkiewicz, Roman and Hou, Yupeng and Koehn, Philipp and Kreutzer, Julia and Mansour, Saab and Perrella, Stefano and Proietti, Lorenzo and Riley, Parker and Sá¡nchez, Eduardo and Schmidtova, Patricia and Shmatova, Mariya and Zouhar, Vilém},
  booktitle = {Proceedings of the Tenth Conference on Machine Translation (WMT 2025)},
  month     = {November},
  year      = {2025},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  pages     = {462--483},   
  abstract  = {The WMT25 Multilingual Instruction Shared Task (MIST) introduces a benchmark to evaluate large language models (LLMs) across 30 languages. The benchmark covers five types of problems: machine translation, linguistic reasoning, open-ended generation, cross-lingual summarization, and LLM-as-a-judge. We provide automatic evaluation and collect human annotations, which highlight the limitations of automatic evaluation and allow further research into metric meta-evaluation. We run on our benchmark a diverse set of open- and closed-weight LLMs, providing a broad assessment of the multilingual capabilities of current LLMs. Results highlight substantial variation across sub-tasks and languages, revealing persistent challenges in reasoning, cross-lingual generation, and evaluation reliability. This work establishes a standardized framework for measuring future progress in multilingual LLM development.},
  url       = {https://aclanthology.org/2025.wmt-1.24}
}
```

The `tool/` contains work-in-progress tool (not production-ready).