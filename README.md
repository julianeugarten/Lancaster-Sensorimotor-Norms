# Lancaster-Sensorimotor-Norms

This repository applies the [Lancaster Sensorimotor Norms](https://www.lancaster.ac.uk/psychology/lsnorms/) (Lynott et al. 2019) to compare sensory language across published fiction, fanfiction, and AI-generated narrative text.

## Corpora

- **[Chicago Corpus](https://aclanthology.org/2024.lrec-main.71/)** (Bizzoni et al. 2024) — published fiction, 1880–2000.
- **[MythFic Corpus](https://data.ru.nl/collections/ru/rich/mythfic_metadata_dsc_550)** (Neugarten 2023) — Ancient Greek Religion and Lore (AGRL) fanfiction, plus five additional fandoms (HP, LOTR, MAG, PJ, RPF).
- **[SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories)** (Finke et al. 2025) — parameterized synthetic short stories, GPT-4o-mini.
- **[StoryScope](https://huggingface.co/datasets/jjrussell10/storyscope)** (Russell et al. 2026) — human-written prompts mirrored by five LLMs (Claude, GPT, Gemini, DeepSeek, Kimi).

## Repository structure

## Repository structure

```
Lancaster-Sensorimotor-Norms/
├── data/
│   ├── meta/                     # Chicago metadata (word counts, publication year, etc.) — not included, see below
│   ├── lemmatized_data/          # Lemmatized, scored text per corpus (regeneratable, gitignored)
│   ├── checkpoint/
│   │   ├── scores/               # Compact per-corpus sensory scores + metadata (git-tracked, ~70MB)
│   │   └── text_local/           # Full lemmatized token lists per corpus (large, gitignored, local only)
│   └── fanfiction_set_m.csv     # Raw fanfiction source data — not included, see below
├── notebooks/
│   ├── compile_data.py           # Loads, merges, and labels raw scored data; saves the checkpoint above
│   ├── main.py                   # Classification, robustness checks, coherence checks, figures
│   ├── sense_dict.py             # Lancaster norms lookup / sensory scoring utilities
│   ├── gen_latex.py              # Generates LaTeX tables from saved results JSONs
│   └── OUT_DIR/                  # Generated results (JSON/txt/figures), gitignored
├── resources/                    # Supplementary scripts (e.g. explicitness prediction)
├── src/                          # Lemmatization and scoring of data
├── figs/                         # Saved plots
├── requirements.txt
└── README.md
```



## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Reproducing the analysis

**Quick reproduction (scores only, no raw data needed):**

The `data/checkpoint/scores/` checkpoint is included in this repository. Running
```bash
python notebooks/main.py
```
directly will load it and reproduce all sensory-feature classification results, robustness checks, and coherence checks. The MFW stylometric baseline is skipped automatically in this case, since it requires the full lemmatized text (`text_local/`), which is not included due to size.

Set `MODE` in `main.py`'s config block to `"three_class"` (Published/Fanfiction/Generated) or `"four_class"` (Published/Fanfiction/SimpleStories/Storyscope).

