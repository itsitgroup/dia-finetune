# Dia Crying Finetuning

Fine‑tune **Dia‑1.6 B** so it can generate realistic *crying* vocalisations.  

The numbered scripts form a linear pipeline:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_prepare_manifest.py` | Scans the Nonverbal Vocalization corpus, resamples crying clips to 44.1 kHz, writes `crying/manifest.csv` |
| 2 | `02_encode_dac.py` | Runs Dia’s discrete‑audio‑codec over each wav, saving token sequences in `crying/dac_tokens/` |
| 3 | `03_finetune.py` | Fine‑tunes **only** the Dia decoder on the token/text pairs |
| 4 | `04_test.py` | Loads the checkpoint from step 3 and synthesises a short demo wav |
| 5 | `dataset.py` | Defines the CryingDataset class, which prepares and returns (text, audio, valid_len) training samples from the manifest and DAC token files for fine-tuning the Dia model. |

```bash
├── 01_prepare_manifest.py
├── 02_encode_dac.py
├── 03_finetune.py
├── 04_test.py
├── dataset.py
├── requirements.txt
├── dataset/        # includes the entire NonverbalVocalization folder
└── crying/              # auto‑generated
```

---

## Quick‑start

```bash
# clean virtual‑env
python -m venv .venv
source .venv/bin/activate

# If you use conda, run:
conda create -n dia-crying python=3.10
conda activate dia-crying

# Install dependencies
pip install -r requirements.txt

# Install Dia straight from GitHub
pip install git+https://github.com/nari-labs/dia.git
```

### Key library versions

| Library        | Tested version |
|----------------|----------------|
| `torch`        | 2.6.0 |
| `torchaudio`   | 2.6.0 |
| `numpy`        | ≥ 1.26 |
| `pandas`       | ≥ 2.2 |
| `tqdm`         | ≥ 4.66 |

*(For perfect reproducibility use `pip install -r requirements.txt`.)*

---

## Dataset

This repo expects the **Deeply – Nonverbal Vocalization** dataset (OpenSLR **SLR‑99**).

```bash
wget https://openslr.org/resources/99/NonverbalVocalization.tgz
tar -xzf NonverbalVocalization.tgz
mkdir -p data && mv NonverbalVocalization data/
```

The manifest script looks for the corpus under `data/NonverbalVocalization/`.  
Edit `SRC_ROOT` inside `01_prepare_manifest.py` if you keep it elsewhere.

---

## Run the pipeline

```bash
# 1 — create manifest, resample wavs
python 01_prepare_manifest.py

# 2 — encode with Dia DAC (GPU suggested)
python 02_encode_dac.py

# 3 — fine‑tune
python 03_finetune.py

# 4 — listen
python 04_test.py
```

> **Tip:** export `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to reduce CUDA OOM likelihood.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out‑of‑memory in step 2 | Lower `--batch_size` or run on CPU (slower) |
| `FileNotFoundError: ... NonverbalVocalization ...` | Check dataset path or tweak `SRC_ROOT` |
| `ModuleNotFoundError: dia` | Re‑run the Dia install line inside the active venv |

---

## Citation

```bibtex
@misc{dia2024,
  title  = {Dia: A Multilingual Neural Codec Model},
  author = {Nari Labs},
  year   = {2024},
  url    = {https://github.com/nari-labs/dia}
}
```
