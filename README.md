# Phish360 — Multimodal phishing detection with Big Data and Deep Learning

Detects phishing web pages by combining **four modalities** of each page — the **URL**, the **HTML**, a **screenshot** and **technical features** — with a distributed **Apache Spark** pipeline and deep learning models (character-level CNN, **DistilBERT**, **ResNet18**) merged by late and intermediate fusion.

*Course project — Big Data & Deep Learning, TEK-UP University (Data Science & AI engineering).*

## Research questions

- **RQ1** — Does fusing modalities beat every single modality?
- **RQ2** — Does fusion generalise better over time, to more recent phishing campaigns?
- **RQ3** — What is the compute cost of the distributed pipeline as data volume grows?

## Results (F1 on the phishing class)

Spark MLlib baselines:

| Model | RQ1 (trainval → test) | RQ2 (older → recent) |
| --- | --- | --- |
| URL TF-IDF + logistic regression | 0.807 | 0.694 |
| HTML TF-IDF + logistic regression | 0.913 | 0.884 |
| Technical features + GBT | **0.934** | 0.867 |
| Concatenation + logistic regression | **0.934** | 0.846 |

Deep models and fusion results are produced by `evaluate_final.py` in `results/deep_results.csv` and `results/fusion_ablation.csv`.

## Architecture

```
Phish360 raw pages ──► Spark ingestion (Parquet) ──► EDA, cleaning, temporal split (RQ2)
                                   │
                    distributed preprocessing of 4 modalities
       ┌──────────────┬──────────────┬──────────────┬──────────────┐
      URL            HTML        screenshot      technical
   char-CNN       DistilBERT      ResNet18      35 features
   (128-d)         (256-d)        (256-d)        XGBoost
       └──────────────┴──────┬───────┴──────────────┘
                fusion: stacking · concat MLP · gated (attention) MLP
                             │
     evaluation (RQ1/RQ2) · explainability (SHAP, Grad-CAM) · cost benchmark (RQ3)
```

- **Fusion strategies** — *stacking* (late fusion: a logistic regression over the 4 modality probabilities, trained only on validation pages), *concat* (intermediate fusion: each modality projected to 64-d, then an MLP) and *gated* (an attention gate weights each modality per page, showing which one the model relied on). The MLPs use **modality dropout** and are averaged over 3 seeds.
- **Ablation** — every combination of modalities (15) is re-trained to answer RQ1 directly.
- **Explainability** — SHAP on technical features, Grad-CAM on screenshots, gradient × input attribution on DistilBERT tokens, error analysis.

## Pipeline

| Step | Script(s) | Role |
| --- | --- | --- |
| 1 | `check_env.py`, `hello_spark.py`, `check_deep.py` | check the Spark / deep learning environment |
| 2 | `ingest.py`, `quality_report.py` | distributed ingestion to Parquet, in batches; quality report |
| 3 | `extract_meta.py`, `eda.py`, `clean_index.py`, `rq2_split.py` | EDA, cleaning, temporal split for RQ2 |
| 4 | `preprocess.py` | distributed preprocessing of the 4 modalities |
| 5 | `baselines.py` | Spark MLlib baselines (TF-IDF, GBT, concatenation) |
| 6 | `url_charcnn.py` | character-level CNN on URLs |
| 7 | `html_distilbert.py` | fine-tuned DistilBERT on page text |
| 8 | `img_resnet18.py` | fine-tuned ResNet18 (ImageNet) on screenshots |
| 9 | `fusion.py` | stacking / concat / gated fusion + ablation |
| 10 | `evaluate_final.py` | final comparison of all models (RQ1, RQ2) |
| 11 | `benchmark_rq3.py` | pipeline cost vs. volume and number of cores (RQ3) |
| 12 | `explain.py` | explainability and error analysis |

`make_fake_data.py` generates synthetic data to test steps 6–12 without the real dataset or a GPU.

## Data

Dataset **Phish360** (trainval: 7,977 pages, test: 2,771). The data is **not** in this repository (size, and it contains real phishing pages). Put it in a local folder, e.g. `C:\bigdata\Phish360`.

## Setup (Windows)

Requirements: Python 3.11 (or 3.13 with PySpark 4), Java 17, `HADOOP_HOME` with `winutils.exe`.

```powershell
.\setup_venv.ps1 -Gpu          # creates .venv, installs requirements.txt and torch (CUDA)
.\.venv\Scripts\Activate.ps1
python src\check_env.py        # every line should print [OK]

python src\ingest.py "C:/bigdata/Phish360"
python src\quality_report.py
python src\extract_meta.py; python src\eda.py; python src\clean_index.py; python src\rq2_split.py
python src\preprocess.py
python src\baselines.py
.\run_sprint2.ps1              # deep models, fusion, evaluation, benchmark, explainability
```

Spark jobs run on 4 cores by default, in resumable batches; timings are logged in `results/timings.csv` (RQ3).

## Tech stack

PySpark · Spark MLlib · PyTorch · Hugging Face Transformers · torchvision · XGBoost · scikit-learn · SHAP · pandas · PyArrow

## Author

**Saif Sbaiti** — final-year engineering student, Data Science & AI, TEK-UP University (Tunisia) · [LinkedIn](https://www.linkedin.com/in/saif-sbaiti-3b9a32360)
