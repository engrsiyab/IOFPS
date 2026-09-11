# Identification of Oncogenic Fungal Protein Sequences

This repository provides code, datasets, feature embeddings, and evaluation results for the identification of oncogenic fungal protein sequences using protein language models (ESM-2, ProtBERT, ProtT5, ProGen) and deep learning architectures.

---

## 📂 Directory Structure

```text
├── Dataset/                           # Curated positive and negative protein sequence datasets
├── esm_results/                       # Embeddings, metrics, and models for ESM
├── progen_results/                    # Embeddings, metrics, and models for ProGen
├── protbert_Results/                  # Embeddings, metrics, and models for ProtBERT
├── prott5_Results/                    # Embeddings, metrics, and models for ProtT5
├── Gradient_search/                   # Hyperparameter tuning results and logs
├── ESM_ProtBert_emb_genrator.py       # Feature extraction script for ESM and ProtBERT
├── Prot_T5_emb_genrator.py            # Feature extraction script for ProtT5
├── Progen_emb_genrator.py             # Feature extraction script for ProGen
├── gradient_search.py                 # Hyperparameter search script
├── model_code.py                      # Classification model training & evaluation script
├── method.pdf                        # Overall workflow and pipeline architecture
└── fig_curation_methodology.pdf       # Step-by-step dataset curation methodology
```

---

## 📄 Documentation Files

- **`method.pdf`**: Illustrates the overall pipeline flow, from sequence pre-processing and language model embedding extraction to deep neural classification.
- **`fig_curation_methodology.pdf`** (or `dataset_curation_methodology.pdf`): Outlines the complete sequence extraction, homology reduction, and curation protocol used to build the benchmark datasets.

---

## 🚀 Running the Code on Kaggle (Recommended)

Because protein language models require GPU acceleration, running on Kaggle is recommended:

1. **Upload Data**: Create a new dataset on Kaggle and upload this folder (`paper_Data.zip`).
2. **Create Notebook**: Open a Kaggle Notebook and link the uploaded dataset under **Input**.
3. **Enable GPU**: Go to the right panel settings and set **Accelerator** to **GPU P100** or **GPU T4 x2**.
4. **Run Scripts**: Execute the scripts directly in the notebook cells:

```bash
# 1. Generate Embeddings (if generating fresh embeddings)
python ESM_ProtBert_emb_genrator.py
python Prot_T5_emb_genrator.py
python Progen_emb_genrator.py

# 2. Hyperparameter Optimization
python gradient_search.py

# 3. Model Training and Evaluation
python model_code.py
```

---

## 📊 Results & Embeddings

- **Datasets**: Pre-processed training and testing sequences are located in `Dataset/`.
- **Embeddings & Results**: Pre-generated embeddings, performance metrics, and evaluation curves for each protein model are stored in their respective result folders:
  - `esm_results/`
  - `progen_results/`
  - `protbert_Results/`
  - `prott5_Results/`
- **Tuning Logs**: Parameter optimization logs and configurations are available in `Gradient_search/`.

---

## 📜 Citation

If you use this work or dataset, please cite our corresponding research article.
