# Receipt Processing Pipeline

An end-to-end document processing system that classifies receipts, extracts structured fields, detects anomalies, and makes routing decisions. Built with ensemble learning and human-in-the-loop feedback.

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RogueTex/StreamingDataforModelTraining/blob/main/NewVerPynbAgent.ipynb)

Click the badge above to run the full pipeline in Google Colab with GPU support.

## What It Does

1. **Classification** — Determines if an uploaded image is a receipt (ViT + ResNet18 ensemble)
2. **OCR** — Extracts text using EasyOCR with automatic retry on low-quality images
3. **Field Extraction** — Identifies vendor, date, and total using a 4-strategy ensemble (LayoutLMv3, regex, position-based, NER)
4. **Anomaly Detection** — Flags suspicious patterns using Isolation Forest, XGBoost, HistGradientBoosting, and One-Class SVM
5. **Routing** — LangGraph workflow decides: approve, reject, or send for human review
6. **Learning** — Fine-tunes models from user feedback every 5 corrections

## Project Structure

```
StreamingDataforModelTraining/
├── NewVerPynbAgent.ipynb      # Main notebook (run this)
├── models/
│   ├── rvl_classifier.pt      # ViT document classifier (~21 MB)
│   ├── resnet18_rvlcdip.pt    # ResNet18 classifier (~44 MB)
│   ├── layoutlm_extractor.pt  # LayoutLMv3 field extraction (~478 MB)
│   └── anomaly_detector.pt    # Anomaly detection ensemble (~2 MB)
├── assets/images/             # Evaluation visualizations
├── feedback_data/             # Stored user corrections
├── docs/
│   ├── aml_presentation.tex   # Technical documentation
│   └── project_presentation.tex
└── README.md
```

## Performance

| Component | Single Model | Ensemble | Improvement |
|-----------|--------------|----------|-------------|
| Classification | 98% | 100% | +2% |
| Field Extraction (F1) | 72% | 79% | +7% |
| Anomaly Detection | 88% | 100% | +12% |

Results from 100-sample test set. Evaluation plots saved to `assets/images/`.

## Models

| File | Size | Description |
|------|------|-------------|
| `rvl_classifier.pt` | 21 MB | ViT-Tiny document classifier |
| `resnet18_rvlcdip.pt` | 44 MB | ResNet18 trained on RVL-CDIP |
| `layoutlm_extractor.pt` | 478 MB | LayoutLMv3 for field extraction |
| `anomaly_detector.pt` | 2 MB | 4-model anomaly ensemble |

Models are stored with Git LFS. Total size: ~550 MB.

## Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- EasyOCR
- LangGraph
- Gradio
- scikit-learn
- XGBoost

All dependencies are installed automatically in the Colab notebook.

## Usage

1. Open the notebook in Google Colab
2. Run cells 1-30 to load models
3. Run the Gradio cell to launch the interface
4. Upload a receipt image
5. Review results and provide feedback if needed

For local development:
```bash
git clone https://github.com/RogueTex/StreamingDataforModelTraining.git
cd StreamingDataforModelTraining
pip install torch transformers easyocr langgraph gradio scikit-learn xgboost
jupyter notebook NewVerPynbAgent.ipynb
```

## Documentation

- `docs/aml_presentation.tex` — Detailed technical documentation
- `docs/project_presentation.tex` — Project overview for presentations

Compile with pdflatex or upload to Overleaf. Images should be placed in an `images/` folder.

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LayoutLMv3](https://arxiv.org/abs/2204.08387)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [RVL-CDIP Dataset](https://www.cs.cmu.edu/~aharley/rvl-cdip/)

## License

MIT
