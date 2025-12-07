---
title: Receipt Processing Agent
emoji: 🧾
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Receipt Processing Agent

An intelligent document processing pipeline that automatically classifies receipts, extracts key fields, detects anomalies, and makes routing decisions.

## Features

- **Document Classification**: ViT + ResNet18 ensemble (100% accuracy)
- **OCR**: EasyOCR with confidence visualization
- **Field Extraction**: Vendor, date, total extraction
- **Anomaly Detection**: Rule-based suspicious pattern detection
- **Decision Routing**: APPROVE / REVIEW / REJECT

## How It Works

1. **Upload** a receipt image
2. **Classification** determines if it's actually a receipt
3. **OCR** extracts all text with bounding boxes
4. **Field Extraction** identifies vendor, date, and total
5. **Anomaly Detection** checks for suspicious patterns
6. **Routing** decides: approve, send for review, or reject

## Model Details

| Component | Model | Performance |
|-----------|-------|-------------|
| Classification | ViT-Tiny + ResNet18 | 100% accuracy |
| OCR | EasyOCR | 74% avg confidence |
| Field Extraction | Regex patterns | 79% F1 |
| Anomaly Detection | Rule-based | 100% accuracy |

## Full Pipeline

This is a simplified demo. The complete system includes:
- LayoutLMv3 for advanced field extraction
- 4-model anomaly detection ensemble (IsolationForest + XGBoost + HistGB + SVM)
- LangGraph agentic workflow with conditional branching
- Human feedback loop with automatic model fine-tuning

## Repository

Full code and documentation: [GitHub](https://github.com/RogueTex/StreamingDataforModelTraining)

## License

MIT

