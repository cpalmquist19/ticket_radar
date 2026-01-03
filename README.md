# Ticket Radar

A production-ready ticket classification system using fine-tuned DistilBERT. Classifies support tickets into categories (Change, Issue, Request) with high accuracy.

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Performance](#-model-performance)
  - [Classification Performance (Test Set)](#classification-performance-test-set)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training a New Model](#training-a-new-model)
    - [Converting to ONNX](#converting-to-onnx)
    - [Validating ONNX Model](#validating-onnx-model)
    - [Running Inference](#running-inference)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
  - [Architecture](#architecture)
  - [Training Configuration](#training-configuration)
  - [ONNX Conversion](#onnx-conversion)
- [Development](#-development)
  - [Data Preparation](#data-preparation)
  - [Model Training Tips](#model-training-tips)
- [Future Development](#-future-development)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

Ticket Radar is an ML-powered ticket classification system that:
- **Classifies tickets** into three categories: Change, Issue, and Request
- **Achieves 98%+ accuracy** on test data
- **Optimized for production** with ONNX conversion (99.7% size reduction)
- **Ready for deployment** with validated model conversion

This project demonstrates end-to-end ML development from training to optimized deployment.

## ✨ Features

- **Fine-tuned DistilBERT** model for ticket classification
- **ONNX optimization**: 255 MB → 0.7 MB (99.7% reduction) while maintaining 100% prediction accuracy
- **Model validation**: Comprehensive validation script to verify ONNX conversion fidelity
- **Production-ready**: Optimized model format suitable for deployment

## 📊 Model Performance

- **Test Accuracy**: 98.20%
- **Model Size**: 0.7 MB (ONNX format)
- **Classes**: Change, Issue, Request
- **Max Sequence Length**: 512 tokens

### Classification Performance (Test Set)

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Change     | 0.98      | 0.98   | 0.98     |
| Issue      | 0.98      | 0.99   | 0.99     |
| Request    | 0.98      | 0.97   | 0.98     |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for training)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/cpalmquist19/ticket_radar.git
cd ticket_radar
```

2. **Create a virtual environment**
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Training a New Model

```bash
python scripts/train_model.py \
    --csv_path training/aa_dataset-tickets-multi-lang-5-2-50-version_cleaned.csv \
    --text_column body \
    --label_column type \
    --output_dir models \
    --batch_size 16 \
    --num_epochs 3
```

#### Converting to ONNX

Convert the trained PyTorch model to ONNX format for optimized inference:

```bash
python scripts/convert_to_onnx.py \
    --model_path models/best_model \
    --opset_version 14
```

This creates `models/best_model/model.onnx` (0.7 MB) from the original PyTorch model.

#### Validating ONNX Model

Verify that the ONNX model matches the PyTorch model:

```bash
python scripts/validate_onnx.py \
    --model_path models/best_model \
    --max_samples 500
```

Expected output:
- ✅ 100% prediction agreement
- ✅ All logits within tolerance
- ✅ Identical accuracy scores

#### Running Inference

Classify new tickets using the trained model:

**Single text prediction:**
```bash
python scripts/inference.py --text "I need to update my password"
```

**Output:**
```
Prediction: Change
Confidence: 99.85%
```

**Multiple texts from file:**
```bash
python scripts/inference.py --file tickets.txt
```

**CSV file with specific column:**
```bash
python scripts/inference.py --file data.csv --csv_column body
```

**JSON output (for programmatic use):**
```bash
python scripts/inference.py --text "My email isn't working!" --json
```

**Verbose output (shows all class probabilities):**
```bash
python scripts/inference.py --text "Ticket text" --verbose
```

**Use PyTorch model instead of ONNX:**
```bash
python scripts/inference.py --text "Ticket text" --use_pytorch
```

The inference script automatically:
- Uses ONNX model by default (faster, smaller)
- Detects and uses GPU if available
- Provides confidence scores for all predictions
- Supports batch processing from files

## 📁 Project Structure

```
ticket_radar/
├── models/
│   └── best_model/          # Trained model files
│       ├── model.onnx       # Optimized ONNX model (0.7 MB)
│       ├── config.json      # Model configuration
│       ├── label_encoder.pkl # Label encoder
│       └── tokenizer files  # Tokenizer configuration
├── scripts/
│   ├── train_model.py       # Model training script
│   ├── convert_to_onnx.py   # ONNX conversion script
│   ├── validate_onnx.py     # Model validation script
│   ├── inference.py         # Inference script for predictions
│   ├── clean_data.py        # Data cleaning utilities
│   └── check_labels.py      # Label analysis
├── training/
│   └── *.csv                # Training datasets
├── docs/
│   ├── project_readme.md    # Project vision and architecture
│   └── project_tracker.md   # Implementation progress
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Model Details

### Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Sequence Classification
- **Classes**: 3 (Change, Issue, Request)
- **Hidden Size**: 768
- **Max Length**: 512 tokens

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Class Weights**: Balanced (handles class imbalance)

### ONNX Conversion
- **Format**: ONNX (Open Neural Network Exchange)
- **Size Reduction**: 255 MB → 0.7 MB (99.7%)
- **Accuracy**: Maintains 100% prediction agreement with PyTorch model
- **Quantization**: INT8 quantization supported (optional)

## 🛠️ Development

### Data Preparation

Clean and prepare your dataset:

```bash
python scripts/clean_data.py \
    --input training/aa_dataset-tickets-multi-lang-5-2-50-version.csv \
    --output training/aa_dataset-tickets-multi-lang-5-2-50-version_cleaned.csv
```

### Model Training Tips

- Use GPU for faster training (automatically detected)
- Adjust `batch_size` based on GPU memory
- Monitor validation accuracy to prevent overfitting
- Class weights are automatically computed for imbalanced datasets

## 🔮 Future Development

This project is part of a larger **Ticket Classifier Observability Hub** vision:

- [ ] **FastAPI Service**: Production REST API with input/output validation
- [ ] **Docker Deployment**: Containerized service for consistent deployments
- [ ] **Event Streaming**: Kafka integration for real-time prediction tracking
- [ ] **Data Warehouse**: Snowflake integration for drift detection
- [ ] **LLM-as-a-Judge**: Quality monitoring using LLM-based evaluation
- [ ] **Infrastructure as Code**: Terraform configurations for scalable deployment

See `docs/project_readme.md` for the complete architecture plan.

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

- **Author**: Caleb Palmquist
- **GitHub**: [@cpalmquist19](https://github.com/cpalmquist19)
- **LinkedIn**: [Caleb Palmquist](https://www.linkedin.com/in/calebfrommaine/)

## 🙏 Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Model optimization using [ONNX Runtime](https://onnxruntime.ai/)
- Training framework: PyTorch

---

**Status**: ✅ Model trained and validated | 🚧 API and deployment are the next milestones.