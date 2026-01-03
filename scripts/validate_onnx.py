"""
Validate ONNX model against original PyTorch model.
Compares predictions, accuracy, and numerical outputs to ensure conversion fidelity.
"""
import os
import pickle
import numpy as np
import torch
import onnxruntime as ort
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import argparse
from tqdm import tqdm

def load_test_data(csv_path, text_column='body', label_column='type', max_samples=1000):
    """Load test data for validation."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    print(f"Loading test data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_column, label_column])
    
    # Apply same preprocessing as training
    df[label_column] = df[label_column].replace({'Problem': 'Issue', 'Incident': 'Issue'})
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Load label encoder
    model_path = Path('models/best_model')
    with open(model_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Encode labels
    encoded_labels = label_encoder.transform(labels)
    
    # Limit samples for faster validation
    if len(texts) > max_samples:
        texts = texts[:max_samples]
        encoded_labels = encoded_labels[:max_samples]
    
    print(f"Loaded {len(texts)} samples for validation")
    return texts, encoded_labels, label_encoder

def predict_pytorch(model, tokenizer, texts, device, batch_size=32, max_length=512):
    """Get predictions from PyTorch model."""
    model.eval()
    predictions = []
    logits_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="PyTorch inference"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
    
    return np.array(predictions), np.concatenate(logits_list, axis=0)

def predict_onnx(onnx_path, tokenizer, texts, device=None, batch_size=1, max_length=512):
    """Get predictions from ONNX model."""
    # Determine execution provider based on device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check available ONNX Runtime providers
    available_providers = ort.get_available_providers()
    
    # Set up ONNX Runtime providers (prefer GPU if available)
    if device.type == 'cuda' and torch.cuda.is_available() and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"   Using GPU (CUDAExecutionProvider) for ONNX inference")
    else:
        providers = ['CPUExecutionProvider']
        if device.type == 'cuda':
            print(f"   ⚠️  CUDAExecutionProvider not available, using CPU")
            print(f"   💡 Install onnxruntime-gpu: pip install onnxruntime-gpu")
        else:
            print(f"   Using CPU for ONNX inference")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    predictions = []
    logits_list = []
    
    # Process one at a time to handle dynamic batch sizes
    # (ONNX models with dynamic axes can be finicky with batching)
    for i in tqdm(range(len(texts)), desc="ONNX inference"):
        text = texts[i]
        
        # Tokenize
        encodings = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )
        input_ids = encodings['input_ids'].astype(np.int64)
        attention_mask = encodings['attention_mask'].astype(np.int64)
        
        # Run ONNX inference
        outputs = session.run(
            None,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        )
        logits = outputs[0]
        preds = np.argmax(logits, axis=1)
        
        predictions.append(preds[0])
        logits_list.append(logits[0])
    
    return np.array(predictions), np.array(logits_list)

def compare_models(
    model_path='models/best_model',
    onnx_path=None,
    csv_path='training/aa_dataset-tickets-multi-lang-5-2-50-version_cleaned.csv',
    max_samples=1000,
    tolerance=1e-4
):
    """
    Compare PyTorch and ONNX models.
    
    Args:
        model_path: Path to PyTorch model directory
        onnx_path: Path to ONNX model (default: model_path/model.onnx)
        csv_path: Path to test CSV file
        max_samples: Maximum number of samples to test
        tolerance: Numerical tolerance for logit comparison
    """
    model_path = Path(model_path)
    if onnx_path is None:
        onnx_path = model_path / 'model.onnx'
    else:
        onnx_path = Path(onnx_path)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    print("="*70)
    print("ONNX Model Validation")
    print("="*70)
    
    # Load test data
    texts, true_labels, label_encoder = load_test_data(csv_path, max_samples=max_samples)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ GPU detected: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device} (CPU)")
    
    # Load PyTorch model
    print(f"\nLoading PyTorch model from {model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    pytorch_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    pytorch_model.to(device)
    pytorch_model.eval()
    
    # Get PyTorch predictions
    print("\n" + "="*70)
    print("Running PyTorch Model Inference")
    print("="*70)
    pytorch_preds, pytorch_logits = predict_pytorch(
        pytorch_model, tokenizer, texts, device
    )
    
    # Get ONNX predictions
    print("\n" + "="*70)
    print("Running ONNX Model Inference")
    print("="*70)
    onnx_preds, onnx_logits = predict_onnx(
        onnx_path, tokenizer, texts, device=device
    )
    
    # Compare predictions
    print("\n" + "="*70)
    print("Comparison Results")
    print("="*70)
    
    # Prediction agreement
    agreement = np.mean(pytorch_preds == onnx_preds)
    disagreements = np.sum(pytorch_preds != onnx_preds)
    
    print(f"\n📊 Prediction Agreement: {agreement*100:.2f}%")
    print(f"   Total samples: {len(pytorch_preds)}")
    print(f"   Agree: {len(pytorch_preds) - disagreements}")
    print(f"   Disagree: {disagreements}")
    
    if disagreements > 0:
        print(f"\n⚠️  Warning: {disagreements} predictions differ between models")
        # Show some examples of disagreements
        diff_indices = np.where(pytorch_preds != onnx_preds)[0][:5]
        print("\n   Example disagreements (first 5):")
        for idx in diff_indices:
            print(f"   Sample {idx}:")
            print(f"     Text: {texts[idx][:100]}...")
            print(f"     True label: {label_encoder.classes_[true_labels[idx]]}")
            print(f"     PyTorch: {label_encoder.classes_[pytorch_preds[idx]]}")
            print(f"     ONNX: {label_encoder.classes_[onnx_preds[idx]]}")
    
    # Compare logits (numerical differences)
    print(f"\n📈 Numerical Comparison (Logits):")
    logit_diff = np.abs(pytorch_logits - onnx_logits)
    max_diff = np.max(logit_diff)
    mean_diff = np.mean(logit_diff)
    std_diff = np.std(logit_diff)
    
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Std difference: {std_diff:.6f}")
    print(f"   Tolerance: {tolerance}")
    
    if max_diff < tolerance:
        print(f"   ✅ All logits within tolerance!")
    else:
        exceeding = np.sum(logit_diff > tolerance)
        print(f"   ⚠️  {exceeding} logit values exceed tolerance")
    
    # Accuracy comparison
    print(f"\n🎯 Accuracy Comparison:")
    pytorch_acc = accuracy_score(true_labels, pytorch_preds)
    onnx_acc = accuracy_score(true_labels, onnx_preds)
    acc_diff = abs(pytorch_acc - onnx_acc)
    
    print(f"   PyTorch accuracy: {pytorch_acc:.4f}")
    print(f"   ONNX accuracy: {onnx_acc:.4f}")
    print(f"   Difference: {acc_diff:.6f}")
    
    if acc_diff < 0.001:  # Less than 0.1% difference
        print(f"   ✅ Accuracy difference is negligible!")
    elif acc_diff < 0.01:  # Less than 1% difference
        print(f"   ⚠️  Small accuracy difference (likely due to numerical precision)")
    else:
        print(f"   ❌ Significant accuracy difference!")
    
    # Classification reports
    print(f"\n📋 PyTorch Model Classification Report:")
    print(classification_report(
        true_labels, pytorch_preds,
        target_names=label_encoder.classes_
    ))
    
    print(f"\n📋 ONNX Model Classification Report:")
    print(classification_report(
        true_labels, onnx_preds,
        target_names=label_encoder.classes_
    ))
    
    # Summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    if agreement == 1.0 and max_diff < tolerance:
        print("✅ PASS: Models are identical!")
        print("   - 100% prediction agreement")
        print("   - All logits within tolerance")
    elif agreement >= 0.99 and acc_diff < 0.01:
        print("✅ PASS: Models are functionally equivalent!")
        print("   - >99% prediction agreement")
        print("   - Accuracy difference < 1%")
    elif agreement >= 0.95:
        print("⚠️  WARNING: Models show minor differences")
        print("   - May be acceptable depending on use case")
        print("   - Review disagreements above")
    else:
        print("❌ FAIL: Models show significant differences")
        print("   - Review conversion process")
    
    return {
        'agreement': agreement,
        'max_logit_diff': max_diff,
        'mean_logit_diff': mean_diff,
        'pytorch_accuracy': pytorch_acc,
        'onnx_accuracy': onnx_acc,
        'accuracy_diff': acc_diff
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate ONNX model against PyTorch model')
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/best_model',
        help='Path to PyTorch model directory'
    )
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=None,
        help='Path to ONNX model (default: model_path/model.onnx)'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='training/aa_dataset-tickets-multi-lang-5-2-50-version_cleaned.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=1000,
        help='Maximum number of samples to test'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-4,
        help='Numerical tolerance for logit comparison'
    )
    
    args = parser.parse_args()
    
    compare_models(
        model_path=args.model_path,
        onnx_path=args.onnx_path,
        csv_path=args.csv_path,
        max_samples=args.max_samples,
        tolerance=args.tolerance
    )

