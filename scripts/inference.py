"""
Inference script for Ticket Radar classification model.
Supports both ONNX (recommended) and PyTorch models.
"""
import pickle
import numpy as np
import torch
import onnxruntime as ort
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import argparse
import json
import sys
from typing import List, Dict, Tuple, Optional

def load_model_onnx(model_path: Path, onnx_path: Optional[Path] = None):
    """Load ONNX model and tokenizer."""
    if onnx_path is None:
        onnx_path = model_path / 'model.onnx'
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Check available providers
    available_providers = ort.get_available_providers()
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    # Load label encoder
    with open(model_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return session, tokenizer, label_encoder, 'onnx'

def load_model_pytorch(model_path: Path, device: Optional[torch.device] = None):
    """Load PyTorch model and tokenizer."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load label encoder
    with open(model_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder, device, 'pytorch'

def predict_onnx(
    session: ort.InferenceSession,
    tokenizer: DistilBertTokenizer,
    texts: List[str],
    max_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using ONNX model."""
    predictions = []
    logits_list = []
    
    for text in texts:
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
        
        # Run inference
        outputs = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        logits = outputs[0][0]
        pred = np.argmax(logits)
        
        predictions.append(pred)
        logits_list.append(logits)
    
    return np.array(predictions), np.array(logits_list)

def predict_pytorch(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using PyTorch model."""
    predictions = []
    logits_list = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encodings = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Run inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0].cpu().numpy()
            pred = np.argmax(logits)
            
            predictions.append(pred)
            logits_list.append(logits)
    
    return np.array(predictions), np.array(logits_list)

def get_confidence_scores(logits: np.ndarray) -> np.ndarray:
    """Convert logits to confidence scores using softmax."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def format_results(
    texts: List[str],
    predictions: np.ndarray,
    confidence_scores: np.ndarray,
    label_encoder,
    output_json: bool = False
) -> List[Dict]:
    """Format prediction results."""
    results = []
    
    for i, text in enumerate(texts):
        pred_class = label_encoder.classes_[predictions[i]]
        confidence = float(confidence_scores[i][predictions[i]])
        
        # Get all class probabilities
        class_probs = {
            label_encoder.classes_[j]: float(confidence_scores[i][j])
            for j in range(len(label_encoder.classes_))
        }
        
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': pred_class,
            'confidence': confidence,
            'class_probabilities': class_probs
        }
        results.append(result)
    
    return results

def print_results(results: List[Dict], verbose: bool = False):
    """Print results in a human-readable format."""
    for i, result in enumerate(results):
        print(f"\n{'='*70}")
        print(f"Prediction {i + 1}")
        print(f"{'='*70}")
        print(f"Text: {result['text']}")
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if verbose:
            print(f"\nAll Class Probabilities:")
            for class_name, prob in sorted(result['class_probabilities'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.2%}")

def load_texts_from_file(file_path: Path, csv_column: Optional[str] = None) -> List[str]:
    """Load texts from a file (one per line or CSV)."""
    if csv_column:
        import pandas as pd
        df = pd.read_csv(file_path)
        if csv_column not in df.columns:
            raise ValueError(f"Column '{csv_column}' not found in CSV. Available columns: {list(df.columns)}")
        return df[csv_column].astype(str).tolist()
    else:
        # Try as text file (one text per line)
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(
        description='Run inference on ticket classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text prediction
  python scripts/inference.py --text "I need to update my password"

  # Multiple texts from file
  python scripts/inference.py --file tickets.txt

  # CSV file with specific column
  python scripts/inference.py --file data.csv --csv_column body

  # Use PyTorch model instead of ONNX
  python scripts/inference.py --text "Ticket text" --use_pytorch

  # Output as JSON
  python scripts/inference.py --text "Ticket text" --json

  # Verbose output with all probabilities
  python scripts/inference.py --text "Ticket text" --verbose
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        type=str,
        help='Text to classify'
    )
    input_group.add_argument(
        '--file',
        type=str,
        help='File containing texts (one per line) or CSV file'
    )
    
    # Model options
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/best_model',
        help='Path to model directory (default: models/best_model)'
    )
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=None,
        help='Path to ONNX model file (default: model_path/model.onnx)'
    )
    parser.add_argument(
        '--use_pytorch',
        action='store_true',
        help='Use PyTorch model instead of ONNX (default: use ONNX if available)'
    )
    
    # Output options
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed probabilities for all classes'
    )
    parser.add_argument(
        '--csv_column',
        type=str,
        default=None,
        help='Column name when input file is CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (for JSON output)'
    )
    
    args = parser.parse_args()
    
    # Load texts
    if args.text:
        texts = [args.text]
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        texts = load_texts_from_file(file_path, args.csv_column)
        if not texts:
            print(f"Error: No texts found in file: {file_path}", file=sys.stderr)
            sys.exit(1)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load model
    try:
        if args.use_pytorch:
            print("Loading PyTorch model...")
            model, tokenizer, label_encoder, device, model_type = load_model_pytorch(model_path)
            print(f"✓ Model loaded (PyTorch on {device})")
            
            predictions, logits = predict_pytorch(model, tokenizer, texts, device)
        else:
            print("Loading ONNX model...")
            onnx_path = Path(args.onnx_path) if args.onnx_path else None
            session, tokenizer, label_encoder, model_type = load_model_onnx(model_path, onnx_path)
            print(f"✓ Model loaded (ONNX)")
            
            predictions, logits = predict_onnx(session, tokenizer, texts)
        
        # Convert logits to confidence scores
        confidence_scores = get_confidence_scores(logits)
        
        # Format results
        results = format_results(texts, predictions, confidence_scores, label_encoder, args.json)
        
        # Output results
        if args.json:
            output_data = {
                'model_type': model_type,
                'num_predictions': len(results),
                'results': results
            }
            json_output = json.dumps(output_data, indent=2, ensure_ascii=False)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"Results saved to: {args.output}")
            else:
                print(json_output)
        else:
            print_results(results, args.verbose)
            print(f"\n{'='*70}")
            print(f"Total predictions: {len(results)}")
            print(f"{'='*70}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.use_pytorch:
            print("Tip: Try using --use_pytorch to use the PyTorch model instead", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

