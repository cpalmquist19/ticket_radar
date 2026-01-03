"""
Convert DistilBERT model to ONNX format with quantization for smaller file size.
ONNX models are more portable and can be significantly smaller with quantization.
"""
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import argparse

def convert_to_onnx(
    model_path,
    output_path=None,
    quantize=True,
    opset_version=14,
    max_length=512
):
    """
    Convert a DistilBERT model to ONNX format.
    
    Args:
        model_path: Path to the saved model directory
        output_path: Path to save ONNX model (default: model_path/model.onnx)
        quantize: Whether to apply INT8 quantization (reduces size by ~4x)
        opset_version: ONNX opset version (14 is stable for transformers)
        max_length: Maximum sequence length
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from {model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Create dummy input for tracing
    dummy_text = "This is a sample ticket description for conversion."
    inputs = tokenizer(
        dummy_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Set output path
    if output_path is None:
        output_path = model_path / 'model.onnx'
    else:
        output_path = Path(output_path)
    
    print(f"\nConverting to ONNX (opset {opset_version})...")
    print(f"Output will be saved to: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    # Get original size
    original_size = (model_path / 'model.safetensors').stat().st_size / (1024 * 1024)
    onnx_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  ONNX size: {onnx_size:.2f} MB")
    print(f"  Size reduction: {((original_size - onnx_size) / original_size * 100):.1f}%")
    
    if quantize:
        print(f"\nApplying INT8 quantization (this may take a few minutes)...")
        quantized_path = model_path / 'model_quantized.onnx'
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                model_input=str(output_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QUInt8,
                optimize_model=True
            )
            
            quantized_size = quantized_path.stat().st_size / (1024 * 1024)
            print(f"✓ Quantization complete!")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Total reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
            print(f"  Quantized model saved to: {quantized_path}")
            
        except ImportError:
            print("⚠ onnxruntime not installed. Skipping quantization.")
            print("  Install with: pip install onnxruntime")
        except Exception as e:
            print(f"⚠ Quantization failed: {e}")
            print("  You can still use the non-quantized ONNX model.")
    
    # Save tokenizer config for ONNX usage
    print(f"\n✓ Model converted successfully!")
    print(f"  ONNX model: {output_path}")
    if quantize and quantized_path.exists():
        print(f"  Quantized model: {quantized_path}")
    print(f"\nNote: You'll need the tokenizer from {model_path} to use this model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert DistilBERT model to ONNX')
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/best_model',
        help='Path to the saved model directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save ONNX model (default: model_path/model.onnx)'
    )
    parser.add_argument(
        '--no_quantize',
        action='store_true',
        help='Skip quantization (faster conversion, larger file)'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=14,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    
    args = parser.parse_args()
    
    convert_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        quantize=not args.no_quantize,
        opset_version=args.opset_version,
        max_length=args.max_length
    )

