import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse
from pathlib import Path
from tqdm import tqdm

class TicketDataset(Dataset):
    """Dataset class for ticket classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(csv_path, text_column='body', label_column='type', test_size=0.2, val_size=0.1):
    """
    Load CSV and prepare train/val/test splits.
    
    Args:
        csv_path: Path to CSV file
        text_column: Column name containing text to classify
        label_column: Column name containing labels
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation (after test split)
    
    Returns:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=[text_column, label_column])
    
    # Combine Problem and Incident into a single "Issue" class
    print("Combining 'Problem' and 'Incident' into 'Issue' class...")
    df[label_column] = df[label_column].replace({'Problem': 'Issue', 'Incident': 'Issue'})
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Total samples: {len(texts)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Class distribution:\n{df[label_column].value_counts()}")
    
    # Split into train/test first
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
    )
    
    # Split train into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=42, stratify=train_labels
    )
    
    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder

def train_epoch(model, dataloader, optimizer, scheduler, device, use_amp=False, scaler=None, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    # Setup weighted loss if class weights provided
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use mixed precision training if available
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # Use weighted loss instead of model's built-in loss
                loss = criterion(outputs.logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Use weighted loss instead of model's built-in loss
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device, use_amp=False):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

def train_model(
    csv_path,
    text_column='body',
    label_column='type',
    model_name='distilbert-base-uncased',
    output_dir='models',
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    max_length=512,
    test_size=0.2,
    val_size=0.1
):
    """
    Fine-tune DistilBERT for text classification.
    
    Args:
        csv_path: Path to CSV file
        text_column: Column name containing text
        label_column: Column name containing labels
        model_name: HuggingFace model name
        output_dir: Directory to save model
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        max_length: Maximum sequence length
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data after test split)
    """
    # Set device and verify GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU detected: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"Using device: {device}")
        
        # Verify GPU is working
        test_tensor = torch.randn(1, 1).to(device)
        _ = test_tensor * 2  # Simple operation to verify GPU works
        print("✓ GPU test passed - GPU is ready for training")
        
        # Enable mixed precision training for faster training (2x speedup on RTX GPUs)
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
        print("✓ Mixed precision training (FP16) enabled - expect ~2x speedup")
        
        # Suggest batch size optimization for 8GB GPU
        if gpu_memory >= 8 and batch_size < 32:
            print(f"💡 Tip: With {gpu_memory:.0f}GB GPU, you can increase batch_size to 32-64 for faster training")
    else:
        device = torch.device('cpu')
        print("⚠ WARNING: CUDA not available, using CPU (training will be very slow)")
        print("⚠ Make sure you have:")
        print("  1. NVIDIA GPU with CUDA support")
        print("  2. PyTorch installed with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        use_amp = False
        scaler = None
    
    # Load and prepare data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder = load_and_prepare_data(
        csv_path, text_column, label_column, test_size, val_size
    )
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights: {class_weights_dict}")
    print("(Higher weight = more penalty for misclassifying that class)")
    
    # Initialize tokenizer and model
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )
    model.to(device)
    
    # Create datasets
    train_dataset = TicketDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TicketDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TicketDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Create dataloaders with optimizations for GPU
    num_workers = 4 if device.type == 'cuda' else 0  # Use workers for GPU, 0 for CPU (Windows multiprocessing issues)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp, scaler, class_weights_tensor)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, use_amp)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_path = output_path / 'best_model'
            best_model_path.mkdir(exist_ok=True)
            
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            
            # Save label encoder
            import pickle
            with open(best_model_path / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            
            print(f"✓ New best model saved! (Val Accuracy: {val_acc:.4f})")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    # Load best model for testing
    model = DistilBertForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)
    
    test_loss, test_acc, test_predictions, test_true_labels = evaluate(model, test_loader, device, use_amp)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_true_labels,
        test_predictions,
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_true_labels, test_predictions)
    print(cm)
    
    print(f"\nModel and tokenizer saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT for ticket classification')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=r'C:\Code\ml\ticket_intent\training\aa_dataset-tickets-multi-lang-5-2-50-version_cleaned.csv',
        help='Path to CSV file'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='body',
        help='Column name containing text to classify'
    )
    parser.add_argument(
        '--label_column',
        type=str,
        default='type',
        help='Column name containing labels'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='Proportion of remaining data for validation'
    )
    
    args = parser.parse_args()
    
    train_model(
        csv_path=args.csv_path,
        text_column=args.text_column,
        label_column=args.label_column,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        test_size=args.test_size,
        val_size=args.val_size
    )

