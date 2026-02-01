"""
Complete training iteration workflow for active learning.

Handles the full training cycle: dataloaders, training, predictions,
metrics calculation, visualization, and next iteration setup.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
import imageio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class TrainEpochWithGradClip(smp.utils.train.TrainEpoch):
    """
    Custom TrainEpoch that adds gradient clipping for training stability.

    Gradient clipping prevents gradient explosion by limiting the total
    magnitude of gradients, which is important when using high penalty
    weights in confidence-based losses.
    """

    def __init__(self, model, loss, metrics, optimizer, device='cpu',
                 verbose=True, use_amp=False, scaler=None, max_grad_norm=5.0):
        super().__init__(model, loss, metrics, optimizer, device, verbose, use_amp, scaler)
        self.max_grad_norm = max_grad_norm

    def batch_update(self, x, y, batch_count=None):
        """Override batch_update to add gradient clipping."""
        self.optimizer.zero_grad()

        # Forward pass (with or without AMP)
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
        else:
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # Gradient clipping with AMP
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss, prediction


class PredictionDataset(Dataset):
    """
    Dataset for generating predictions with same preprocessing as training.

    Uses imageio + float32 to match CustomDataset behavior in dataloader.py.
    Transform includes ToTensor and optional dataset-specific normalization.
    """
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        # Load image using imageio (same as training)
        image = imageio.imread(str(self.image_paths[index]))
        image = np.asarray(image, dtype='float32')  # Keep [0, 255] range!

        # Apply transform (ToTensor + optional normalization from dataset config)
        if self.transform:
            image = self.transform(image)

        return image, self.image_paths[index].name

    def __len__(self):
        return len(self.image_paths)


def compute_penalty_schedule(epoch, max_epochs):
    """
    Compute linearly ramped penalty weights for confidence-based losses.

    Implements curriculum learning by starting with gentle penalties and
    gradually increasing to full penalties over the first 2/3 of training,
    then maintaining full penalties for the final 1/3.

    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total number of epochs

    Returns:
        tuple: (uncertain_correct_penalty, uncertain_wrong_penalty, confident_wrong_penalty)
    """
    # Reach max penalties at 2/3 of training
    ramp_epochs = int(max_epochs * 2 / 3)

    # Compute progress (0.0 at start, 1.0 at 2/3 mark, stays 1.0 after)
    progress = min(epoch / max(ramp_epochs - 1, 1), 1.0)

    # Linear ramp from start to end values
    # Start: gentle penalties (early training, model is learning basics)
    # 2/3 mark: full penalties (enforce calibration for final 1/3)
    uncertain_correct = 1.0 + progress * 0.5   # 1.0 → 1.5
    uncertain_wrong = 5.0 + progress * 10.0     # 5.0 → 15.0
    confident_wrong = 15.0 + progress * 35.0    # 15.0 → 50.0

    return uncertain_correct, uncertain_wrong, confident_wrong


def run_training_iteration(
    session_path,
    dataset_config,
    training_config,
    model,
    device,
    train_loss,
    val_loss,
    metrics,
    optimizer,
    iteration='latest',
    visualize=True,
    use_lr_schedule=True
):
    """
    Run complete training iteration workflow.

    This function handles:
    1. Finding the iteration to train
    2. Creating dataloaders
    3. Loading previous model weights (if available)
    4. Training the model
    5. Generating predictions
    6. Calculating and saving metrics
    7. Visualizing results (optional)
    8. Creating next iteration

    Args:
        session_path: Path to session directory
        dataset_config: Dataset configuration dict
        training_config: Training configuration dict
        model: PyTorch model
        device: Device (cuda/cpu)
        train_loss: Training loss function
        val_loss: Validation loss function
        metrics: List of metric functions
        optimizer: Optimizer
        iteration: Iteration to train. Can be:
            - 'latest': Use latest iteration (default)
            - 'current': Same as 'latest'
            - int: Specific iteration number (e.g., 0, 1, 2)
        visualize: Whether to show visualization plots (default True)
        use_lr_schedule: Whether to use warmup-peak-decay LR schedule (default True).
            If False, uses constant learning rate from optimizer.

    Returns:
        dict: Training result with keys:
            - iteration: Iteration number trained
            - success: Overall success status
            - best_miou: Best mIoU achieved
            - pixel_accuracy: Pixel accuracy on validation set
            - train_loss: Final training loss
            - val_loss: Final validation loss
            - num_predictions: Number of predictions generated
            - next_iteration: Next iteration number created
            - message: Status message
    """
    print(f"{'='*60}")
    print(f"TRAINING ITERATION WORKFLOW")
    print(f"{'='*60}\n")

    session_path = Path(session_path)

    # ============================================================
    # STEP 1: Find Iteration to Train
    # ============================================================
    iteration_dirs = sorted([d for d in session_path.glob('iteration_*') if d.is_dir()])

    if not iteration_dirs:
        return {
            'iteration': None,
            'success': False,
            'best_miou': 0.0,
            'pixel_accuracy': 0.0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'num_predictions': 0,
            'next_iteration': None,
            'message': 'No iterations found in session'
        }

    iterations = [int(d.name.split('_')[1]) for d in iteration_dirs]

    # Determine which iteration to use
    if iteration == 'latest' or iteration == 'current':
        current_iter = max(iterations)
    elif isinstance(iteration, int):
        if iteration not in iterations:
            return {
                'iteration': iteration,
                'success': False,
                'best_miou': 0.0,
                'pixel_accuracy': 0.0,
                'train_loss': 0.0,
                'val_loss': 0.0,
                'num_predictions': 0,
                'next_iteration': None,
                'message': f'Iteration {iteration} does not exist. Available: {iterations}'
            }
        current_iter = iteration
    else:
        return {
            'iteration': None,
            'success': False,
            'best_miou': 0.0,
            'pixel_accuracy': 0.0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'num_predictions': 0,
            'next_iteration': None,
            'message': f'Invalid iteration parameter: {iteration}'
        }

    print(f"Training Iteration: {current_iter}")
    print(f"Session: {session_path}\n")

    iter_path = session_path / f'iteration_{current_iter}'
    masks_dir = iter_path / 'masks'
    models_dir = iter_path / 'models'
    predictions_dir = iter_path / 'predictions'

    # ============================================================
    # STEP 2: Prepare Data Loaders
    # ============================================================
    print(f"{'='*60}")
    print(f"STEP 1: Preparing Data Loaders")
    print(f"{'-'*60}\n")

    from src.training.dataloader import create_dataloaders

    train_loader, val_loader, train_images, base_transform, class_pixel_counts = create_dataloaders(
        dataset_config=dataset_config,
        training_config=training_config,
        masks_dir=masks_dir
    )

    # ============================================================
    # STEP 3: Load Previous Model Weights (if available)
    # ============================================================
    print(f"{'='*60}")
    print(f"STEP 2: Load Previous Model Weights")
    print(f"{'-'*60}\n")

    if current_iter > 0:
        prev_model_path = session_path / f'iteration_{current_iter - 1}' / 'models' / 'best_model.pth'
        if prev_model_path.exists():
            print(f"Loading weights from iteration {current_iter - 1}...")
            model.load_state_dict(torch.load(prev_model_path))
            print(f"✓ Weights loaded from {prev_model_path}\n")
        else:
            print(f"⚠ No previous model found, starting from ImageNet weights\n")
    else:
        print(f"Iteration 0: Starting from ImageNet weights\n")

    # ============================================================
    # STEP 4: Train Model
    # ============================================================
    print(f"{'='*60}")
    print(f"STEP 3: Training Model")
    print(f"{'-'*60}\n")

    # Enable cuDNN benchmark for faster training (RTX 4090 optimization)
    torch.backends.cudnn.benchmark = True
    print(f"✓ cuDNN benchmark enabled (optimizing for fixed input size)")

    # Enable mixed precision training (AMP) for faster training
    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"✓ Mixed precision (AMP) ENABLED (fp16 for speed, fp32 for stability)")
    print(f"✓ Gradient clipping enabled (max_norm=5.0) for training stability")
    print()

    # Create train and validation epochs (with AMP support + gradient clipping)
    train_epoch = TrainEpochWithGradClip(
        model,
        loss=train_loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
        use_amp=use_amp,
        scaler=scaler,
        max_grad_norm=5.0  # Gradient clipping for training stability
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=val_loss,
        metrics=metrics,
        device=device,
        verbose=True,
        use_amp=use_amp
    )

    # Training loop
    num_epochs = training_config['training']['num_epochs']
    best_miou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []

    print(f"Training for {num_epochs} epochs...")

    if use_lr_schedule:
        print(f"Learning rate schedule (Warmup-Peak-Decay):")
        print(f"  Epochs 1-100: lr = 0.0001 (warmup)")
        print(f"  Epochs 101-200: lr = 0.001 (peak)")
        print(f"  Epochs 201-{num_epochs}: lr = 0.0001 (decay)")
    else:
        print(f"Learning rate: {optimizer.param_groups[0]['lr']} (constant)")

    # Check if using confidence-based loss with penalty scheduling
    if hasattr(train_loss, 'uncertain_correct_penalty'):
        print(f"\nPenalty schedule (Linear curriculum learning):")
        start_uc, start_uw, start_cw = compute_penalty_schedule(0, num_epochs)
        end_uc, end_uw, end_cw = compute_penalty_schedule(num_epochs - 1, num_epochs)
        print(f"  Start: uncertain_correct={start_uc:.1f}, uncertain_wrong={start_uw:.1f}, confident_wrong={start_cw:.1f}")
        print(f"  End:   uncertain_correct={end_uc:.1f}, uncertain_wrong={end_uw:.1f}, confident_wrong={end_cw:.1f}")
        print(f"  Confidence threshold: {train_loss.confidence_threshold}")

    print(f"\nDevice: {device}\n")

    for epoch in range(num_epochs):
        # Learning rate schedule: Warmup-Peak-Decay (0.0001 → 0.001 → 0.0001)
        if use_lr_schedule:
            if epoch == 0:
                # Warmup: Set initial learning rate to 0.0001
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001
                print(f"→ Learning rate set to 0.0001 (warmup: epochs 1-100)")
            elif epoch == 100:
                # Peak: Increase learning rate to 0.001
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
                print(f"→ Learning rate increased to 0.001 (peak: epochs 101-200)")
            elif epoch == 200:
                # Decay: Reduce learning rate to 0.0001
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001
                print(f"→ Learning rate reduced to 0.0001 (decay: epochs 201-{num_epochs})")

        current_lr = optimizer.param_groups[0]['lr']

        # Update penalty schedule for confidence-based losses (DWCCE, DWCDL)
        # Check if loss has penalty attributes
        if hasattr(train_loss, 'uncertain_correct_penalty'):
            uc, uw, cw = compute_penalty_schedule(epoch, num_epochs)
            train_loss.uncertain_correct_penalty = uc
            train_loss.uncertain_wrong_penalty = uw
            train_loss.confident_wrong_penalty = cw

            # Print penalty updates at key epochs
            if epoch == 0:
                print(f"→ Penalty schedule initialized (linear ramp over {num_epochs} epochs)")
                print(f"   Start: uncertain_correct={uc:.1f}, uncertain_wrong={uw:.1f}, confident_wrong={cw:.1f}")
            elif epoch == num_epochs - 1:
                print(f"→ Final penalties reached: uncertain_correct={uc:.1f}, uncertain_wrong={uw:.1f}, confident_wrong={cw:.1f}")

        print(f"Epoch {epoch + 1}/{num_epochs} (lr={current_lr})")

        # Train
        train_logs = train_epoch.run(train_loader)

        # Validate
        valid_logs = valid_epoch.run(val_loader)

        # Track metrics (dynamically detect loss key)
        # Get train loss - find the key that contains 'loss' (excluding val metrics)
        train_loss_key = [k for k in train_logs.keys() if 'loss' in k.lower()][0]
        train_losses.append(train_logs[train_loss_key])

        # Get val loss - find the key that contains 'loss'
        val_loss_key = [k for k in valid_logs.keys() if 'loss' in k.lower()][0]
        val_losses.append(valid_logs[val_loss_key])

        # Get IoU metric
        val_ious.append(valid_logs['miou'])

        # Save best model based on HIGHEST validation mIoU
        if valid_logs['miou'] > best_miou:
            best_miou = valid_logs['miou']
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), models_dir / 'best_model.pth')
            print(f"→ Best model saved! mIoU: {best_miou:.4f}, Val Loss: {valid_logs[val_loss_key]:.4f}")

        print()

    print(f"✓ Training complete!")
    print(f"✓ Best mIoU: {best_miou:.4f}\n")

    # Load best model for predictions
    model.load_state_dict(torch.load(models_dir / 'best_model.pth'))

    # ============================================================
    # STEP 5: Generate Predictions
    # ============================================================
    print(f"{'='*60}")
    print(f"STEP 4: Generating Predictions")
    print(f"{'-'*60}\n")

    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Create prediction dataset using base preprocessing (NO augmentation!)
    # Uses imageio + float32 WITHOUT /255 normalization → [0, 255] range
    # IMPORTANT: Uses base_transform (no random flips/augmentation)
    pred_dataset = PredictionDataset(
        image_paths=train_images,
        transform=base_transform  # NO augmentation for predictions!
    )
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Verify we're using the same images as training
    print(f"\n[Image Path Verification]")
    print(f"Training uses {len(train_images)} images from: {train_images[0].parent}")
    print(f"Prediction uses {len(pred_dataset.image_paths)} images from: {pred_dataset.image_paths[0].parent}")
    print(f"First training image: {train_images[0].name}")
    print(f"First prediction image: {pred_dataset.image_paths[0].name}")
    if train_images == pred_dataset.image_paths:
        print(f"✓ VERIFIED: Training and prediction use IDENTICAL image lists")
    else:
        print(f"✗ WARNING: Image lists differ!")
    print()

    model.eval()
    with torch.no_grad():
        for i, (images, img_names) in enumerate(tqdm(pred_loader, desc="Generating predictions")):
            images = images.to(device)

            # Debug: Verify preprocessing and filenames match (only first image)
            if i == 0:
                print(f"\n[Preprocessing Verification]")
                print(f"Prediction input range: [{images.min().item():.1f}, {images.max().item():.1f}]")
                print(f"Expected: [0.0, 255.0] (same as training)")

                # Cross-check with training dataloader
                train_sample, _ = train_loader.dataset[0]
                train_sample = train_sample.to(device)
                print(f"Training input range: [{train_sample.min().item():.1f}, {train_sample.max().item():.1f}]")

                # Verify tensors are identical
                pred_sample = images[0]
                if torch.allclose(pred_sample, train_sample, rtol=1e-5):
                    print(f"✓ VERIFIED: Preprocessing produces IDENTICAL tensors")
                else:
                    diff = (pred_sample - train_sample).abs().max().item()
                    print(f"✗ WARNING: Tensors differ by max {diff:.6f}")

                # Verify filename preservation
                original_path = train_images[0]
                predicted_name = img_names[0]
                if original_path.name == predicted_name:
                    print(f"✓ VERIFIED: Filename preserved: {original_path.name} → {predicted_name}")
                else:
                    print(f"✗ WARNING: Filename mismatch: {original_path.name} vs {predicted_name}")
                print()

            # Predict
            prediction = model(images)
            prediction_mask = prediction.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)

            # Save prediction as PNG (annotation tool expects .png files)
            # Get stem (filename without extension) and always save as .png
            img_stem = Path(img_names[0]).stem
            pred_path = predictions_dir / f"{img_stem}.png"
            Image.fromarray(prediction_mask).save(pred_path)

    num_predictions = len(train_images)
    print(f"✓ Generated {num_predictions} predictions\n")

    # ============================================================
    # STEP 6: Calculate Metrics
    # ============================================================
    print(f"{'='*60}")
    print(f"STEP 5: Calculate Metrics")
    print(f"{'-'*60}\n")

    # Calculate detailed per-class metrics on validation set
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.flatten())
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Filter out ignore_index
    num_classes = dataset_config['classes']['num_classes']
    ignore_index = dataset_config['classes']['ignore_index']
    class_names = dataset_config['classes']['names']
    valid_mask = all_targets != ignore_index
    all_preds_filtered = all_preds[valid_mask]
    all_targets_filtered = all_targets[valid_mask]

    # Pixel accuracy
    pixel_accuracy = (all_preds_filtered == all_targets_filtered).mean()

    # Compute confusion matrix
    cm = confusion_matrix(all_targets_filtered, all_preds_filtered, labels=list(range(num_classes)))

    # Extract per-class TP, FP, FN
    tp = np.diag(cm)  # True Positives for each class
    fn = cm.sum(axis=1) - tp  # False Negatives
    fp = cm.sum(axis=0) - tp  # False Positives

    # Compute precision, recall, and F1-score for each class
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_targets_filtered, all_preds_filtered, average=None, labels=list(range(num_classes)), zero_division=0
    )

    # Compute per-class IoU
    iou_per_class = tp / (tp + fp + fn + 1e-8)  # Avoid division by zero
    mean_iou = np.mean(iou_per_class)

    # Print summary metrics
    print(f"Validation Metrics Summary:")
    print(f"  Mean IoU:        {mean_iou:.4f}")
    print(f"  Mean Precision:  {np.mean(precision):.4f}")
    print(f"  Mean Recall:     {np.mean(recall):.4f}")
    print(f"  Mean F1-Score:   {np.mean(f1_score):.4f}")
    print(f"  Pixel Accuracy:  {pixel_accuracy:.4f}")
    print(f"  Final Val Loss:  {val_losses[-1]:.4f}")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}\n")

    # Print per-class metrics
    print(f"Per-Class Metrics:")
    for i in range(num_classes):
        print(f"  Class {i} ({class_names[i]}): "
              f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, "
              f"F1={f1_score[i]:.4f}, IoU={iou_per_class[i]:.4f}")
    print()

    # Save detailed metrics to CSV
    metrics_history_file = session_path / 'metrics_history.csv'

    # Build column names for CSV
    base_columns = ['iteration', 'miou', 'mean_precision', 'mean_recall', 'mean_f1',
                    'pixel_accuracy', 'train_loss', 'val_loss',
                    'total_tp', 'total_fp', 'total_fn',
                    'total_annotated_pixels']

    # Add per-class annotated pixel columns
    annotated_pixel_columns = [f'class_{i}_annotated_pixels' for i in range(num_classes)]

    per_class_columns = []
    for i in range(num_classes):
        per_class_columns.extend([
            f'class_{i}_tp', f'class_{i}_fp', f'class_{i}_fn',
            f'class_{i}_precision', f'class_{i}_recall',
            f'class_{i}_f1', f'class_{i}_iou'
        ])

    all_columns = base_columns + annotated_pixel_columns + per_class_columns

    if metrics_history_file.exists():
        metrics_df = pd.read_csv(metrics_history_file)
    else:
        metrics_df = pd.DataFrame(columns=all_columns)

    # Build new row
    new_row = {
        'iteration': current_iter,
        'miou': mean_iou,
        'mean_precision': np.mean(precision),
        'mean_recall': np.mean(recall),
        'mean_f1': np.mean(f1_score),
        'pixel_accuracy': pixel_accuracy,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'total_tp': int(tp.sum()),
        'total_fp': int(fp.sum()),
        'total_fn': int(fn.sum()),
        'total_annotated_pixels': int(class_pixel_counts.sum())
    }

    # Add per-class annotated pixel counts
    for i in range(num_classes):
        new_row[f'class_{i}_annotated_pixels'] = int(class_pixel_counts[i])

    # Add per-class metrics
    for i in range(num_classes):
        new_row[f'class_{i}_tp'] = int(tp[i])
        new_row[f'class_{i}_fp'] = int(fp[i])
        new_row[f'class_{i}_fn'] = int(fn[i])
        new_row[f'class_{i}_precision'] = precision[i]
        new_row[f'class_{i}_recall'] = recall[i]
        new_row[f'class_{i}_f1'] = f1_score[i]
        new_row[f'class_{i}_iou'] = iou_per_class[i]

    metrics_df.loc[len(metrics_df)] = new_row
    metrics_df.to_csv(metrics_history_file, index=False)

    print(f"✓ Metrics saved to {metrics_history_file}\n")

    # ============================================================
    # STEP 7: Create Next Iteration
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 7: Create Next Iteration")
    print(f"{'-'*60}\n")

    next_iter = current_iter + 1
    next_iter_path = session_path / f'iteration_{next_iter}'

    # Create directory structure
    (next_iter_path / 'annotations').mkdir(parents=True, exist_ok=True)
    (next_iter_path / 'masks').mkdir(parents=True, exist_ok=True)
    (next_iter_path / 'models').mkdir(parents=True, exist_ok=True)
    (next_iter_path / 'predictions').mkdir(parents=True, exist_ok=True)

    print(f"✓ Created iteration_{next_iter} structure")

    # Copy annotations from current iteration to next
    current_annotations_dir = iter_path / 'annotations'
    next_annotations_dir = next_iter_path / 'annotations'

    annotation_files = list(current_annotations_dir.glob('*.json'))
    for json_file in tqdm(annotation_files, desc="Copying annotations"):
        shutil.copy(json_file, next_annotations_dir / json_file.name)

    print(f"✓ Copied {len(annotation_files)} annotations to iteration_{next_iter}\n")

    # ============================================================
    # Summary
    # ============================================================
    print(f"{'='*60}")
    print(f"ITERATION {current_iter} COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults:")
    print(f"  Mean IoU:        {mean_iou:.4f}")
    print(f"  Mean Precision:  {np.mean(precision):.4f}")
    print(f"  Mean Recall:     {np.mean(recall):.4f}")
    print(f"  Mean F1-Score:   {np.mean(f1_score):.4f}")
    print(f"  Pixel Accuracy:  {pixel_accuracy:.4f}")
    print(f"  Model saved:     {models_dir / 'best_model.pth'}")
    print(f"  Predictions:     {num_predictions} files")
    print(f"\nNext Steps:")
    print(f"  1. Run Cell 4 to annotate iteration {next_iter}")
    print(f"  2. Review predictions (overlay) to find uncertain areas")
    print(f"  3. Add/refine annotation points")
    print(f"  4. Run Cell 5 to train iteration {next_iter}")
    print(f"\nActive Learning Loop: Cell 4 → Cell 5 → Cell 4 → Cell 5...")
    print(f"{'='*60}")

    return {
        'iteration': current_iter,
        'success': True,
        'mean_iou': mean_iou,
        'mean_precision': np.mean(precision),
        'mean_recall': np.mean(recall),
        'mean_f1': np.mean(f1_score),
        'pixel_accuracy': pixel_accuracy,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'num_predictions': num_predictions,
        'next_iteration': next_iter,
        'message': 'Training iteration completed successfully'
    }
