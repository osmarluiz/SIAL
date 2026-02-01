"""
Training setup module for creating model, losses, metrics, and optimizer.
"""
import torch
import segmentation_models_pytorch as smp


def create_model(dataset_config, training_config):
    """
    Create segmentation model from configurations.

    Args:
        dataset_config: Dataset configuration dict
        training_config: Training configuration dict

    Returns:
        model: Initialized segmentation model
    """
    model = smp.Unet(
        encoder_name=training_config['model']['encoder'],
        encoder_weights='imagenet',
        classes=dataset_config['classes']['num_classes'],
        activation='softmax',
        in_channels=dataset_config['image']['channels']
    )

    return model


def create_losses(dataset_config, training_config):
    """
    Create training and validation loss functions from configuration.

    Supports:
    - DWCCEMulticlass: CrossEntropy + confidence penalties + curriculum learning
    - DWCDLMulticlass: Dice Loss + confidence weighting (legacy)

    Args:
        dataset_config: Dataset configuration dict
        training_config: Training configuration dict

    Returns:
        train_loss: Loss function for training
        val_loss: Loss function for validation
    """
    loss_config = training_config.get('loss', {})
    train_config = loss_config.get('train', {})
    loss_name = train_config.get('name', 'DWCCEMulticlass')
    loss_params = train_config.get('params', {}).copy()

    # Add ignore_index from dataset config
    loss_params['ignore_index'] = dataset_config['classes']['ignore_index']

    # Create training loss based on config
    if loss_name == 'DWCCEMulticlass':
        train_loss = smp.utils.losses.DWCCEMulticlass(
            eps=loss_params.get('eps', 1e-7),
            confidence_threshold=loss_params.get('confidence_threshold', 0.95),
            uncertain_correct_penalty=loss_params.get('uncertain_correct_penalty', 1.0),
            uncertain_wrong_penalty=loss_params.get('uncertain_wrong_penalty', 5.0),
            confident_wrong_penalty=loss_params.get('confident_wrong_penalty', 15.0),
            activation=loss_params.get('activation', None),
            ignore_index=loss_params['ignore_index']
        )
    elif loss_name == 'DWCDLMulticlass':
        train_loss = smp.utils.losses.DWCDLMulticlass(
            eps=loss_params.get('eps', 1.0),
            confidence_threshold=loss_params.get('confidence_threshold', 0.8),
            uncertain_correct_penalty=loss_params.get('uncertain_correct_penalty', 2.0),
            uncertain_wrong_penalty=loss_params.get('uncertain_wrong_penalty', 4.0),
            confident_wrong_penalty=loss_params.get('confident_wrong_penalty', 10.0),
            activation=loss_params.get('activation', 'softmax'),
            ignore_index=loss_params['ignore_index']
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    # Validation loss: CrossEntropyLoss
    val_loss = smp.utils.losses.CrossEntropyLoss(
        ignore_index=dataset_config['classes']['ignore_index']
    )

    return train_loss, val_loss


def create_metrics():
    """
    Create evaluation metrics.

    Returns:
        metrics: List of metric functions
    """
    metrics = [smp.utils.metrics.mIoU()]
    return metrics


def create_optimizer(model, training_config):
    """
    Create optimizer from configuration.

    Args:
        model: PyTorch model
        training_config: Training configuration dict

    Returns:
        optimizer: Configured optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['optimizer']['params']['lr']
    )

    return optimizer


def setup_training(dataset_config, training_config):
    """
    Complete training setup from configurations.

    Creates model, losses, metrics, and optimizer in one call.

    Args:
        dataset_config: Dataset configuration dict
        training_config: Training configuration dict

    Returns:
        model: Initialized model on correct device
        device: Device (cuda/cpu)
        train_loss: Training loss function
        val_loss: Validation loss function
        metrics: List of metrics
        optimizer: Configured optimizer
    """
    print("Setting up training components...")

    # Create model
    model = create_model(dataset_config, training_config)
    print(f"✓ Model: {training_config['model']['architecture']} + {training_config['model']['encoder']}")

    # Move to device
    device = training_config['training']['device']
    model = model.to(device)
    print(f"✓ Device: {device}")

    # Create losses
    train_loss, val_loss = create_losses(dataset_config, training_config)
    print(f"✓ Train Loss: DWCDL (confident_wrong=10.0x, uncertain_wrong=4.0x, uncertain_correct=2.0x)")
    print(f"✓ Val Loss: CrossEntropyLoss")

    # Create metrics
    metrics = create_metrics()
    print(f"✓ Metrics: mIoU")

    # Create optimizer
    optimizer = create_optimizer(model, training_config)
    print(f"✓ Optimizer: Adam (lr={training_config['optimizer']['params']['lr']})")

    print("✓ Training setup complete\n")

    return model, device, train_loss, val_loss, metrics, optimizer
