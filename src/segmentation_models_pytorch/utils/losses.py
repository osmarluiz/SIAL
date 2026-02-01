import torch
import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
    
class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
       
        y_pr = self.activation(y_pr) #.squeeze(1)

        #print(f"Prediction shape: {y_pr.shape}")  # Debugging
        #print(f"Ground Truth shape: {y_gt.shape}")  # Debugging

        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index)
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass

class DynamicWeightedConfidenceDiceLoss(base.Loss):
    __name__ = "dynamic_weighted_confidence_dice_loss"

    def __init__(self, eps=1., beta=1., amplification_factor=2.0, confidence_threshold=0.8, correctness_threshold=0.5, activation=None, ignore_channels=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.amplification_factor = amplification_factor  # Factor to amplify confident incorrect predictions
        self.confidence_threshold = confidence_threshold  # Confidence threshold for amplification
        self.correctness_threshold = correctness_threshold  # Correctness threshold for amplification
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # Apply activation to get probabilities
        y_pr = self.activation(y_pr).squeeze(1)
        
        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index).float()
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        # Step 1: Calculate confidence as distance from 0.5
        confidence = torch.abs(y_pr - 0.5) * 2  # High values for high confidence

        # Step 2: Calculate correctness as agreement with ground truth
        correctness = torch.abs(y_pr - y_gt.float())  # High values for incorrect predictions

        # Step 3: Calculate initial weights
        weights = 1 - confidence * (1 - correctness)

        # Step 4: Amplify weights for confident, incorrect predictions
        weights = torch.where(
            (confidence > self.confidence_threshold) & (correctness > self.correctness_threshold),
            weights * self.amplification_factor,
            weights
        )

        # Normalize weights to avoid instability
        weights = weights / (weights.mean() + 1e-8)

        # Calculate the weighted Dice loss
        intersection = (weights * y_pr * y_gt).sum()
        denominator = (weights * (y_pr + y_gt)).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (denominator + self.eps)
        
        return dice_loss
    
class DynamicWeightedConfidenceMulticlassDiceLoss(base.Loss):
    __name__ = "dynamic_weighted_confidence_multiclass_dice_loss"

    def __init__(self, eps=1., beta=1., amplification_factor=2.0, confidence_threshold=0.8, 
                 correctness_threshold=0.5, activation="softmax", ignore_channels=None, 
                 ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.amplification_factor = amplification_factor
        self.confidence_threshold = confidence_threshold
        self.correctness_threshold = correctness_threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, C, H, W) - Model predictions (logits)
            y_gt: (B, H, W) - Ground truth labels (integer values per pixel)
        """
        # Apply activation to obtain probabilities
        y_pr = self.activation(y_pr)  # Shape: (B, C, H, W)

        #print("Unique values in y_gt:", torch.unique(y_gt))  # Debugging
        assert torch.all((y_gt >= 0) & (y_gt < y_pr.shape[1]) | (y_gt == self.ignore_index)), \
            f"y_gt contains invalid class indices! Expected [0-{y_pr.shape[1]-1}], but got {torch.unique(y_gt)}"

        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index).float()  # Shape: (B, H, W), False for class 5
            valid_mask = valid_mask.unsqueeze(1)  # Reshape to (B, 1, H, W) for broadcasting
            y_pr = y_pr * valid_mask  # Ignore class 5 in predictions

        # Convert y_gt to class probabilities (without invalid indices)
        y_gt_clamped = y_gt.clone()
        y_gt_clamped[y_gt == self.ignore_index] = 0  # Set ignored pixels to class 0 for scatter

        y_gt_probs = torch.zeros_like(y_pr)  # (B, C, H, W)
        y_gt_probs.scatter_(1, y_gt_clamped.unsqueeze(1), 1)  # Convert indices to class-wise probabilities

        # Apply valid_mask to remove influence of ignored pixels
        y_gt_probs = y_gt_probs * valid_mask

        # Step 1: Compute confidence (distance from uniform probability)
        confidence = torch.abs(y_pr - 1 / y_pr.shape[1]) * 2

        # Step 2: Compute correctness using class probabilities
        correctness = torch.abs(y_pr - y_gt_probs)

        # Step 3: Compute initial weights
        weights = 1 - confidence * (1 - correctness)

        # Step 4: Amplify weights for confident, incorrect predictions
        weights = torch.where(
            (confidence > self.confidence_threshold) & (correctness > self.correctness_threshold),
            weights * self.amplification_factor,
            weights
        )

        # Normalize weights to avoid instability
        weights = weights / (weights.mean(dim=[2, 3], keepdim=True) + 1e-8)

        # Step 5: Compute per-class Dice loss
        intersection = (weights * y_pr * y_gt_probs).sum(dim=[2, 3])
        denominator = (weights * (y_pr + y_gt_probs)).sum(dim=[2, 3])
        dice_per_class = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        # Step 6: Average across classes
        dice_loss = dice_per_class.mean()

        return dice_loss


class DWCDL(base.Loss):
    """
    Dynamic Weighted Confidence Dice Loss (Pure Error-Based).

    Weights pixels based purely on error patterns, without class balancing.
    Four scenarios based on confidence and correctness:
    - Confident Correct: weight = 1.0 (baseline, doing well)
    - Uncertain Correct: weight = uncertain_correct_penalty (needs more confidence)
    - Confident Wrong: weight = confident_wrong_penalty (CRITICAL - overconfident mistake)
    - Uncertain Wrong: weight = uncertain_wrong_penalty (wrong but at least uncertain)

    Weights are dynamically computed every forward pass based on current predictions.
    """
    __name__ = "dwcdl_loss"

    def __init__(self,
                 eps=1.,
                 confidence_threshold=0.8,
                 uncertain_correct_penalty=2.0,
                 uncertain_wrong_penalty=4.0,
                 confident_wrong_penalty=10.0,
                 activation=None,
                 ignore_index=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.confidence_threshold = confidence_threshold
        self.uncertain_correct_penalty = uncertain_correct_penalty
        self.uncertain_wrong_penalty = uncertain_wrong_penalty
        self.confident_wrong_penalty = confident_wrong_penalty
        self.activation = Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # Apply activation to get probabilities
        y_pr = self.activation(y_pr).squeeze(1)

        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            mask = (y_gt != self.ignore_index).float()
            y_pr = y_pr * mask
            y_gt = y_gt * mask

        # Determine confidence: is prediction above threshold?
        is_confident = (y_pr > self.confidence_threshold)

        # Determine correctness: does thresholded prediction match ground truth?
        is_correct = ((y_pr > 0.5) == (y_gt > 0.5))

        # Initialize all weights at baseline
        weights = torch.ones_like(y_gt, dtype=torch.float32)

        # Scenario 1: Confident + Correct → 1.0 (baseline, already initialized)

        # Scenario 2: Uncertain + Correct → direct penalty
        mask_unc_correct = ~is_confident & is_correct
        weights[mask_unc_correct] = self.uncertain_correct_penalty

        # Scenario 3: Confident + Wrong → CRITICAL (direct penalty)
        mask_conf_wrong = is_confident & ~is_correct
        weights[mask_conf_wrong] = self.confident_wrong_penalty

        # Scenario 4: Uncertain + Wrong → direct penalty
        mask_unc_wrong = ~is_confident & ~is_correct
        weights[mask_unc_wrong] = self.uncertain_wrong_penalty

        # Normalize weights to avoid instability
        weights = weights / (weights.mean() + 1e-8)

        # Calculate the weighted Dice loss
        intersection = (weights * y_pr * y_gt).sum()
        denominator = (weights * (y_pr + y_gt)).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        return dice_loss


class DWCDLSimple(base.Loss):
    """
    Dynamic Weighted Confidence Dice Loss - Simplified Version.

    Uses fixed weights for 4 categories based on confidence and correctness:
    - Confident Correct (CC): w_cc (default 1.0)
    - Confident Wrong (CW): w_cw (default 10.0) - CRITICAL errors
    - Uncertain Correct (UC): w_uc (default 1.5)
    - Uncertain Wrong (UW): w_uw (default 4.0)

    Confidence is calculated as max(prob, 1-prob), so both high and low
    probabilities are considered confident if above threshold.

    Args:
        confidence_threshold: Threshold for confidence (default 0.85)
        w_cc: Weight for Confident Correct (default 1.0)
        w_cw: Weight for Confident Wrong (default 10.0)
        w_uc: Weight for Uncertain Correct (default 1.5)
        w_uw: Weight for Uncertain Wrong (default 4.0)
        ignore_index: Value to ignore in ground truth (default 2 for sparse masks)
    """
    __name__ = "dwcdl_simple_loss"

    def __init__(self,
                 eps=1.,
                 confidence_threshold=0.85,
                 w_cc=1.0,
                 w_cw=10.0,
                 w_uc=1.5,
                 w_uw=4.0,
                 activation=None,
                 ignore_index=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.confidence_threshold = confidence_threshold
        self.w_cc = w_cc
        self.w_cw = w_cw
        self.w_uc = w_uc
        self.w_uw = w_uw
        self.activation = Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # Apply activation to get probabilities
        y_pr = self.activation(y_pr).squeeze(1)

        # Convert ground truth to float
        y_gt_float = y_gt.float()

        # Create valid mask (exclude ignore_index)
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(y_gt, dtype=torch.float32)

        # Calculate confidence: max(prob, 1-prob)
        # This way, both 0.9 and 0.1 are considered confident
        confidence = torch.max(y_pr, 1 - y_pr)
        is_confident = confidence > self.confidence_threshold

        # Determine correctness: does thresholded prediction match ground truth?
        pred_class = (y_pr > 0.5).float()
        gt_class = (y_gt_float > 0.5).float()
        is_correct = (pred_class == gt_class)

        # Initialize weights
        weights = torch.ones_like(y_gt, dtype=torch.float32)

        # Apply weights for each category
        # CC: Confident + Correct
        mask_cc = is_confident & is_correct
        weights[mask_cc] = self.w_cc

        # CW: Confident + Wrong (CRITICAL)
        mask_cw = is_confident & ~is_correct
        weights[mask_cw] = self.w_cw

        # UC: Uncertain + Correct
        mask_uc = ~is_confident & is_correct
        weights[mask_uc] = self.w_uc

        # UW: Uncertain + Wrong
        mask_uw = ~is_confident & ~is_correct
        weights[mask_uw] = self.w_uw

        # Apply valid mask
        weights = weights * valid_mask
        y_pr_masked = y_pr * valid_mask
        y_gt_masked = y_gt_float * valid_mask

        # Normalize weights to avoid instability
        weights = weights / (weights.mean() + 1e-8)

        # Calculate weighted Dice loss
        intersection = (weights * y_pr_masked * y_gt_masked).sum()
        denominator = (weights * (y_pr_masked + y_gt_masked)).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        return dice_loss


class DWCDLMulticlass(base.Loss):
    """
    Dynamic Weighted Confidence Dice Loss (Multiclass, Pure Error-Based).

    Weights pixels based purely on error patterns, without class balancing.
    Four scenarios based on confidence and correctness:
    - Confident Correct: weight = 1.0 (baseline, doing well)
    - Uncertain Correct: weight = uncertain_correct_penalty (needs more confidence)
    - Confident Wrong: weight = confident_wrong_penalty (CRITICAL - overconfident mistake)
    - Uncertain Wrong: weight = uncertain_wrong_penalty (wrong but at least uncertain)

    Confidence measured as max probability (normalized to uniform baseline).
    Weights are dynamically computed every forward pass based on current predictions.
    """
    __name__ = "dwcdl_multiclass_loss"

    def __init__(self,
                 eps=1.,
                 confidence_threshold=0.8,
                 uncertain_correct_penalty=2.0,
                 uncertain_wrong_penalty=4.0,
                 confident_wrong_penalty=10.0,
                 activation="softmax",
                 ignore_index=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.confidence_threshold = confidence_threshold
        self.uncertain_correct_penalty = uncertain_correct_penalty
        self.uncertain_wrong_penalty = uncertain_wrong_penalty
        self.confident_wrong_penalty = confident_wrong_penalty
        self.activation = Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, C, H, W) - Model predictions (logits)
            y_gt: (B, H, W) - Ground truth labels (integer values per pixel)
        """
        # Apply activation to obtain probabilities
        y_pr = self.activation(y_pr)  # Shape: (B, C, H, W)

        # Validate ground truth
        assert torch.all((y_gt >= 0) & (y_gt < y_pr.shape[1]) | (y_gt == self.ignore_index)), \
            f"y_gt contains invalid class indices! Expected [0-{y_pr.shape[1]-1}], but got {torch.unique(y_gt)}"

        # Mask out ignore_index pixels if specified
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index).float()  # Shape: (B, H, W)
        else:
            valid_mask = torch.ones_like(y_gt, dtype=torch.float32)  # Shape: (B, H, W)

        # Convert y_gt to one-hot encoded probabilities
        y_gt_clamped = y_gt.clone()
        if self.ignore_index is not None:
            y_gt_clamped[y_gt == self.ignore_index] = 0  # Temporary value for scatter

        y_gt_probs = torch.zeros_like(y_pr)  # (B, C, H, W)
        y_gt_probs.scatter_(1, y_gt_clamped.unsqueeze(1), 1)  # One-hot encoding

        # Determine confidence: max probability normalized to uniform baseline
        uniform_prob = 1.0 / y_pr.shape[1]
        max_prob = torch.max(y_pr, dim=1)[0]  # (B, H, W)
        confidence_normalized = (max_prob - uniform_prob) / (1.0 - uniform_prob)  # [0, 1]

        # Determine if confident (above threshold)
        is_confident = (confidence_normalized > self.confidence_threshold)

        # Determine correctness: does predicted class match ground truth?
        predicted_class = torch.argmax(y_pr, dim=1)  # (B, H, W)
        is_correct = (predicted_class == y_gt)

        # Initialize all weights at baseline (broadcast to y_pr shape)
        weights = torch.ones_like(y_pr)  # (B, C, H, W)

        # We need to broadcast the per-pixel weights to all channels
        # Create per-pixel weight (B, H, W) then broadcast to (B, C, H, W)
        pixel_weights = torch.ones_like(y_gt, dtype=torch.float32)  # (B, H, W)

        # Scenario 1: Confident + Correct → 1.0 (baseline, already initialized)

        # Scenario 2: Uncertain + Correct → direct penalty
        mask_unc_correct = ~is_confident & is_correct & (valid_mask > 0)
        pixel_weights[mask_unc_correct] = self.uncertain_correct_penalty

        # Scenario 3: Confident + Wrong → CRITICAL (direct penalty)
        mask_conf_wrong = is_confident & ~is_correct & (valid_mask > 0)
        pixel_weights[mask_conf_wrong] = self.confident_wrong_penalty

        # Scenario 4: Uncertain + Wrong → direct penalty
        mask_unc_wrong = ~is_confident & ~is_correct & (valid_mask > 0)
        pixel_weights[mask_unc_wrong] = self.uncertain_wrong_penalty

        # Apply valid_mask to zero out ignored pixels
        pixel_weights = pixel_weights * valid_mask

        # Broadcast to (B, C, H, W)
        weights = pixel_weights.unsqueeze(1).expand_as(y_pr)

        # Normalize weights
        weights = weights / (weights.mean(dim=[2, 3], keepdim=True) + 1e-8)

        # Compute per-class Dice loss
        intersection = (weights * y_pr * y_gt_probs).sum(dim=[2, 3])
        denominator = (weights * (y_pr + y_gt_probs)).sum(dim=[2, 3])
        dice_per_class = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        # Average across classes
        dice_loss = dice_per_class.mean()

        return dice_loss


class DWCCEMulticlass(base.Loss):
    """
    Dynamic Weighted Confidence CrossEntropy (DWCCE) for multiclass segmentation.

    Applies confidence-based penalty weights to CrossEntropy loss per pixel:
    - Confident & Correct (conf > threshold, pred correct): weight = 1.0
    - Confident & Wrong (conf > threshold, pred wrong): weight = confident_wrong_penalty
    - Uncertain & Correct (conf <= threshold, pred correct): weight = uncertain_correct_penalty
    - Uncertain & Wrong (conf <= threshold, pred wrong): weight = uncertain_wrong_penalty

    Unlike DWCDL which uses Dice loss (global overlap metric), DWCCE uses CrossEntropy
    which naturally operates per-pixel and is less prone to pushing predictions to extremes.

    Args:
        eps: Small constant for numerical stability (default: 1e-7)
        confidence_threshold: Threshold to determine if prediction is confident (default: 0.85)
        uncertain_correct_penalty: Weight for uncertain but correct predictions (default: 1.0, ramps to 1.5)
        uncertain_wrong_penalty: Weight for uncertain and wrong predictions (default: 5.0, ramps to 15.0)
        confident_wrong_penalty: Weight for confident but wrong predictions (default: 15.0, ramps to 50.0)
        activation: Activation to apply to predictions ('softmax', 'sigmoid', or None)
        ignore_index: Class index to ignore in loss computation

    Note: Penalty weights are typically ramped linearly during training (curriculum learning).
          Default values represent starting penalties. Use compute_penalty_schedule() for ramping.
    """
    __name__ = "dwcce_multiclass_loss"

    def __init__(self,
                 eps=1e-7,
                 confidence_threshold=0.85,
                 uncertain_correct_penalty=1.0,
                 uncertain_wrong_penalty=5.0,
                 confident_wrong_penalty=15.0,
                 activation=None,
                 ignore_index=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.confidence_threshold = confidence_threshold
        self.uncertain_correct_penalty = uncertain_correct_penalty
        self.uncertain_wrong_penalty = uncertain_wrong_penalty
        self.confident_wrong_penalty = confident_wrong_penalty
        self.activation = Activation(activation) if activation else None
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, C, H, W) - Model predictions (logits or probabilities)
            y_gt: (B, H, W) - Ground truth labels (integer values per pixel)
        """
        # Apply activation if specified
        if self.activation is not None:
            y_pr = self.activation(y_pr)

        # Get predictions and confidence
        confidence, predictions = torch.max(y_pr, dim=1)  # (B, H, W)

        # Create valid mask (pixels to consider)
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index)
        else:
            valid_mask = torch.ones_like(y_gt, dtype=torch.bool)

        # For ignore_index pixels, temporarily set y_gt to 0 to avoid index errors
        y_gt_safe = y_gt.clone()
        y_gt_safe[~valid_mask] = 0

        # Compute CrossEntropy per pixel
        # Gather probability of ground truth class for each pixel
        p_correct = torch.gather(y_pr, 1, y_gt_safe.unsqueeze(1)).squeeze(1)  # (B, H, W)
        p_correct = torch.clamp(p_correct, min=self.eps, max=1.0)

        # CrossEntropy per pixel: -log(p_correct)
        ce_per_pixel = -torch.log(p_correct)  # (B, H, W)

        # Determine if predictions are correct (only for valid pixels)
        is_correct = (predictions == y_gt) & valid_mask

        # Determine if predictions are confident (only for valid pixels)
        is_confident = (confidence > self.confidence_threshold) & valid_mask

        # Initialize weights (start with 1.0 for confident & correct)
        weights = torch.ones_like(ce_per_pixel)

        # Apply penalty weights based on categories
        # Confident & Correct: weight = 1.0 (default, already set)

        # Confident & Wrong: weight = confident_wrong_penalty
        confident_wrong_mask = is_confident & ~is_correct & valid_mask
        weights[confident_wrong_mask] = self.confident_wrong_penalty

        # Uncertain & Correct: weight = uncertain_correct_penalty
        uncertain_correct_mask = ~is_confident & is_correct & valid_mask
        weights[uncertain_correct_mask] = self.uncertain_correct_penalty

        # Uncertain & Wrong: weight = uncertain_wrong_penalty
        uncertain_wrong_mask = ~is_confident & ~is_correct & valid_mask
        weights[uncertain_wrong_mask] = self.uncertain_wrong_penalty

        # Apply weights and mask
        weighted_loss = ce_per_pixel * weights * valid_mask.float()

        # Return average over valid pixels
        return weighted_loss.sum() / (valid_mask.float().sum() + self.eps)


class DWCBCELoss(base.Loss):
    """
    Dynamic Weighted Confidence Binary Cross Entropy (DWCBCE) for binary segmentation.

    Binary version of DWCCE. Applies confidence-based penalty weights to BCE loss per pixel:
    - Confident & Correct (conf > threshold, pred correct): weight = 1.0
    - Confident & Wrong (conf > threshold, pred wrong): weight = confident_wrong_penalty
    - Uncertain & Correct (conf <= threshold, pred correct): weight = uncertain_correct_penalty
    - Uncertain & Wrong (conf <= threshold, pred wrong): weight = uncertain_wrong_penalty

    For binary segmentation with sparse annotations:
    - Input predictions: (B, 1, H, W) logits or probabilities
    - Ground truth: (B, H, W) with values 0=background, 1=foreground, 2=ignore (unlabeled)
    - Confidence: max(prob, 1-prob) (ranges from 0.5 to 1.0)
    - Correct: (prob > 0.5 and gt == 1) or (prob <= 0.5 and gt == 0)

    Args:
        eps: Small constant for numerical stability (default: 1e-7)
        confidence_threshold: Threshold to determine if prediction is confident (default: 0.85)
        uncertain_correct_penalty: Weight for uncertain but correct predictions (default: 1.0, ramps to 1.5)
        uncertain_wrong_penalty: Weight for uncertain and wrong predictions (default: 5.0, ramps to 15.0)
        confident_wrong_penalty: Weight for confident but wrong predictions (default: 15.0, ramps to 50.0)
        ignore_value: Value in ground truth to ignore (default: 2)
        from_logits: If True, apply sigmoid to convert logits to probabilities (default: True)

    Note: Penalty weights are typically ramped linearly during training (curriculum learning).
          Default values represent starting penalties. Use compute_penalty_schedule() for ramping.
    """
    __name__ = "dwcbce_loss"

    def __init__(self,
                 eps=1e-7,
                 confidence_threshold=0.85,
                 uncertain_correct_penalty=1.0,
                 uncertain_wrong_penalty=5.0,
                 confident_wrong_penalty=15.0,
                 ignore_value=2,
                 from_logits=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.confidence_threshold = confidence_threshold
        self.uncertain_correct_penalty = uncertain_correct_penalty
        self.uncertain_wrong_penalty = uncertain_wrong_penalty
        self.confident_wrong_penalty = confident_wrong_penalty
        self.ignore_value = ignore_value
        self.from_logits = from_logits

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, 1, H, W) or (B, H, W) - Model predictions (logits or probabilities)
            y_gt: (B, H, W) - Ground truth labels (0=background, 1=foreground, 2=ignore)
        """
        # Handle input shapes
        if y_pr.dim() == 4 and y_pr.shape[1] == 1:
            y_pr = y_pr.squeeze(1)  # (B, H, W)

        # Convert to probabilities if from_logits
        if self.from_logits:
            probs = torch.sigmoid(y_pr)  # (B, H, W)
        else:
            probs = y_pr  # Already probabilities

        # Create valid mask (pixels to consider: gt == 0 or gt == 1, not ignore)
        valid_mask = (y_gt != self.ignore_value)

        # Ground truth: replace ignore values with 0 to avoid numerical issues in BCE
        # (these pixels will be masked out anyway, but we need valid values for computation)
        y_gt_safe = y_gt.clone()
        y_gt_safe[~valid_mask] = 0  # Replace ignore (2) with 0
        y_gt_binary = y_gt_safe.float()

        # Compute BCE per pixel
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        probs_clamped = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        bce_per_pixel = -(y_gt_binary * torch.log(probs_clamped) +
                         (1 - y_gt_binary) * torch.log(1 - probs_clamped))

        # Calculate confidence: max(prob, 1-prob)
        # Ranges from 0.5 (maximally uncertain) to 1.0 (maximally confident)
        confidence = torch.max(probs, 1 - probs)  # (B, H, W)

        # Determine if predictions are correct
        # Correct if: (prob > 0.5 and gt == 1) or (prob <= 0.5 and gt == 0)
        pred_foreground = probs > 0.5
        gt_foreground = y_gt == 1
        is_correct = (pred_foreground == gt_foreground) & valid_mask

        # Determine if predictions are confident
        is_confident = (confidence > self.confidence_threshold) & valid_mask

        # Initialize weights (start with 1.0 for confident & correct)
        weights = torch.ones_like(bce_per_pixel)

        # Apply penalty weights based on categories
        # Confident & Correct: weight = 1.0 (default, already set)

        # Confident & Wrong: weight = confident_wrong_penalty
        confident_wrong_mask = is_confident & ~is_correct & valid_mask
        weights[confident_wrong_mask] = self.confident_wrong_penalty

        # Uncertain & Correct: weight = uncertain_correct_penalty
        uncertain_correct_mask = ~is_confident & is_correct & valid_mask
        weights[uncertain_correct_mask] = self.uncertain_correct_penalty

        # Uncertain & Wrong: weight = uncertain_wrong_penalty
        uncertain_wrong_mask = ~is_confident & ~is_correct & valid_mask
        weights[uncertain_wrong_mask] = self.uncertain_wrong_penalty

        # Apply weights and mask
        weighted_loss = bce_per_pixel * weights * valid_mask.float()

        # Return average over valid pixels
        return weighted_loss.sum() / (valid_mask.float().sum() + self.eps)


class FocalLoss(base.Loss):
    """
    Focal Loss for multiclass semantic segmentation.

    Focal loss applies a modulating term to the cross entropy loss to focus learning
    on hard misclassified examples. It helps with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
    - p_t is the model's estimated probability for the true class
    - gamma is the focusing parameter (gamma > 0 reduces loss for well-classified examples)
    - alpha is the weighting factor for class imbalance
    """
    __name__ = "focal_loss"

    def __init__(self, alpha=None, gamma=2.0, ignore_index=None, **kwargs):
        """
        Args:
            alpha: Optional class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (default: 2.0)
            ignore_index: Optional index to ignore in loss computation
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, C, H, W) - Model predictions (logits or probabilities)
            y_gt: (B, H, W) - Ground truth labels (integer values per pixel)
        """
        # Apply softmax to get probabilities if not already applied
        if y_pr.dim() == 4:  # (B, C, H, W)
            log_probs = torch.nn.functional.log_softmax(y_pr, dim=1)
            probs = torch.nn.functional.softmax(y_pr, dim=1)
        else:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {y_pr.dim()}D")

        # Handle ignore_index
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index)  # (B, H, W)
        else:
            valid_mask = torch.ones_like(y_gt, dtype=torch.bool)

        # Gather log probabilities and probabilities for the true class
        # y_gt: (B, H, W), need to unsqueeze to (B, 1, H, W) for gather
        y_gt_clamped = y_gt.clone()
        if self.ignore_index is not None:
            y_gt_clamped[~valid_mask] = 0  # Temporary value for gather

        y_gt_unsqueezed = y_gt_clamped.unsqueeze(1)  # (B, 1, H, W)

        # Gather the log_prob and prob for the true class
        log_pt = torch.gather(log_probs, dim=1, index=y_gt_unsqueezed).squeeze(1)  # (B, H, W)
        pt = torch.gather(probs, dim=1, index=y_gt_unsqueezed).squeeze(1)  # (B, H, W)

        # Compute focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Compute focal loss: -focal_weight * log(pt)
        focal_loss = -focal_weight * log_pt

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.device != y_gt.device:
                self.alpha = self.alpha.to(y_gt.device)
            # Gather alpha for each pixel's true class
            alpha_t = torch.gather(self.alpha.unsqueeze(0).expand(y_gt.shape[0], -1),
                                   dim=1,
                                   index=y_gt_clamped.view(y_gt.shape[0], -1))
            alpha_t = alpha_t.view_as(y_gt)  # (B, H, W)
            focal_loss = alpha_t * focal_loss

        # Apply valid mask (ignore pixels)
        focal_loss = focal_loss * valid_mask.float()

        # Return mean loss
        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            return focal_loss.sum() / num_valid
        else:
            return focal_loss.sum()  # Should be 0


class DWCBCELossSimple(base.Loss):
    """
    Dynamic Weighted Confidence BCE Loss - Simplified Version.

    Simple 4-category weighting with fixed weights (no curriculum learning).
    Designed for ablation studies with clear, interpretable parameters.

    Categories:
    - Confident Correct (CC): conf > threshold, pred matches gt → weight = w_cc
    - Confident Wrong (CW): conf > threshold, pred != gt → weight = w_cw
    - Uncertain Correct (UC): conf <= threshold, pred matches gt → weight = w_uc
    - Uncertain Wrong (UW): conf <= threshold, pred != gt → weight = w_uw

    Where confidence = max(prob, 1-prob), ranging from 0.5 to 1.0

    Args:
        confidence_threshold: Threshold to determine confident vs uncertain (default: 0.85)
        w_cc: Weight for confident correct (default: 1.0, baseline)
        w_cw: Weight for confident wrong (default: 10.0, critical errors)
        w_uc: Weight for uncertain correct (default: 1.5, encourage confidence)
        w_uw: Weight for uncertain wrong (default: 4.0, moderate penalty)
        ignore_value: Value in ground truth to ignore (default: 2)
        from_logits: If True, apply sigmoid to predictions (default: False)
        eps: Small constant for numerical stability (default: 1e-7)
    """
    __name__ = "dwcbce_simple_loss"

    def __init__(self,
                 confidence_threshold=0.85,
                 w_cc=1.0,
                 w_cw=10.0,
                 w_uc=1.5,
                 w_uw=4.0,
                 ignore_value=2,
                 from_logits=False,
                 eps=1e-7,
                 **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.w_cc = w_cc
        self.w_cw = w_cw
        self.w_uc = w_uc
        self.w_uw = w_uw
        self.ignore_value = ignore_value
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, 1, H, W) or (B, H, W) - Model predictions
            y_gt: (B, H, W) - Ground truth (0=bg, 1=fg, 2=ignore)

        Returns:
            Weighted BCE loss (scalar)
        """
        # Handle input shapes
        if y_pr.dim() == 4 and y_pr.shape[1] == 1:
            y_pr = y_pr.squeeze(1)

        # Convert to probabilities if needed
        if self.from_logits:
            probs = torch.sigmoid(y_pr)
        else:
            probs = y_pr

        # Valid mask (exclude ignore pixels)
        valid_mask = (y_gt != self.ignore_value)

        # Safe ground truth (replace ignore with 0 for computation)
        y_gt_safe = y_gt.clone()
        y_gt_safe[~valid_mask] = 0
        y_gt_float = y_gt_safe.float()

        # Compute BCE per pixel
        probs_clamped = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        bce = -(y_gt_float * torch.log(probs_clamped) +
                (1 - y_gt_float) * torch.log(1 - probs_clamped))

        # Compute confidence: max(prob, 1-prob) → [0.5, 1.0]
        confidence = torch.max(probs, 1 - probs)

        # Determine categories
        is_confident = confidence > self.confidence_threshold
        is_correct = ((probs > 0.5) == (y_gt == 1))

        # Assign weights based on 4 categories
        weights = torch.ones_like(bce)

        # Confident Correct
        mask_cc = is_confident & is_correct & valid_mask
        weights[mask_cc] = self.w_cc

        # Confident Wrong (CRITICAL)
        mask_cw = is_confident & ~is_correct & valid_mask
        weights[mask_cw] = self.w_cw

        # Uncertain Correct
        mask_uc = ~is_confident & is_correct & valid_mask
        weights[mask_uc] = self.w_uc

        # Uncertain Wrong
        mask_uw = ~is_confident & ~is_correct & valid_mask
        weights[mask_uw] = self.w_uw

        # Weighted loss
        weighted_bce = bce * weights * valid_mask.float()

        # Return mean over valid pixels
        num_valid = valid_mask.float().sum() + self.eps
        return weighted_bce.sum() / num_valid

    def get_category_counts(self, y_pr, y_gt):
        """
        Utility to get counts per category (for logging/debugging).

        Returns dict with counts: cc, cw, uc, uw
        """
        if y_pr.dim() == 4 and y_pr.shape[1] == 1:
            y_pr = y_pr.squeeze(1)

        probs = torch.sigmoid(y_pr) if self.from_logits else y_pr
        valid_mask = (y_gt != self.ignore_value)
        confidence = torch.max(probs, 1 - probs)

        is_confident = confidence > self.confidence_threshold
        is_correct = ((probs > 0.5) == (y_gt == 1))

        return {
            'cc': (is_confident & is_correct & valid_mask).sum().item(),
            'cw': (is_confident & ~is_correct & valid_mask).sum().item(),
            'uc': (~is_confident & is_correct & valid_mask).sum().item(),
            'uw': (~is_confident & ~is_correct & valid_mask).sum().item(),
            'total': valid_mask.sum().item()
        }


# =============================================================================
# ERROR-WEIGHTED DICE LOSS (EWDL) - Simplified version
# =============================================================================

class EWDLBinary(base.Loss):
    """
    Error-Weighted Dice Loss (EWDL) for binary segmentation.

    A simplified approach that weights pixels based only on correctness:
    - Correct predictions: weight = 1.0
    - Wrong predictions: weight = wrong_penalty (α)

    This removes the complexity of confidence thresholds, leaving only one
    hyperparameter to tune: the penalty for errors.

    Args:
        eps: Small constant for numerical stability (default: 1.0)
        wrong_penalty: Weight multiplier for incorrect predictions (default: 10.0)
        activation: Activation function ('sigmoid' or None)
        ignore_index: Value to ignore in ground truth (default: 2 for sparse masks)
    """
    __name__ = "ewdl_binary_loss"

    def __init__(self,
                 eps=1.0,
                 wrong_penalty=10.0,
                 activation=None,
                 ignore_index=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.wrong_penalty = wrong_penalty
        self.activation = Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, 1, H, W) or (B, H, W) - Model predictions
            y_gt: (B, H, W) - Ground truth (0=background, 1=foreground, ignore_index=ignore)
        """
        # Apply activation to get probabilities
        y_pr = self.activation(y_pr)
        if y_pr.dim() == 4:
            y_pr = y_pr.squeeze(1)  # (B, H, W)

        # Convert ground truth to float
        y_gt_float = y_gt.float()

        # Create valid mask (exclude ignore_index)
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(y_gt, dtype=torch.float32)

        # Determine correctness: does thresholded prediction match ground truth?
        pred_class = (y_pr > 0.5).float()
        gt_class = (y_gt_float > 0.5).float()
        is_correct = (pred_class == gt_class)

        # Assign weights: 1.0 for correct, wrong_penalty for wrong
        weights = torch.ones_like(y_gt, dtype=torch.float32)
        weights[~is_correct] = self.wrong_penalty

        # Apply valid_mask
        weights = weights * valid_mask

        # Clamp y_gt to valid range for Dice computation
        y_gt_clamped = torch.clamp(y_gt_float, 0, 1) * valid_mask

        # Normalize weights to avoid instability
        weights = weights / (weights.sum() + 1e-8) * valid_mask.sum()

        # Calculate weighted Dice loss
        intersection = (weights * y_pr * y_gt_clamped).sum()
        denominator = (weights * (y_pr + y_gt_clamped)).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        return dice_loss


class EWDLMulticlass(base.Loss):
    """
    Error-Weighted Dice Loss (EWDL) for multiclass segmentation.

    A simplified approach that weights pixels based only on correctness:
    - Correct predictions: weight = 1.0
    - Wrong predictions: weight = wrong_penalty (α)

    This removes the complexity of confidence thresholds, leaving only one
    hyperparameter to tune: the penalty for errors.

    Args:
        eps: Small constant for numerical stability (default: 1.0)
        wrong_penalty: Weight multiplier for incorrect predictions (default: 10.0)
        activation: Activation function ('softmax' or None)
        ignore_index: Class index to ignore in loss computation
    """
    __name__ = "ewdl_multiclass_loss"

    def __init__(self,
                 eps=1.0,
                 wrong_penalty=10.0,
                 activation="softmax",
                 ignore_index=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.wrong_penalty = wrong_penalty
        self.activation = Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        """
        Args:
            y_pr: (B, C, H, W) - Model predictions (logits)
            y_gt: (B, H, W) - Ground truth labels (integer values per pixel)
        """
        # Apply activation to obtain probabilities
        y_pr = self.activation(y_pr)  # Shape: (B, C, H, W)

        num_classes = y_pr.shape[1]

        # Create valid mask (exclude ignore_index)
        if self.ignore_index is not None:
            valid_mask = (y_gt != self.ignore_index).float()  # (B, H, W)
        else:
            valid_mask = torch.ones_like(y_gt, dtype=torch.float32)

        # Convert y_gt to one-hot encoding
        y_gt_clamped = y_gt.clone()
        if self.ignore_index is not None:
            y_gt_clamped[y_gt == self.ignore_index] = 0  # Temporary for scatter

        y_gt_onehot = torch.zeros_like(y_pr)  # (B, C, H, W)
        y_gt_onehot.scatter_(1, y_gt_clamped.unsqueeze(1), 1)

        # Determine correctness: does predicted class match ground truth?
        predicted_class = torch.argmax(y_pr, dim=1)  # (B, H, W)
        is_correct = (predicted_class == y_gt)  # (B, H, W)

        # Assign weights: 1.0 for correct, wrong_penalty for wrong
        pixel_weights = torch.ones_like(y_gt, dtype=torch.float32)  # (B, H, W)
        pixel_weights[~is_correct] = self.wrong_penalty

        # Apply valid_mask
        pixel_weights = pixel_weights * valid_mask

        # Normalize weights
        pixel_weights = pixel_weights / (pixel_weights.sum() + 1e-8) * valid_mask.sum()

        # Broadcast to (B, C, H, W)
        weights = pixel_weights.unsqueeze(1).expand_as(y_pr)

        # Apply valid_mask to y_gt_onehot
        y_gt_onehot = y_gt_onehot * valid_mask.unsqueeze(1)

        # Calculate weighted Dice loss per class
        intersection = (weights * y_pr * y_gt_onehot).sum(dim=(0, 2, 3))  # (C,)
        denominator = (weights * (y_pr + y_gt_onehot)).sum(dim=(0, 2, 3))  # (C,)

        dice_per_class = 1 - (2 * intersection + self.eps) / (denominator + self.eps)

        # Average across classes
        dice_loss = dice_per_class.mean()

        return dice_loss