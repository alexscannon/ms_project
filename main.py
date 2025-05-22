import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.models.backbone.vit import VisionTransformer

# Placeholder for your data loading and streaming logic
# You will need to replace this with your actual data loading and splitting.
def get_test_data_stream(config: DictConfig, batch_size: int = 32):
    """
    Generates a test data stream.

    The stream should yield batches of (images, original_labels, is_ood_label).
    - 'images': Batch of input images.
    - 'original_labels': Batch of original class labels (for informational purposes).
    - 'is_ood_label': Batch of binary labels (0 for In-Distribution, 1 for Out-of-Distribution).

    This stream should combine:
    1. The 25% held-out examples from the 80% pre-training classes (is_ood_label = 0).
    2. All examples from the 20% held-out classes (is_ood_label = 1).
    """
    print("Setting up test data stream (using placeholder)...")
    # Example: Simulate a stream of 10 batches.
    # Replace with your actual data loading logic.
    num_batches = 10
    try:
        img_size = config.model.img_size
    except AttributeError:
        print("Warning: config.model.img_size not found, defaulting to 224 for placeholder.")
        img_size = 224 # Default placeholder image size

    for i in range(num_batches):
        # Simulate a batch of images
        images = torch.randn(batch_size, 3, img_size, img_size)

        # Simulate original labels (e.g., if you have 100 classes total)
        original_labels = torch.randint(0, 100, (batch_size,))

        # Simulate OOD labels: mix of IND (0) and OOD (1)
        # In a real scenario, this would come from your data split.
        # For this placeholder, let's make half IND and half OOD if batch_size is even.
        if i % 2 == 0: # Even batches mostly IND
            is_ood_labels = torch.zeros(batch_size, dtype=torch.long)
            is_ood_labels[batch_size//4:] = 1 # Some OOD
        else: # Odd batches mostly OOD
            is_ood_labels = torch.ones(batch_size, dtype=torch.long)
            is_ood_labels[batch_size//4:] = 0 # Some IND

        # Shuffle OOD labels within the batch for more realistic placeholder
        is_ood_labels = is_ood_labels[torch.randperm(batch_size)]

        yield images, original_labels, is_ood_labels
    print("Placeholder data stream finished.")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # Determine device
    if torch.cuda.is_available() and hasattr(config, 'system') and hasattr(config.system, 'device') and 'cuda' in config.system.device:
        device = torch.device(config.system.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load pre-trained model
    print("Loading pre-trained Vision Transformer model...")
    model = VisionTransformer(config=config)
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded and in evaluation mode.")

    # Get the test data stream
    # You might want to make batch_size configurable, e.g., config.data.batch_size_eval
    eval_batch_size = config.get('eval_batch_size', 32)
    test_stream = get_test_data_stream(config, batch_size=eval_batch_size)

    all_ood_scores = []
    all_is_ood_labels = []

    print("Starting OOD detection evaluation...")
    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        for batch_idx, (images, _, is_ood_batch) in enumerate(test_stream):
            images = images.to(device)
            # is_ood_batch remains on CPU for easier collection

            # Get model outputs (logits)
            logits = model(images) # Assuming model(images) returns logits

            # Calculate OOD score (e.g., Max Softmax Probability - MSP)
            # For MSP, a lower score (higher max probability) means more In-Distribution.
            # To make higher scores mean more OOD, we use -max_prob.
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            ood_scores_batch = -max_probs # Higher score means more OOD

            all_ood_scores.append(ood_scores_batch.cpu())
            all_is_ood_labels.append(is_ood_batch.cpu()) # Ensure labels are on CPU

            if (batch_idx + 1) % 1 == 0: # Log progress (e.g. every batch for placeholder)
                 print(f"Processed batch {batch_idx + 1}")


    if not all_ood_scores:
        print("No data processed from the stream. Skipping metrics calculation.")
        return

    all_ood_scores = torch.cat(all_ood_scores)
    all_is_ood_labels = torch.cat(all_is_ood_labels)

    # Calculate OOD metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Ensure there are both IND and OOD samples for metrics calculation
        if len(torch.unique(all_is_ood_labels)) < 2:
            print("Warning: Test data contains only one class (all IND or all OOD). Cannot compute AUROC/AUPR meaningfully.")
            if len(all_is_ood_labels) > 0:
                 print(f"Unique labels found: {torch.unique(all_is_ood_labels)}")
        else:
            auroc = roc_auc_score(all_is_ood_labels.numpy(), all_ood_scores.numpy())

            # AUPR with OOD samples as positive class
            aupr_in = average_precision_score(all_is_ood_labels.numpy(), all_ood_scores.numpy())

            # AUPR with IND samples as positive class
            # For this, invert labels (1 for IND, 0 for OOD) and scores (higher score for IND)
            aupr_out = average_precision_score(1 - all_is_ood_labels.numpy(), -all_ood_scores.numpy())

            print("\nOOD Detection Performance:")
            print(f"  AUROC: {auroc:.4f}")
            print(f"  AUPR (OOD as positive): {aupr_in:.4f}")
            print(f"  AUPR (IND as positive / OOD as negative): {aupr_out:.4f}")

    except ImportError:
        print("scikit-learn not found. Cannot calculate AUROC/AUPR.")
        print("Raw OOD scores and labels collected.")
        # You can save these or process them differently:
        # print("Number of scores:", len(all_ood_scores))
        # print("Number of labels:", len(all_is_ood_labels))
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print("This can happen if all test samples are predicted as one class or if data is malformed.")
        print(f"Unique labels: {torch.unique(all_is_ood_labels)}")
        print(f"Number of scores: {len(all_ood_scores)}, Number of labels: {len(all_is_ood_labels)}")


if __name__ == "__main__":
    main()