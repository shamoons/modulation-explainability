# src/utils/wandb_utils.py
import wandb


def log_weights_to_wandb(model, epoch):
    """
    Log model weights to WandB for each epoch.

    Args:
        model (torch.nn.Module): The model whose weights are to be logged.
        epoch (int): The current epoch number.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Log weights for each layer to WandB as histograms
            wandb.log({f"weights/{name}_epoch_{epoch}": wandb.Histogram(param.cpu().data.numpy())})
