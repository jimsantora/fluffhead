# src/training.py
import os
from tqdm import tqdm
import torch
from diffusers.optimization import get_scheduler
import wandb
from pathlib import Path


def train_loop(pipeline, optimizer, dataloader, config):
    """
    Main training loop for fine-tuning a diffusion model on MPS (Apple Silicon).

    Args:
        pipeline: Diffusers pipeline to train.
        optimizer: PyTorch optimizer.
        dataloader: DataLoader providing the training data.
        config: Dictionary containing training configuration.
    """
    # Set up device for MPS (Apple GPU)
    device = torch.device("mps" if torch.has_mps else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scheduler for learning rate
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config["max_train_steps"],
    )

    # Initialize Weights & Biases logging
    wandb.init(project="diffusion-training", config=config)

    # Move pipeline to MPS
    pipeline.to(device)
    pipeline.unet.train()

    # Training loop
    global_step = 0
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)
        for step, batch in enumerate(progress_bar):
            if global_step >= config["max_train_steps"]:
                print("Reached max training steps. Exiting.")
                break

            # Move batch to MPS
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass (no mixed precision on MPS)
            loss = pipeline(batch).loss  # Assuming pipeline has a loss method

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Logging
            wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step})

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), step=global_step)

            # Save checkpoint periodically
            if global_step % 100 == 0 or global_step == config["max_train_steps"] - 1:
                save_path = output_dir / f"checkpoint-{global_step}"
                pipeline.save_pretrained(save_path)
                print(f"Checkpoint saved at {save_path}")

            global_step += 1

    print("Training completed.")
    pipeline.save_pretrained(output_dir / "final_checkpoint")
    wandb.finish()
