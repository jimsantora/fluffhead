# src/training.py
import os
from tqdm import tqdm
import torch
from diffusers.optimization import get_scheduler


def train_loop(pipeline, optimizer, dataloader, config):
    """Core training loop."""
    scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=config["max_train_steps"],
    )

    step = 0
    checkpoint_freq = config.get(
        "checkpoint_freq", 100
    )  # Default to 100 if not specified

    for epoch in range(config["num_epochs"]):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in loop:
            pixel_values = batch["pixel_values"].to("mps")

            # Forward pass and loss computation
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=config["use_mixed_precision"],
            ):
                outputs = pipeline.unet(pixel_values)
                loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            step += 1
            loop.set_postfix(loss=loss.item())

            # Save checkpoints
            if step % checkpoint_freq == 0:
                pipeline.save_pretrained(
                    os.path.join(config["output_dir"], f"checkpoint-{step}")
                )

            if step >= config["max_train_steps"]:
                break
        if step >= config["max_train_steps"]:
            break

    # Final save
    pipeline.save_pretrained(config["output_dir"])
