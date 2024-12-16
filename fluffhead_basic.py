import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLTrainingPipeline
from diffusers.optimization import get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import argparse


# === Module 1: Configuration ===
def get_config():
    parser = argparse.ArgumentParser(description="DreamBooth Training for SDXL")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to the pretrained SDXL model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the image dataset.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Max training steps.")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training.")
    return parser.parse_args()


# === Module 2: Dataset Preparation ===
def prepare_dataset(dataset_path):
    """Loads and preprocesses the dataset for training."""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    dataset = dataset.with_transform(lambda examples: {"pixel_values": transform(examples["image"])})
    return dataset


# === Module 3: Model and Optimizer Setup ===
def setup_model_and_optimizer(pretrained_model_path, learning_rate):
    """Loads the SDXL model and prepares the optimizer."""
    pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_path, revision="fp16", torch_dtype=torch.float16)
    pipeline.to("mps")  # Use MPS backend for Apple Silicon

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
    return pipeline, optimizer


# === Module 4: Training Loop ===
def train_loop(pipeline, optimizer, dataloader, args):
    """Core training loop."""
    scheduler = get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps)

    step = 0
    for epoch in range(args.num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in loop:
            pixel_values = batch["pixel_values"].to("mps")
            
            # Forward pass and loss computation
            with torch.cuda.amp.autocast(enabled=args.use_mixed_precision):
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
            if step % 100 == 0:
                pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))

            if step >= args.max_train_steps:
                break
        if step >= args.max_train_steps:
            break

    # Final save
    pipeline.save_pretrained(args.output_dir)


# === Module 5: Main Script ===
def main():
    args = get_config()

    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load model and optimizer
    print("Loading model and optimizer...")
    pipeline, optimizer = setup_model_and_optimizer(args.pretrained_model, args.learning_rate)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train the model
    print("Starting training...")
    train_loop(pipeline, optimizer, dataloader, args)

    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
