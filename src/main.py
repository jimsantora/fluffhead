import argparse
from pathlib import Path
import yaml
from training import train_loop
from dataset import prepare_dataset
from model import setup_model_and_optimizer
from torch.utils.data import DataLoader


def get_config():
    parser = argparse.ArgumentParser(description="DreamBooth Training for SDXL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--pretrained_model", type=str, help="Override pretrained model path")
    parser.add_argument("--dataset_path", type=str, help="Override dataset path")
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override with CLI arguments if provided
    if args.pretrained_model:
        config["pretrained_model"] = args.pretrained_model
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path

    return config


def main():
    config = get_config()

    print("Preparing dataset...")
    dataset = prepare_dataset(config["dataset_path"])
    dataloader = DataLoader(dataset["train"], batch_size=config["batch_size"], shuffle=True)

    print("Loading model and optimizer...")
    pipeline, optimizer = setup_model_and_optimizer(config["pretrained_model"], config["learning_rate"])

    Path(config["output_dir"]).mkdir(exist_ok=True)

    print("Starting training...")
    train_loop(pipeline, optimizer, dataloader, config)

    print(f"Training complete. Model saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
