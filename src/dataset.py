# src/dataset.py
from torchvision import transforms
from datasets import load_dataset


def prepare_dataset(dataset_path):
    """Loads and preprocesses the dataset for training."""
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = load_dataset("imagefolder", data_dir=dataset_path)

    def transform_images(examples):
        # Process one image at a time
        transformed_images = [transform(image) for image in examples["image"]]
        return {"pixel_values": transformed_images}

    # Apply transformations using batched map
    dataset = dataset.with_transform(transform_images)

    return dataset
