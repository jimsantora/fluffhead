# tests/test_training.py
import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from src.dataset import prepare_dataset
from src.model import setup_model_and_optimizer
from src.training import train_loop


@dataclass
class MockUNetOutput:
    sample: torch.Tensor
    loss: float


class TestSDXLTraining:
    @pytest.fixture
    def sample_dataset(self):
        # Create temp dir with dummy images
        temp_dir = Path(tempfile.mkdtemp()) / "images"
        temp_dir.mkdir(parents=True)

        # Create 3 random test images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            img.save(temp_dir / f"test_{i}.jpg")

        yield str(temp_dir)
        shutil.rmtree(temp_dir.parent)

    @pytest.fixture
    def mock_pipeline(self):
        mock = MagicMock()

        # Create a mock UNet that returns proper output structure
        def mock_forward(x):
            output = torch.nn.Conv2d(3, 1, 3, padding=1)(x)
            return MockUNetOutput(sample=output, loss=torch.tensor(0.5, device="mps"))

        mock.unet = MagicMock()
        mock.unet.__call__ = mock_forward
        mock.unet.parameters = lambda: [torch.nn.Parameter(torch.randn(1))]
        mock.device = torch.device("mps")

        # Mock save_pretrained to create the directory
        def mock_save_pretrained(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        mock.save_pretrained = mock_save_pretrained
        return mock

    def test_dataset_loading(self, sample_dataset):
        dataset = prepare_dataset(sample_dataset)
        assert len(dataset["test"]) == 3

        # Check image shape and values
        sample = dataset["test"][0]
        assert "pixel_values" in sample
        assert isinstance(sample["pixel_values"], torch.Tensor)
        assert sample["pixel_values"].shape == (3, 512, 512)
        assert -1.0 <= sample["pixel_values"].min() <= sample["pixel_values"].max() <= 1.0

    def test_model_setup(self, mock_pipeline):
        with patch(
            "diffusers.StableDiffusionXLPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            pipeline, optimizer = setup_model_and_optimizer("dummy_path", learning_rate=1e-5)

            # Basic pipeline checks
            assert pipeline.device.type == "mps"
            assert pipeline.unet is not None

            # Optimizer checks
            assert isinstance(optimizer, torch.optim.AdamW)
            assert optimizer.param_groups[0]["lr"] == 1e-5

    @patch("wandb.init")
    @patch("wandb.log")
    def test_minimal_training_run(self, mock_wandb_log, mock_wandb_init, sample_dataset, mock_pipeline):
        mock_wandb_init.return_value = MagicMock()
        # Setup minimal training config
        config = {
            "batch_size": 1,
            "learning_rate": 1e-5,
            "num_epochs": 1,
            "max_train_steps": 2,
            "use_mixed_precision": False,
            "output_dir": tempfile.mkdtemp(),
            "checkpoint_freq": 1,  # Save checkpoint every step
        }

        with patch(
            "diffusers.StableDiffusionXLPipeline.from_pretrained",
            return_value=mock_pipeline,
        ):
            # Prepare components
            dataset = prepare_dataset(sample_dataset)
            dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=config["batch_size"])
            pipeline, optimizer = setup_model_and_optimizer("dummy_path", config["learning_rate"])

            # Run training
            try:
                train_loop(pipeline, optimizer, dataloader, config)
                assert Path(config["output_dir"]).exists()
                # Check for checkpoint files
                checkpoint_path = Path(config["output_dir"]) / "checkpoint-1"
                assert checkpoint_path.exists()
            finally:
                shutil.rmtree(config["output_dir"])


if __name__ == "__main__":
    pytest.main(["-v"])
