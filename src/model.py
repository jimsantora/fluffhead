import torch
from diffusers import StableDiffusionXLPipeline


def setup_model_and_optimizer(pretrained_model_path, learning_rate):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_path, torch_dtype=torch.float32  # Changed from float16
    )
    pipeline.to("mps")
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
    return pipeline, optimizer
