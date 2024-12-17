import torch
from diffusers import StableDiffusionXLPipeline


def setup_model_and_optimizer(pretrained_model_path, learning_rate):
    """Loads the SDXL model and prepares the optimizer."""
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_path, revision="fp16", torch_dtype=torch.float16
    )
    pipeline.to("mps")

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
    return pipeline, optimizer
