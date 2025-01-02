import os
import time
import torch
import psutil
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline
import numpy as np
from dataclasses import dataclass
from typing import List
import platform
import json
import dataclasses
import argparse

SD3_MODEL_CACHE = "./sd3-cache"

# Define seed first
seed = None
if seed is None:
    seed = int.from_bytes(os.urandom(2), "big")
print(f"Using seed: {seed}")

# Update device handling
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Modify pipeline creation with recommended MPS settings
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
    cache_dir=SD3_MODEL_CACHE,
    use_safetensors=True,
    variant="fp16",
)

# Memory optimizations for Apple Silicon
pipe.enable_attention_slicing()

# Move pipeline to device
pipe = pipe.to(device)

# Generator should be on CPU for MPS
generator = torch.Generator("cpu").manual_seed(seed)

# Check available system RAM and enable attention slicing if less than 64 GB
if (available_ram := psutil.virtual_memory().available / (1024**3)) < 64:
    pipe.enable_attention_slicing()

prompt = """Futuristic deep space observatory interior,
massive holographic star maps, advanced astronomical instruments,
volumetric nebula projections, glass observation dome,
photorealistic space visualization, ray-traced reflections,
inspired by ESO observatories, extreme technical detail,
perfect exposure, award-winning scientific photography,
cinematic lighting, 16k resolution render"""

@dataclass
class GenerationMetrics:
    total_time: float
    tokens_per_second: float
    memory_peak: float
    inference_steps_time: List[float]
    device_type: str
    image_resolution: tuple
    prompt_length: int
    temperature: float
    device_info: dict

def get_device_info():
    if torch.backends.mps.is_available():
        return {
            "device": "Apple Silicon",
            "model": platform.processor(),
            "os": platform.platform(),
            "ram": f"{psutil.virtual_memory().total / (1024**3):.1f}GB"
        }
    return {}

def parse_args():
    parser = argparse.ArgumentParser(description='SD3 on Apple Silicon Benchmarking')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'benchmark'],
                       help='Run mode: single image or full benchmark')
    parser.add_argument('--scene-type', type=str, default='renaissance_cyberpunk',
                       help='Scene type for image naming')
    return parser.parse_args()

def generate_with_metrics(prompt, height, width, num_inference_steps, guidance_scale=4.5):
    memory_start = psutil.Process().memory_info().rss / 1024**2
    step_times = []
    start_time = time.time()
    
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        step_end = time.time()
        step_times.append(step_end - start_time)
        return callback_kwargs

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
        return_dict=True,
        callback_on_step_end=callback_on_step_end,
    ).images[0]

    end_time = time.time()
    memory_peak = psutil.Process().memory_info().rss / 1024**2 - memory_start

    metrics = GenerationMetrics(
        total_time=end_time - start_time,
        tokens_per_second=len(prompt.split()) / (end_time - start_time),
        memory_peak=memory_peak,
        inference_steps_time=step_times,
        device_type=device,
        image_resolution=(height, width),
        prompt_length=len(prompt.split()),
        temperature=guidance_scale,
        device_info=get_device_info()
    )

    return image, metrics

def run_comprehensive_benchmark():
    benchmark_configs = {
        'resolutions': [(512, 512), (768, 768), (1024, 1024)],
        'steps': [50, 75, 100],
        'prompts': [
            # Complex Scene
            """Renaissance-meets-cyberpunk throne room interior,
            ornate marble columns with holographic displays,
            volumetric god rays through stained glass windows,
            cinematic lighting, unreal engine 5 quality,
            architectural masterpiece, 16k resolution,
            perfect perspective, photorealistic materials,
            inspired by Zaha Hadid and Leonardo da Vinci""",
            
            # Technical Detail
            """Hyperdetailed quantum computer laboratory,
            holographic displays, clean room environment,
            perfect depth of field, studio lighting setup,
            photorealistic materials, ray-traced reflections,
            extreme technical detail, scientific visualization,
            professional industrial photography, 16k resolution""",
            
            # Architectural
            """Futuristic Tokyo skyscraper at golden hour,
            bioluminescent architectural details, floating gardens,
            photorealistic 16k resolution, perfect architectural accuracy,
            volumetric lighting, ray-traced global illumination,
            professional color grading, award-winning architectural photography,
            extreme structural detail, physically accurate materials"""
        ]
    }
    
    results = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    for prompt in benchmark_configs['prompts']:
        for res in benchmark_configs['resolutions']:
            for step in benchmark_configs['steps']:
                print(f"\nGenerating: {res[0]}x{res[1]}, steps={step}")
                image, metrics = generate_with_metrics(
                    prompt=prompt,
                    height=res[0],
                    width=res[1],
                    num_inference_steps=step
                )
                
                # Save image and metrics
                scene_type = f"benchmark_{res[0]}x{res[1]}_{step}steps"
                image_path = f"sd3-{scene_type}-{timestamp}.png"
                metrics_path = f"sd3-{scene_type}-{timestamp}-metrics.json"
                
                image.save(image_path)
                with open(metrics_path, 'w') as f:
                    json.dump(dataclasses.asdict(metrics), f, indent=2)
                
                results.append(metrics)
                
                print(f"Total Time: {metrics.total_time:.2f}s")
                print(f"Memory Peak: {metrics.memory_peak:.2f}MB")
    
    # Save aggregate results
    aggregate_path = f"sd3-benchmark-aggregate-{timestamp}.json"
    with open(aggregate_path, 'w') as f:
        json.dump({
            'configs': benchmark_configs,
            'results': [dataclasses.asdict(r) for r in results]
        }, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'single':
        # Single image generation with metrics
        image, metrics = generate_with_metrics(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=75
        )
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = f"sd3-{args.scene_type}-{timestamp}.png"
        metrics_path = f"sd3-{args.scene_type}-{timestamp}-metrics.json"
        
        image.save(image_path)
        with open(metrics_path, 'w') as f:
            json.dump(dataclasses.asdict(metrics), f, indent=2)
            
        print(f"\nPerformance Metrics:")
        print(f"Total Generation Time: {metrics.total_time:.2f}s")
        print(f"Memory Peak Usage: {metrics.memory_peak:.2f}MB")
        print(f"Tokens/Second: {metrics.tokens_per_second:.2f}")
        print(f"Average Step Time: {np.mean(metrics.inference_steps_time):.3f}s")
    
    else:
        # Full benchmark mode
        run_comprehensive_benchmark()
