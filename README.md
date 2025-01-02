# Infinite Canvas CLI: Stable Diffusion 3.5 for Apple Silicon

Run Stability AI's latest Stable Diffusion 3.5 model entirely on Apple Silicon devices, with enterprise-grade performance monitoring and optimization. Part of the Infinite Canvas framework for on-device AI.

## Key Features

- **Local-First Processing**: Generate high-quality images without cloud dependencies
- **Enterprise Performance**: Comprehensive metrics tracking and benchmarking
- **Hardware Optimization**: Automatic MPS configuration for M1/M2/M3 chips
- **Memory Management**: Smart RAM detection and optimization
- **Benchmark Suite**: Built-in tools for performance analysis

## Prerequisites

- Python 3.11
- Conda
- Hugging Face API token (optional)

## Quick Start

1. Create environment:
```bash
conda create -n sd3 python=3.11 -y
conda activate sd3
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: Set HF token:

```bash
export HF_API_TOKEN=your_token_here
```

## Usage

### Single Image Generation

```bash
python sd3-on-mps.py --mode single --scene-type your_scene_name
```

### Comprehensive Benchmarking

```bash
python sd3-on-mps.py --mode benchmark
```

## Performance Metrics

The tool tracks:

- Generation time
- Memory usage
- Tokens per second
- Step-by-step inference timing
- Hardware utilization
- Device-specific metrics

## Configuration Options

### Core Parameters

- `seed`: Control image reproducibility
- `height/width`: Output resolution (up to 2048x2048)
- `num_inference_steps`: Quality vs. speed tradeoff
- `guidance_scale`: Output fidelity control

### Advanced Settings

- Memory optimization thresholds
- Device-specific configurations
- Benchmark parameters
- Output formats and paths

## Enterprise Features

1. **Performance Monitoring**
   - Real-time memory tracking
   - Step-by-step inference timing
   - Hardware utilization metrics

2. **Quality Control**
   - Reproducible outputs via seed control
   - Configurable quality parameters
   - Detailed generation logs

3. **Resource Management**
   - Automatic memory optimization
   - Hardware-aware configurations
   - Cache management

## Benchmarking

The built-in benchmark suite tests:

- Multiple resolutions (512x512 to 1024x1024)
- Various inference step counts
- Different prompt complexities
- Hardware performance metrics

Results are saved as JSON files with comprehensive metadata.

## Integration

Part of the Infinite Canvas framework, this tool is designed for:

- Creative agencies
- Design studios
- Enterprise content teams
- Local AI workflows

## License

MIT License - See LICENSE file for details.