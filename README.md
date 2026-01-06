# Auto Photo Cluster

**Auto Photo Cluster** is an intelligent tool that automatically groups your photos into folders based on visual similarity. It uses state-of-the-art AI (OpenAI's CLIP model) to "see" your photos and a smart balancing algorithm to ensure your folders are evenly sized.

## Features

- **🧠 AI-Powered**: Uses CLIP-ViT-Large to understand image content (semantics, style, composition).
- **⚖️ Balanced Clustering**: Unlike standard algorithms that create one giant folder, this tool recursively splits clusters to ensure no folder exceeds your specified limit (default 50 images).
- **🚀 GPU Accelerated**: Automatically uses NVIDIA CUDA if available for fast processing.
- **💾 Smart Caching**: Caches image features (`features_cache.npz`) so you can experiment with different clustering settings instantly without re-processing images.

## Installation

1. Clone or download this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note**: For GPU support, ensure you have the correct version of PyTorch installed for your CUDA version: [PyTorch.org](https://pytorch.org/get-started/locally/)

## Usage

### One-Click Splitting (Default)
Tries to split images in the current directory into folders of ~30-50 images.

```bash
python main.py
```

### Basic Usage
Specify input and output directories:

```bash
python main.py --input-dir /path/to/my/photos --output-dir /path/to/clustered
```

### Customizing Granularity
Control how many images end up in each folder:

- **Max Size**: The strict limit. No folder will have more than this.
- **Ideal Size**: The target average size.

```bash
# Fine-grained (Small folders of max 50 images)
python main.py --max-size 50 --ideal-size 30

# Coarse-grained (Larger folders of max 200 images)
python main.py --max-size 200 --ideal-size 100
```

### Other Options
- `--move`: **Move** files instead of copying them (Use with caution!).
- `--no-cache`: Force re-extraction of features.

## How It Works

1.  **Feature Extraction**: The script reads all images and uses the CLIP model to generate a 768-dimensional vector representation for each image.
2.  **Recursive Stratified Ward Clustering**: 
    -   It starts by attempting to cluster images into groups.
    -   If any group is larger than `max-size`, it isolates that group and recursively splits it again using Ward Linkage (which minimizes variance).
    -   This process repeats until all clusters satisfy the size constraints.
