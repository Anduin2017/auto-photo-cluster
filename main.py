#!/usr/bin/env python3
"""
Auto Photo Cluster
==================

A tool to automatically cluster images into folders based on visual similarity.
It uses OpenAI's CLIP model for feature extraction and a Recursive Stratified Ward Clustering algorithm
to ensure balanced cluster sizes.

Usage:
    python main.py --input-dir /path/to/photos --output-dir /path/to/clustered
"""

import os
import shutil
import logging
import argparse
import math
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AutoCluster")

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLIPFeatureExtractor:
    """Handles loading the CLIP model and extracting image features."""
    
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        logger.info("Initializing CLIP model...")
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            logger.error("Missing dependencies. Please run: pip install -r requirements.txt")
            raise

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, image_paths: List[Path], batch_size=32) -> Tuple[np.ndarray, List[Path]]:
        """
        Extracts features for a list of images.
        Returns: (features_matrix, list_of_successfully_processed_paths)
        """
        import torch

        valid_paths = []
        features_list = []
        
        # Pre-filter images to avoid crashing efficiently
        valid_inputs = []
        for p in image_paths:
            try:
                # Lazy check
                valid_inputs.append(p)
            except Exception:
                pass
        
        logger.info(f"Extracting features for {len(valid_inputs)} images...")
        
        # Process in batches
        for i in tqdm(range(0, len(valid_inputs), batch_size), unit="batch"):
            batch_paths = valid_inputs[i:i + batch_size]
            batch_images = []
            current_batch_paths = []
            
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    batch_images.append(img)
                    current_batch_paths.append(p)
                except Exception as e:
                    logger.warning(f"Skipping {p.name}: {e}")
                    continue

            if not batch_images:
                continue

            try:
                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    # L2 Normalize
                    outputs /= outputs.norm(dim=-1, keepdim=True)
                
                features_list.append(outputs.cpu().numpy())
                valid_paths.extend(current_batch_paths)
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")

        if not features_list:
            return np.array([]), []

        return np.concatenate(features_list, axis=0), valid_paths


class RecursiveClusterer:
    """
    Hierarchical clustering that recursively splits clusters larger than `max_size`.
    """
    def __init__(self, ideal_size=30, max_size=50):
        self.ideal_size = ideal_size
        self.max_size = max_size
        self.global_labels = {}
        self.next_cluster_id = 0
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            self.cluster_algo = AgglomerativeClustering
        except ImportError:
            logger.error("scikit-learn not found. Please pip install scikit-learn")
            raise

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        indices = np.arange(n_samples)
        
        logger.info(f"Clustering {n_samples} items (Target Max Size: {self.max_size})...")
        self._recursive_split(features, indices)
        
        # Reconstruct labels array
        result = np.zeros(n_samples, dtype=int)
        for idx, label in self.global_labels.items():
            result[idx] = label
        return result

    def _recursive_split(self, features: np.ndarray, indices: np.ndarray):
        n_samples = features.shape[0]
        
        # Base case: Cluster is small enough
        if n_samples <= self.max_size:
            self._assign_new_label(indices)
            return

        # Calculate number of clusters to split into
        # We aim for ideal_size chunks
        n_clusters = math.ceil(n_samples / self.ideal_size)
        n_clusters = max(2, n_clusters) # At least split in two
        
        # Ward linkage minimizes variance -> creates balanced, spherical clusters
        model = self.cluster_algo(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = model.fit_predict(features)
        
        # Recurse on children
        for label in np.unique(labels):
            mask = labels == label
            self._recursive_split(features[mask], indices[mask])

    def _assign_new_label(self, indices: np.ndarray):
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        for idx in indices:
            self.global_labels[idx] = cluster_id


def get_image_paths(input_dir: Path, recursive: bool = False) -> List[Path]:
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
    files = []
    for ext in extensions:
        if recursive:
            files.extend(input_dir.rglob(f"*{ext}"))
            files.extend(input_dir.rglob(f"*{ext.upper()}"))
        else:
            files.extend(input_dir.glob(f"*{ext}"))
            files.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def main():
    parser = argparse.ArgumentParser(description="Auto Photo Cluster: Smartly organize your photos.")
    parser.add_argument('input_dir', nargs='?', default='.', help="Directory containing images to cluster")
    parser.add_argument('--output-dir', default='clustered_output', help="Directory to save clustered folders")
    parser.add_argument('--max-size', type=int, default=50, help="Maximum number of images per folder")
    parser.add_argument('--ideal-size', type=int, default=30, help="Ideal target number of images per folder")
    parser.add_argument('--move', action='store_true', help="Move files instead of copying (WARNING: modifies source)")
    parser.add_argument('--no-cache', action='store_true', help="Ignore cached features and re-compute")
    parser.add_argument('--batch-size', type=int, default=32, help="Inference batch size")
    parser.add_argument('--recursive', action='store_true', help="Recursively search for images in subdirectories")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    # 1. Gather files
    image_paths = get_image_paths(input_path, args.recursive)
    # Ignore images already in output dir if input is parent
    image_paths = [p for p in image_paths if output_path not in p.parents]
    
    if not image_paths:
        logger.info("No images found.")
        return
        
    logger.info(f"Found {len(image_paths)} images.")

    # 2. Extract Features (with caching)
    input_cache_file = input_path / "features_cache.npz"
    output_cache_file = output_path / "features_cache.npz"
    
    features = None
    valid_paths = None

    # Determine which cache file to try loading
    cache_to_load = None
    if not args.no_cache:
        if input_cache_file.exists():
            cache_to_load = input_cache_file
        elif output_cache_file.exists():
            cache_to_load = output_cache_file
    
    if cache_to_load:
        logger.info(f"Found cache: {cache_to_load}")
        try:
            data = np.load(cache_to_load, allow_pickle=True)
            cached_names = set(data['names'])
            current_names = set(str(p.relative_to(input_path)) for p in image_paths)
            
            # Simple consistency check
            # If cache has subset or superset, we might want to recompute or align
            # For simplicity: if strictly equal set of filenames, use cache
            if cached_names == current_names:
                features = data['features']
                # Reconstruct path objects 
                # (Assuming filenames are unique enough or flat directory)
                name_to_path = {str(p.relative_to(input_path)): p for p in image_paths}
                valid_paths = [name_to_path[n] for n in data['names']]
                logger.info("Cache loaded successfully.")
            else:
                logger.warning("File list changed. Re-computing features...")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    if features is None:
        extractor = CLIPFeatureExtractor()
        features, valid_paths = extractor.extract_features(image_paths, batch_size=args.batch_size)
        
        # Save cache
        names = [str(p.relative_to(input_path)) for p in valid_paths]
        
        # Try saving to input directory first, then output directory
        try:
            np.savez(input_cache_file, features=features, names=names)
            logger.info(f"Features cached to {input_cache_file}")
        except PermissionError:
            logger.warning(f"Permission denied saving to {input_cache_file}. Attempting to save to output directory.")
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                np.savez(output_cache_file, features=features, names=names)
                logger.info(f"Features cached to {output_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache to output directory: {e}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    if len(features) == 0:
        logger.error("No features extracted. Exiting.")
        return

    # 3. Cluster
    clusterer = RecursiveClusterer(ideal_size=args.ideal_size, max_size=args.max_size)
    labels = clusterer.fit_predict(features)

    # 4. Organize
    if output_path.exists():
        if not args.no_cache: # Safety check? Maybe just warn
            logger.warning(f"Output directory {output_path} already exists.")
        # shutil.rmtree(output_path) # Let's not auto-delete unless sure. 
        # Actually user wants "one click", so maybe auto-cleanup or merge? 
        # Standard behavior: create into it.
        pass
    else:
        output_path.mkdir(parents=True)

    clusters = defaultdict(list)
    for path, label in zip(valid_paths, labels):
        clusters[label].append(path)
    
    # Sort by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    logger.info(f"Organizing into {output_path}...")
    
    for i, (label, paths) in enumerate(sorted_clusters):
        count = len(paths)
        # Naming: cluster_001_size_50
        folder_name = f"cluster_{i:03d}_size_{count}"
        target_dir = output_path / folder_name
        target_dir.mkdir(exist_ok=True)
        
        for src in paths:
            dst = target_dir / src.name
            try:
                if args.move:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
            except Exception as e:
                logger.error(f"Failed to move/copy {src.name}: {e}")

    logger.info(f"Done! Created {len(sorted_clusters)} clusters in {output_path}")

if __name__ == "__main__":
    main()
