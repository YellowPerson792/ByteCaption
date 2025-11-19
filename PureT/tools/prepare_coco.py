#!/usr/bin/env python3
"""
Prepare COCO dataset from HuggingFace and convert to PureT format.

This script:
1. Loads COCO 2014 dataset from HuggingFace
2. Converts it to the format expected by PureT
3. Creates train/val/test splits with proper image IDs and captions
4. Saves JSON files compatible with PureT's data loaders
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import datasets


def load_coco(split='train'):
    """Load COCO dataset from local HuggingFace cache."""
    path = '/root/autodl-fs/AbdoTW___coco_2014_karpathy'  # Update this path if needed
    return datasets.load_from_disk(f"{path}/{split}")


def convert_split_to_puret_format(dataset, output_dir, split_name):
    """Convert a specific split of COCO dataset to PureT format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing COCO {split_name} split...")
    
    # Prepare data structures
    images = []
    annotations = []
    image_id_mapping = {}  # filename -> image_id
    next_image_id = 1
    next_ann_id = 1
    
    for idx, item in enumerate(dataset):
        # COCO dataset should have 'image' and 'caption' fields
        image = item.get('image')
        captions = item.get('caption', [])
        
        # Generate consistent filename for image
        filename = f"coco_{split_name}_{idx:012d}.jpg"
        
        # Skip if no captions found
        if not captions:
            continue
        
        # Get image dimensions (default to common COCO dimensions if not available)
        if hasattr(image, 'size'):
            width, height = image.size
        else:
            # Default COCO-like dimensions
            width, height = 640, 480
        
        # Assign image ID if not seen before
        if filename not in image_id_mapping:
            image_id_mapping[filename] = next_image_id
            images.append({
                'id': next_image_id,
                'file_name': filename,
                'height': height,
                'width': width,
            })
            next_image_id += 1
        
        image_id = image_id_mapping[filename]
        
        # Handle captions - COCO usually has a list of captions
        if isinstance(captions, str):
            captions = [captions]
        elif not isinstance(captions, list):
            captions = []
            
        # Add captions as annotations
        for caption in captions:
            if caption and caption.strip():
                annotations.append({
                    'id': next_ann_id,
                    'image_id': image_id,
                    'caption': caption.strip()
                })
                next_ann_id += 1
    
    # Create COCO-format annotation structure
    coco_format = {
        'info': {
            'description': f'COCO 2014 {split_name} set in PureT format',
            'version': '1.0',
            'year': 2024,
            'contributor': 'Converted from HuggingFace COCO dataset',
            'date_created': '2024-01-01'
        },
        'images': images,
        'annotations': annotations,
        'licenses': [],
        'categories': []
    }
    
    # Save COCO-format annotation file
    ann_file = output_dir / f'captions_{split_name}.json'
    with open(ann_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)
    
    # Create image IDs file (mapping image_id -> filename)
    ids_file = output_dir / f'{split_name}_ids.json'
    image_ids = {str(img['id']): img['file_name'] for img in images}
    with open(ids_file, 'w') as f:
        json.dump(image_ids, f, indent=2)
    
    print(f"  Saved {len(images)} images and {len(annotations)} captions")
    print(f"  Annotation file: {ann_file}")
    print(f"  IDs file: {ids_file}")
    
    return len(images), len(annotations)
def create_config_template(output_dir):
    """Create a configuration template for using COCO with PureT."""
    output_dir = Path(output_dir)
    
    config_template = f"""
# COCO Configuration for PureT
# Add these settings to your config file

# Dataset paths
DATA_LOADER:
  TRAIN_ID: '{output_dir}/train_ids.json'
  VAL_ID: '{output_dir}/val_ids.json'
  TEST_ID: '{output_dir}/val_ids.json'  # Use validation as test for now
  
  # Note: These will be dummy paths since we don't have precomputed features
  # You'll need to either:
  # 1. Precompute features for your images, or
  # 2. Modify the data loader to load raw images (current implementation)
  TRAIN_GV_FEAT: ''
  VAL_GV_FEAT: ''
  TEST_GV_FEAT: ''
  
  TRAIN_ATT_FEATS: ''
  VAL_ATT_FEATS: ''
  TEST_ATT_FEATS: ''

# Evaluation settings
INFERENCE:
  EVAL: 'COCO'  # Use COCO evaluator
  VAL_ANNFILE: '{output_dir}/captions_val.json'
  TEST_ANNFILE: '{output_dir}/captions_val.json'
  ID_KEY: 'image_id'
  CAP_KEY: 'caption'
  VOCAB: '{output_dir}/coco_vocabulary.txt'

# Model settings
MODEL:
  VOCAB_SIZE: 9487  # Will be updated based on actual vocabulary size
  SEQ_LEN: 17  # Maximum sequence length for captions
"""
    
    config_file = output_dir / 'coco_config.yaml'
    with open(config_file, 'w') as f:
        f.write(config_template)
    
    print(f"Configuration template saved to: {config_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare COCO dataset for PureT')
    parser.add_argument('--output_dir', default='./data/coco_karpathy', 
                       help='Output directory for processed data')
    args = parser.parse_args()
    
    print("Loading COCO dataset from HuggingFace...")
    try:
        # Load all available splits from COCO
        available_splits = ['train']  # Start with train
        
        # Try to load validation and test if they exist
        try:
            val_dataset = load_coco('validation')
            available_splits.append('validation')
        except:
            print("Validation split not found, will use subset of train for validation")
            
        try:
            test_dataset = load_coco('test')  
            available_splits.append('test')
        except:
            print("Test split not found, will use subset of train for test")
        
        total_images = 0
        total_annotations = 0
        
        # Process each available split
        for split in available_splits:
            dataset = load_coco(split)
            print(f"Loaded {len(dataset)} samples from COCO {split} set")
            
            images, annotations = convert_split_to_puret_format(dataset, args.output_dir, split)
            total_images += images
            total_annotations += annotations
        
        # If we only have train split, create val split from it
        if len(available_splits) == 1:
            print("Creating validation split from training data (10%)...")
            train_dataset = load_coco('train')
            # Use last 10% as validation
            val_start = int(len(train_dataset) * 0.9)
            val_dataset = train_dataset.select(range(val_start, len(train_dataset)))
            val_images, val_annotations = convert_split_to_puret_format(val_dataset, args.output_dir, 'val')
            total_images += val_images
            total_annotations += val_annotations
        
        print("Creating configuration template...")
        create_config_template(args.output_dir)
        
        print(f"\nSuccess! COCO data prepared in: {args.output_dir}")
        print(f"Total: {total_images} images, {total_annotations} captions")
        print("\nNext steps:")
        print("1. Update your config file with the paths from coco_config.yaml")
        print("2. Build vocabulary using: python generate_coco_vocab.py --output data/coco/coco_vocabulary.txt")
        print("3. Train your model with the prepared data")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
