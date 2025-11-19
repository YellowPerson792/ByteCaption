#!/usr/bin/env python3
"""
Prepare Flickr8k dataset from HuggingFace and convert to PureT format.

This script:
1. Downloads Flickr8k from HuggingFace
2. Converts it to the format expected by PureT (COCO-like structure)
3. Creates train/val/test splits with proper image IDs and captions
4. Saves JSON files compatible with PureT's data loaders
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def safe_load_flickr8k():
    """Safely load Flickr8k dataset from HuggingFace."""
    import sys
    import importlib
    import site
    
    # Ensure we import HF datasets, not local PureT datasets
    local_cached = sys.modules.pop('datasets', None)
    original_sys_path = list(sys.path)
    
    # Prioritize site-packages
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_paths.insert(0, user_site)
    except Exception:
        pass
    sys.path = site_paths + [p for p in sys.path if p not in site_paths]
    
    try:
        # Disable multiprocessing and caching
        os.environ['HF_DATASETS_DISABLE_CACHING'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        hf_datasets = importlib.import_module('datasets')
        if hasattr(hf_datasets, 'disable_caching'):
            hf_datasets.disable_caching()
        
        # Load all splits
        dataset = hf_datasets.load_dataset(
            'jxie/flickr8k',
            num_proc=None,
            cache_dir=None,
            trust_remote_code=False,
            streaming=False,
            verification_mode='no_checks'
        )
        return dataset
    finally:
        sys.path = original_sys_path
        if local_cached is not None:
            sys.modules['datasets'] = local_cached


def convert_to_puret_format(dataset, output_dir):
    """Convert HF Flickr8k dataset to PureT format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split...")
        
        # Convert split name to match PureT conventions
        if split_name == 'validation':
            puret_split = 'val'
        else:
            puret_split = split_name
        
        # Prepare data structures
        images = []
        annotations = []
        image_id_mapping = {}  # filename -> image_id
        next_image_id = 1
        next_ann_id = 1
        
        # Group captions by image
        image_captions = defaultdict(list)
        
        for idx, item in enumerate(split_data):
            # Generate filename for image (PIL images don't have consistent filename)
            filename = f"flickr8k_{split_name}_{idx:06d}.jpg"
            
            # Get captions from caption_0 to caption_4 fields
            captions = []
            for i in range(5):  # Flickr8k typically has 5 captions per image
                caption_key = f'caption_{i}'
                if caption_key in item and item[caption_key] and item[caption_key].strip():
                    captions.append(item[caption_key].strip())
            
            # Skip if no captions found
            if not captions:
                continue
            
            # Assign image ID if not seen before
            if filename not in image_id_mapping:
                image_id_mapping[filename] = next_image_id
                images.append({
                    'id': next_image_id,
                    'file_name': filename,
                    'height': 480,  # Default Flickr8k image size
                    'width': 640,
                })
                next_image_id += 1
            
            image_id = image_id_mapping[filename]
            
            # Add captions as annotations
            for caption in captions:
                if caption and caption.strip():
                    annotations.append({
                        'id': next_ann_id,
                        'image_id': image_id,
                        'caption': caption.strip()
                    })
                    next_ann_id += 1
        
        # Create COCO-like structure
        coco_format = {
            'info': {
                'description': f'Flickr8k {split_name} set in COCO format',
                'version': '1.0',
                'year': 2024,
                'contributor': 'Converted from HuggingFace jxie/flickr8k',
                'date_created': '2024-01-01'
            },
            'images': images,
            'annotations': annotations,
            'licenses': [],
            'categories': []
        }
        
        # Save COCO-format annotation file
        ann_file = output_dir / f'captions_{puret_split}.json'
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2, ensure_ascii=False)
        
        # Create image IDs file (list of image IDs for data loading)
        ids_file = output_dir / f'{puret_split}_ids.json'
        image_ids = {str(img['id']): img['file_name'] for img in images}
        with open(ids_file, 'w') as f:
            json.dump(image_ids, f, indent=2)
        
        print(f"  Saved {len(images)} images and {len(annotations)} captions")
        print(f"  Annotation file: {ann_file}")
        print(f"  IDs file: {ids_file}")


def create_config_template(output_dir):
    """Create a configuration template for using Flickr8k with PureT."""
    output_dir = Path(output_dir)
    
    config_template = f"""
# Flickr8k Configuration for PureT
# Add these settings to your config file

# Dataset paths
DATA_LOADER:
  TRAIN_ID: '{output_dir}/train_ids.json'
  VAL_ID: '{output_dir}/val_ids.json'
  TEST_ID: '{output_dir}/test_ids.json'
  
  # Note: These will be dummy paths since we don't have precomputed features
  # You'll need to either:
  # 1. Precompute features for your images, or
  # 2. Modify the data loader to load raw images
  TRAIN_GV_FEAT: '{output_dir}/dummy_gv_feat.npz'
  VAL_GV_FEAT: '{output_dir}/dummy_gv_feat.npz'
  TEST_GV_FEAT: '{output_dir}/dummy_gv_feat.npz'
  
  TRAIN_ATT_FEATS: '{output_dir}/dummy_att_feats'
  VAL_ATT_FEATS: '{output_dir}/dummy_att_feats'
  TEST_ATT_FEATS: '{output_dir}/dummy_att_feats'

# Evaluation settings
INFERENCE:
  EVAL: 'COCO'  # Use COCO evaluator with Flickr8k annotations
  VAL_ANNFILE: '{output_dir}/captions_val.json'
  TEST_ANNFILE: '{output_dir}/captions_test.json'
  ID_KEY: 'image_id'
  CAP_KEY: 'caption'

# Vocabulary (you may need to build this from the captions)
INFERENCE:
  VOCAB: 'your_vocab_file.json'
"""
    
    config_file = output_dir / 'flickr8k_config.yaml'
    with open(config_file, 'w') as f:
        f.write(config_template)
    
    print(f"Configuration template saved to: {config_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Flickr8k dataset for PureT')
    parser.add_argument('--output_dir', default='./data/flickr8k', 
                       help='Output directory for processed data')
    args = parser.parse_args()
    
    print("Loading Flickr8k dataset from HuggingFace...")
    try:
        dataset = safe_load_flickr8k()
        print(f"Loaded dataset with splits: {list(dataset.keys())}")
        
        print("Converting to PureT format...")
        convert_to_puret_format(dataset, args.output_dir)
        
        print("Creating configuration template...")
        create_config_template(args.output_dir)
        
        print(f"\nSuccess! Flickr8k data prepared in: {args.output_dir}")
        print("\nNext steps:")
        print("1. Update your config file with the paths from flickr8k_config.yaml")
        print("2. Either precompute visual features or modify data loader for raw images")
        print("3. Build vocabulary from the caption files")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
