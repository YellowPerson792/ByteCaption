"""
Dynamic feature extraction for Flickr8k images, maintaining PureT compatibility.
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# timm interp compatibility
try:
    from timm.data.transforms import _pil_interp
except ImportError:
    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        if method == 'bilinear':
            return Image.BILINEAR
        if method == 'nearest':
            return Image.NEAREST
        return Image.BICUBIC


class PureTFeatureExtractor:
    """
    Feature extractor for PureT, following original implementation:
    - gv_feat: Simple placeholder (PureT computes global features dynamically)
    - att_feats: Preprocessed image tensor for Swin Transformer backbone
    """
    
    def __init__(self):
        # Transform for attention features matching original PureT pipeline
        self.att_transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        
    def extract_features(self, image):
        """
        Extract features following PureT's original approach.
        
        Args:
            image: PIL Image
            
        Returns:
            tuple: (gv_feat, att_feats)
                - gv_feat: numpy array placeholder (PureT will compute dynamically)
                - att_feats: torch.Tensor of shape (3, 384, 384) for Swin backbone
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL Image")
            
        # Attention features: preprocessed image for Swin Transformer
        att_feats = self.att_transform(image)
        
        # Global features: placeholder (PureT computes from att_feats dynamically)
        # Use small placeholder to maintain compatibility
        gv_feat = np.zeros((1,), dtype=np.float32)
        
        return gv_feat, att_feats


# Global feature extractor instance (singleton pattern)
_feature_extractor = None

def get_feature_extractor():
    """Get global feature extractor instance."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = PureTFeatureExtractor()
    return _feature_extractor
