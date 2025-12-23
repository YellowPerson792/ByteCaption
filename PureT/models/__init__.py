from models.pure_transformer import PureT
from models.pure_transformer import PureT_Base
from models.pure_transformer import PureT_Base_22K
from models.pure_byteformer import PureT_byteformer
from .hf_caption_model import HFCaptionModel
from .qwen3_vl_caption_model import Qwen3VLCaptionModel
from .openrouter_caption_model import OpenRouterCaptionModel

__factory = {
    'PureT': PureT,
    'PureT_Base': PureT_Base,
    'PureT_Base_22K': PureT_Base_22K,
    'PureT_byteformer': PureT_byteformer,
    # Unified HuggingFace/Transformer vision captioner
    'BLIP': HFCaptionModel,
    'HF_BLIP': HFCaptionModel,
    'HF': HFCaptionModel,
    'GIT': HFCaptionModel,
    'HF_GIT': HFCaptionModel,
    'QWEN': Qwen3VLCaptionModel,
    'HF_QWEN': Qwen3VLCaptionModel,
    'OPENROUTER': OpenRouterCaptionModel,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)
