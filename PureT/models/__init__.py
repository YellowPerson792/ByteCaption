from models.pure_transformer import PureT
from models.pure_transformer import PureT_Base
from models.pure_transformer import PureT_Base_22K
from PureT.models.bytecaption_model import PureT_byteformer
from .hf_visual_model import HFVisualModel
from .hf_vl_chat_model import HFVLChatModel
from .openrouter_model import OpenRouterCaptionModel

__factory = {
    'PureT': PureT,
    'PureT_Base': PureT_Base,
    'PureT_Base_22K': PureT_Base_22K,
    'PureT_byteformer': PureT_byteformer,
    # Unified HuggingFace/Transformer vision captioner
    'BLIP': HFVisualModel,
    'HF_BLIP': HFVisualModel,
    'HF': HFVisualModel,
    'GIT': HFVisualModel,
    'HF_GIT': HFVisualModel,
    'QWEN': HFVLChatModel,
    'INTERNVL': HFVLChatModel,
    'GLM': HFVLChatModel,
    'MINISTRAL': HFVLChatModel,
    'MISTRAL': HFVLChatModel,
    'OPENROUTER': OpenRouterCaptionModel,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)
