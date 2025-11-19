import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from lib.config import cfg

class BlipWrapper(nn.Module):
    """
    一个包装器，使HuggingFace的BLIP模型看起来像项目中的byteformer模型。
    它遵循相同的接口，例如 decode_beam 方法。
    """
    def __init__(self):
        super(BlipWrapper, self).__init__()
        model_name = "Salesforce/blip-image-captioning-base"
        model_dir = "blip-image-captioning-base"
        device = 'cuda'
        
        print(f"[BlipWrapper] Initializing model '{model_name}' on device '{device}'...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_dir,use_safetensors=True)
        self.model.to(device)
        self.model.eval() # BLIP模型仅用于评估，始终处于eval模式

    def forward(self, *args, **kwargs):
        # 训练时用不到，可以留空
        pass

    def decode_beam(self, **kwargs):
        """
        统一的解码接口。它接收与其他模型相同的kwargs，但只使用'att_feats'。
        """
        # 在我们的统一数据流中，'att_feats' 将是原始的图像张量或损坏后解码的图像
        images = kwargs[cfg.PARAM.ATT_FEATS]
        beam_size = kwargs.get('BEAM_SIZE', 3)
        
        # 筛选出有效的图像（非None）
        valid_images_with_indices = [(i, img) for i, img in enumerate(images) if img is not None]
        
        # 若全部为损坏图像，直接返回了
        if not valid_images_with_indices:
            dummy_caption = "this is a dummy caption for an undecodable image"
            return [dummy_caption for _ in range(len(images))], None

        original_indices, valid_images = zip(*valid_images_with_indices)
        
        inputs = self.processor(images=list(valid_images), return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            num_beams=beam_size,
            max_length=50
        )

        generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 将结果按原始顺序重组，为无法解码的图像填充默认标题
        final_captions = ["" for _ in range(len(images))]
        dummy_caption = "this is a dummy caption for an undecodable image"

        # 1. 填充成功生成的标题
        for i, caption in zip(original_indices, generated_captions):
            # 如果模型生成了空标题，也使用 dummy caption
            final_captions[i] = caption.strip() if caption.strip() else dummy_caption
        
        # 2. 为所有原始图像为 None 的位置填充 dummy caption
        for i in range(len(images)):
            if images[i] is None:
                final_captions[i] = dummy_caption


        return final_captions, None

    def decode(self, **kwargs):
        kwargs['BEAM_SIZE'] = 1
        return self.decode_beam(**kwargs)