from typing import List

from .hf_caption_model import HFCaptionModel


class GLMCaptionModel(HFCaptionModel):
    def _build_chat_messages(self, image) -> List[dict]:
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        user_content: List[dict] = []
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})
        user_content.append({"type": "image", "image": image})
        messages.append({"role": "user", "content": user_content})
        return messages
