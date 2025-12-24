import os
import shutil
from typing import List

from .hf_caption_model import HFCaptionModel


class InternVLCaptionModel(HFCaptionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backup_snapshot()

    def _build_chat_messages(self, image):
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

    def _download_snapshot(self) -> None:
        super()._download_snapshot()
        self._backup_snapshot()

    def _backup_snapshot(self) -> None:
        if not self.local_dir or not os.path.isdir(self.local_dir):
            return
        backup_root = "/root/autodl-fs"
        backup_name = os.path.basename(os.path.normpath(self.local_dir))
        if not backup_name:
            return
        backup_dir = os.path.join(backup_root, backup_name)
        if self._snapshot_ready(backup_dir):
            return
        os.makedirs(backup_dir, exist_ok=True)
        self._copy_tree(self.local_dir, backup_dir)

    @staticmethod
    def _snapshot_ready(path: str) -> bool:
        if not path or not os.path.isdir(path):
            return False
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_weights = False
        for fname in os.listdir(path):
            if fname.startswith("pytorch_model") or fname.endswith(".safetensors"):
                has_weights = True
                break
        return has_config and has_weights

    @staticmethod
    def _copy_tree(src: str, dst: str) -> None:
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            dest_root = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(dest_root, exist_ok=True)
            for fname in files:
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(dest_root, fname)
                if os.path.exists(dst_path):
                    try:
                        if os.path.getsize(dst_path) == os.path.getsize(src_path):
                            continue
                    except OSError:
                        pass
                shutil.copy2(src_path, dst_path)
