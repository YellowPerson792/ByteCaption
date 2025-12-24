import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

"""
python "tools\test_openrouter_api.py" \
    --image "PureT\data\coco_karpathy\test_sample_500\00000_id42.jpg" \
    --model 'anthropic/claude-haiku-4.5'
"""

def _resolve_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    key_path = args.api_key_path or os.getenv("OPENROUTER_API_KEY_PATH", "").strip()
    if not key_path:
        return ""
    path = Path(key_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _resolve_image_url(image: Optional[str]) -> Optional[str]:
    if not image:
        return None
    if image.startswith("data:image"):
        return image
    if image.startswith("http://") or image.startswith("https://"):
        return image
    path = Path(image)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return None
    mime, _ = mimetypes.guess_type(path.as_posix())
    if not mime:
        mime = "image/jpeg"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_text(data: dict) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                return "".join(parts)
            return str(content)
        if isinstance(choice.get("text"), str):
            return choice["text"]
    return ""


def _print_error(data: dict) -> None:
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return
    msg = err.get("message")
    code = err.get("code")
    meta = err.get("metadata") or {}
    provider = meta.get("provider_name") or meta.get("provider") or "unknown"
    raw = meta.get("raw")
    print(f"error.message: {msg}")
    print(f"error.code: {code}")
    print(f"provider: {provider}")
    if raw:
        print(f"provider_raw: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenRouter API probe (vision caption)")
    parser.add_argument("--model", default="openai/gpt-5.1")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-path", default="openrouter_api")
    parser.add_argument("--http-referer", default="")
    parser.add_argument("--app-title", default="ByteCaption")
    parser.add_argument("--image", default="")
    parser.add_argument("--prompt", default="Describe this image in one short COCO-style caption.")
    parser.add_argument("--system", default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--save-request", default="")
    parser.add_argument("--save-response", default="")
    args = parser.parse_args()

    key = _resolve_key(args)
    if not key:
        print("Missing OpenRouter API key. Set OPENROUTER_API_KEY or provide --api-key/--api-key-path.")
        return 2

    image_url = _resolve_image_url(args.image)
    if args.image and not image_url:
        print(f"Image not found or unreadable: {args.image}")
        return 2

    content = [{"type": "text", "text": args.prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }

    if args.save_request:
        Path(args.save_request).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.app_title:
        headers["X-Title"] = args.app_title

    req = urllib.request.Request(
        args.api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read()
        data = json.loads(body)
        if args.save_response:
            Path(args.save_response).write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        if args.print_json:
            print(json.dumps(data, ensure_ascii=True, indent=2))
        else:
            text = _extract_text(data)
            if text:
                print(text)
            else:
                print("No caption text returned.")
                _print_error(data)
        return 0
    except urllib.error.HTTPError as err:
        body = err.read()
        try:
            data = json.loads(body)
        except Exception:
            data = {"raw": body.decode("utf-8", errors="replace")}
        print(f"HTTP {err.code}")
        if args.print_json:
            print(json.dumps(data, ensure_ascii=True, indent=2))
        _print_error(data)
        return 3
    except urllib.error.URLError as err:
        print(f"Request failed: {err.reason}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
