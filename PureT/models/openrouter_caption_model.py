import base64
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageFile

from lib.config import cfg


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except Exception:
            return None
    return None


def _coerce_optional_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except Exception:
            return None
    return None


def _normalize_stop_sequences(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if str(v).strip()]
        return items if items else None
    return None


def _coerce_image_bytes(item) -> Optional[bytes]:
    if item is None:
        return None
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, Image.Image):
        buf = BytesIO()
        item.save(buf, format="JPEG", quality=60)
        return buf.getvalue()
    return None


def _normalize_caption(text: str, placeholder: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return placeholder
    if raw[0] in "{[":
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in ("caption", "text", "output"):
                    if isinstance(parsed.get(key), str):
                        raw = parsed[key]
                        break
            elif isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, str):
                    raw = first
                elif isinstance(first, dict):
                    for key in ("caption", "text", "output"):
                        if isinstance(first.get(key), str):
                            raw = first[key]
                            break
        except Exception:
            pass
    cleaned = " ".join(str(raw).strip().strip('"').strip("'").split())
    if not cleaned:
        return placeholder
    if cleaned == placeholder:
        return placeholder
    if cleaned.endswith(".") and cleaned[:-1].strip() == placeholder:
        return placeholder
    return cleaned


class OpenRouterCaptionModel(nn.Module):
    """
    OpenRouter API captioning wrapper.
    Accepts a list of corrupted JPEG byte streams via cfg.PARAM.ATT_FEATS.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_path: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        super().__init__()
        or_cfg = getattr(cfg.MODEL, "OPENROUTER", None)
        self.model_id = model_id or (or_cfg.MODEL_ID if or_cfg else "openai/gpt-5.1")
        self.api_base = api_base or (or_cfg.API_BASE if or_cfg else "https://openrouter.ai/api/v1/chat/completions")
        self.api_key = api_key or (or_cfg.API_KEY if or_cfg else "")
        self.api_key_path = api_key_path or (or_cfg.API_KEY_PATH if or_cfg else "openrouter_api")
        self.http_referer = (or_cfg.HTTP_REFERER if or_cfg else "") or ""
        self.app_title = (or_cfg.APP_TITLE if or_cfg else "") or ""
        self.proxy = (or_cfg.PROXY if or_cfg else "") or ""
        self.timeout = float(or_cfg.TIMEOUT if or_cfg else 60)
        self.max_tokens = int(or_cfg.MAX_TOKENS if or_cfg else 64)
        self.temperature = float(or_cfg.TEMPERATURE if or_cfg else 0.0)
        self.top_p = float(or_cfg.TOP_P if or_cfg else 1.0)
        self.max_workers = int(max_workers if max_workers is not None else (or_cfg.MAX_WORKERS if or_cfg else 6))
        self.request_batch_size = int(or_cfg.BATCH_SIZE if or_cfg else 1)
        self.placeholder = (or_cfg.PLACEHOLDER if or_cfg else "this is a dummy caption for an undecodable image").strip()
        self.system_prompt = (or_cfg.SYSTEM_PROMPT if or_cfg else "").strip()
        self.user_prompt = (or_cfg.USER_PROMPT if or_cfg else "").strip()
        self.reasoning = getattr(or_cfg, "REASONING", None) if or_cfg else None
        self.frequency_penalty = _coerce_optional_float(getattr(or_cfg, "FREQUENCY_PENALTY", None) if or_cfg else None)
        self.presence_penalty = _coerce_optional_float(getattr(or_cfg, "PRESENCE_PENALTY", None) if or_cfg else None)
        self.repetition_penalty = _coerce_optional_float(
            getattr(or_cfg, "REPETITION_PENALTY", None) if or_cfg else None
        )
        self.min_p = _coerce_optional_float(getattr(or_cfg, "MIN_P", None) if or_cfg else None)
        self.top_k = _coerce_optional_int(getattr(or_cfg, "TOP_K", None) if or_cfg else None)
        self.seed = _coerce_optional_int(getattr(or_cfg, "SEED", None) if or_cfg else None)
        self.stop_sequences = _normalize_stop_sequences(getattr(or_cfg, "STOP", None) if or_cfg else None)
        self.response_format = getattr(or_cfg, "RESPONSE_FORMAT", None) if or_cfg else None
        self.logprobs = getattr(or_cfg, "LOGPROBS", None) if or_cfg else None
        self.top_logprobs = _coerce_optional_int(getattr(or_cfg, "TOP_LOGPROBS", None) if or_cfg else None)
        self.image_detail = (getattr(or_cfg, "IMAGE_DETAIL", None) if or_cfg else None) or ""
        self.extra_headers = getattr(or_cfg, "EXTRA_HEADERS", None) if or_cfg else None
        self.extra_payload = getattr(or_cfg, "EXTRA_PAYLOAD", None) if or_cfg else None
        retry_cfg = getattr(or_cfg, "RETRY", None) if or_cfg else None
        self.retry_attempts = int(retry_cfg.MAX_ATTEMPTS) if retry_cfg else 3
        self.retry_backoff_base = float(retry_cfg.BACKOFF_BASE) if retry_cfg else 1.5
        self.retry_backoff_max = float(retry_cfg.BACKOFF_MAX) if retry_cfg else 20.0
        self.retry_on_empty = bool(getattr(retry_cfg, "ON_EMPTY_RESPONSE", True)) if retry_cfg else True
        self.retry_on_truncated = bool(getattr(retry_cfg, "ON_TRUNCATED", True)) if retry_cfg else True
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self._seen_errors = set()
        self._opener = self._build_opener(self.proxy)

        if not self.placeholder:
            self.placeholder = "this is a dummy caption for an undecodable image"

        if not self.user_prompt:
            self.user_prompt = (
                "You are given a possibly corrupted JPEG image. "
                "If you can decode it, output a short COCO-style caption. "
                f"If you cannot decode it, output exactly: {self.placeholder} "
                "Output only the caption with no extra text."
            )

        self.api_key = self.api_key or self._resolve_api_key()
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY or provide MODEL.OPENROUTER.API_KEY/_PATH."
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("OpenRouterCaptionModel is inference-only in this pipeline.")

    def decode_beam(self, **kwargs) -> Tuple[List[str], None]:
        images = kwargs[cfg.PARAM.ATT_FEATS]
        outputs = [self.placeholder for _ in range(len(images))]
        if not images:
            return outputs, None

        indexed_urls = []
        for idx, item in enumerate(images):
            if item is None:
                continue
            if isinstance(item, (bytes, bytearray, memoryview)):
                if not self._is_decodable_bytes(bytes(item)):
                    continue
            data_url = self._to_data_url(item)
            if not data_url:
                continue
            indexed_urls.append((idx, data_url))

        if not indexed_urls:
            return outputs, None

        batch_size = self.request_batch_size if self.request_batch_size is not None else 1
        if batch_size <= 0:
            batch_size = len(indexed_urls)
        else:
            batch_size = max(1, batch_size)
        if self.max_workers <= 1 or len(indexed_urls) == 1:
            for idx, url in indexed_urls:
                caption = self._request_caption(url)
                outputs[idx] = _normalize_caption(caption, self.placeholder)
            return outputs, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for start in range(0, len(indexed_urls), batch_size):
                chunk = indexed_urls[start : start + batch_size]
                future_map = {executor.submit(self._request_caption, url): idx for idx, url in chunk}
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    caption = fut.result()
                    outputs[idx] = _normalize_caption(caption, self.placeholder)

        return outputs, None

    def decode(self, **kwargs):
        kwargs["BEAM_SIZE"] = 1
        return self.decode_beam(**kwargs)

    def _resolve_api_key(self) -> str:
        env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if env_key:
            return env_key
        key_path = os.getenv("OPENROUTER_API_KEY_PATH", "").strip() or self.api_key_path
        if not key_path:
            return ""
        path = Path(key_path)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(Path.cwd() / path)
            candidates.append(_PROJECT_ROOT / path)
        for candidate in candidates:
            if candidate.exists():
                try:
                    return candidate.read_text(encoding="utf-8").strip()
                except Exception:
                    continue
        return ""

    def _to_data_url(self, item) -> Optional[str]:
        if isinstance(item, str):
            if item.startswith("data:image"):
                return item
            return f"data:image/jpeg;base64,{item}"
        raw = _coerce_image_bytes(item)
        if not raw:
            return None
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _is_decodable_bytes(self, raw: bytes) -> bool:
        if not raw:
            return False
        old_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(BytesIO(raw)) as img:
                img.load()
            return True
        except Exception:
            return False
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old_truncated

    def _build_payload(self, image_url: str) -> dict:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            **({"detail": self.image_detail} if self.image_detail else {}),
                        },
                    },
                ],
            }
        )
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stop_sequences:
            payload["stop"] = self.stop_sequences
        if self.response_format:
            payload["response_format"] = self.response_format
        if self.logprobs is not None:
            payload["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            payload["top_logprobs"] = self.top_logprobs
        reasoning = self._resolve_reasoning()
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if isinstance(self.extra_payload, dict) and self.extra_payload:
            payload.update(self.extra_payload)
        return payload

    def _resolve_reasoning(self):
        value = self.reasoning
        if value is None:
            return None
        if isinstance(value, bool):
            return {"effort": "none"} if not value else None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if not lowered:
                return None
            if lowered in ("0", "false", "off", "disabled", "none"):
                return {"effort": "none"}
            return {"effort": lowered}
        if isinstance(value, dict):
            # Support legacy/authoring-friendly schema:
            # - {enabled: false} -> effort none
            # - {enabled: true, effort: low} -> effort low
            enabled = value.get("enabled", None)
            if isinstance(enabled, bool):
                if not enabled:
                    return {"effort": "none"}
                effort = value.get("effort", None)
                if isinstance(effort, str) and effort.strip():
                    return {"effort": effort.strip().lower()}
                # enabled=true but no effort specified -> omit to use provider default
                return None
            # If already in the provider schema, pass through.
            return value
        return None

    def _request_caption(self, image_url: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        if isinstance(self.extra_headers, dict) and self.extra_headers:
            for key, value in self.extra_headers.items():
                if value is None:
                    continue
                header_value = str(value).strip()
                if not header_value:
                    continue
                headers[str(key)] = header_value

        payload = self._build_payload(image_url)
        max_tokens = int(payload.get("max_tokens") or self.max_tokens)
        max_tokens_cap = max(self.max_tokens, 256)

        attempt = 0
        last_error = ""
        while attempt < self.retry_attempts:
            attempt += 1
            payload["max_tokens"] = max_tokens
            payload_bytes = json.dumps(payload).encode("utf-8")
            try:
                req = urllib.request.Request(self.api_base, data=payload_bytes, headers=headers, method="POST")
                if self._opener is not None:
                    open_fn = self._opener.open
                else:
                    open_fn = urllib.request.urlopen
                with open_fn(req, timeout=self.timeout) as resp:
                    body = resp.read()
                    resp_headers = resp.headers
                raw_text = body.decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw_text)
                except Exception:
                    last_error = "invalid_json"
                    self._log_api_error("invalid_json", raw_text)
                    self._sleep_backoff_if_retry(attempt, None)
                    continue
                if isinstance(data, dict) and data.get("error"):
                    retry_after = None
                    if resp_headers is not None:
                        retry_after = resp_headers.get("Retry-After")
                    last_error = self._format_error(data)
                    self._log_api_error("api_error", data)
                    if self._is_retryable_error(data):
                        self._sleep_backoff_if_retry(attempt, retry_after)
                        continue
                    return ""
                choice_error = self._extract_choice_error(data)
                if choice_error is not None:
                    retry_after = None
                    if resp_headers is not None:
                        retry_after = resp_headers.get("Retry-After")
                    last_error = self._format_error(choice_error)
                    self._log_api_error("choice_error", choice_error)
                    if self._is_retryable_error(choice_error):
                        self._sleep_backoff_if_retry(attempt, retry_after)
                        continue
                    return ""
                caption = self._extract_caption(data)
                if caption:
                    return caption
                finish_reason, native_finish_reason = self._extract_finish_reasons(data)
                if self._is_truncated_finish(finish_reason, native_finish_reason):
                    last_error = "truncated_output"
                    self._log_api_error(
                        "truncated_output",
                        {
                            "finish_reason": finish_reason,
                            "native_finish_reason": native_finish_reason,
                            "max_tokens": max_tokens,
                        },
                    )
                    if not self.retry_on_truncated:
                        return ""
                    if max_tokens < max_tokens_cap:
                        max_tokens = min(max_tokens_cap, max_tokens * 2)
                    else:
                        return ""
                    self._sleep_backoff_if_retry(attempt, None)
                    continue
                if raw_text:
                    self._log_api_error("missing_choices", raw_text)
                last_error = "empty_response"
                if not self.retry_on_empty:
                    return ""
                self._sleep_backoff_if_retry(attempt, None)
            except urllib.error.HTTPError as err:
                try:
                    err_body = err.read().decode("utf-8", errors="replace")
                    err_data = json.loads(err_body) if err_body else None
                except Exception:
                    err_body = ""
                    err_data = None
                last_error = f"HTTP {err.code}"
                if err_data:
                    self._log_api_error(err.code, err_data)
                else:
                    self._log_api_error(err.code, err_body)
                if err.code in (429, 500, 502, 503, 504):
                    self._sleep_backoff_if_retry(attempt, err.headers.get("Retry-After"))
                    continue
                return ""
            except urllib.error.URLError as err:
                last_error = str(err.reason)
                self._sleep_backoff_if_retry(attempt, None)
            except Exception as err:
                last_error = str(err)
                self._sleep_backoff_if_retry(attempt, None)

        if last_error:
            try:
                self.logger.warning("OpenRouter request failed after retries: %s", last_error)
            except Exception:
                pass
        return ""

    def _build_opener(self, proxy: str):
        if not proxy:
            return None
        proxy_url = proxy.strip()
        if not proxy_url:
            return None
        proxies = {"http": proxy_url, "https": proxy_url}
        try:
            handler = urllib.request.ProxyHandler(proxies)
            return urllib.request.build_opener(handler)
        except Exception:
            return None

    def _log_api_error(self, status, body) -> None:
        if not body:
            return
        preview = body
        if isinstance(body, dict):
            preview = json.dumps(body, ensure_ascii=True)[:200]
        key = (str(status), preview)
        if key in self._seen_errors:
            return
        self._seen_errors.add(key)
        try:
            self.logger.error("OpenRouter API error %s: %s", status, preview)
        except Exception:
            pass

    def _format_error(self, data: dict) -> str:
        err = data.get("error") if isinstance(data, dict) else None
        if not isinstance(err, dict):
            return "api_error"
        code = err.get("code") or "unknown"
        message = err.get("message") or ""
        return f"{code}:{message}".strip(":")

    def _is_retryable_error(self, data: dict) -> bool:
        err = data.get("error") if isinstance(data, dict) else None
        if not isinstance(err, dict):
            return False
        code = str(err.get("code") or "").lower()
        message = str(err.get("message") or "").lower()
        raw = ""
        meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else None
        if meta and isinstance(meta.get("raw"), str):
            raw = meta["raw"].lower()
        combined = " ".join([code, message, raw])
        if any(
            token in combined
            for token in (
                "rate",
                "limit",
                "overload",
                "timeout",
                "temporarily",
                "busy",
                "server_error",
                "gateway",
                "bad_gateway",
                "unavailable",
                "502",
                "503",
            )
        ):
            return True
        return False

    def _extract_choice_error(self, data: dict):
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        if "error" in choice and isinstance(choice["error"], dict):
            return {"error": choice["error"]}
        return None

    def _extract_finish_reasons(self, data: dict) -> Tuple[str, str]:
        if not isinstance(data, dict):
            return "", ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return "", ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return "", ""
        finish_reason = str(choice.get("finish_reason") or "")
        native_finish_reason = str(choice.get("native_finish_reason") or "")
        return finish_reason, native_finish_reason

    def _is_truncated_finish(self, finish_reason: str, native_finish_reason: str) -> bool:
        for reason in (finish_reason, native_finish_reason):
            if str(reason).lower() in ("length", "max_output_tokens"):
                return True
        return False

    def _extract_caption(self, data: dict) -> str:
        if isinstance(data, dict) and data.get("error"):
            return ""
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

    def _sleep_backoff_if_retry(self, attempt: int, retry_after: Optional[str]) -> None:
        if attempt >= self.retry_attempts:
            return
        self._sleep_backoff(attempt, retry_after)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
                time.sleep(delay)
                return
            except Exception:
                pass
        base = self.retry_backoff_base ** max(attempt - 1, 0)
        delay = min(self.retry_backoff_max, base + random.random())
        time.sleep(delay)
