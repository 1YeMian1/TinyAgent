"""Story illustration providers and a safe local SVG fallback."""

from __future__ import annotations

import base64
import html
import uuid
from typing import Any
from urllib.parse import quote

import requests

from backend.config import (
    AI_REQUEST_TIMEOUT,
    GENERATED_IMAGE_DIR,
    IMAGE_API_KEY,
    IMAGE_API_MODEL,
    IMAGE_API_URL,
    IMAGE_SIZE,
)


def _image_endpoint(url: str) -> str:
    return url if url.rstrip("/").endswith("/images/generations") else url.rstrip("/") + "/images/generations"


def _prompt_for(story: Any, index: int) -> str:
    excerpt = " ".join(story.content.split())[:220]
    scene = ("开端场景" if index == 0 else "关键转折" if index == 1 else "温暖结尾")
    return (
        f"儿童绘本插画，{scene}。故事标题：{story.title}。故事类型：{story.category}。"
        f"故事片段：{excerpt}。画面明快温馨、圆润手绘、水彩质感、适合3到10岁儿童，"
        "无文字、无水印、无恐怖元素，主体清晰，16:9构图。"
    )


def _remote_images(story: Any, count: int) -> list[str]:
    if not (IMAGE_API_URL and IMAGE_API_MODEL):
        raise RuntimeError("image API is not configured")
    headers = {"Content-Type": "application/json"}
    if IMAGE_API_KEY:
        headers["Authorization"] = f"Bearer {IMAGE_API_KEY}"
    response = requests.post(
        _image_endpoint(IMAGE_API_URL),
        headers=headers,
        json={"model": IMAGE_API_MODEL, "prompt": _prompt_for(story, 0), "n": count, "size": IMAGE_SIZE},
        timeout=AI_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    if not isinstance(data, list):
        raise RuntimeError("image API returned an unsupported response")
    urls: list[str] = []
    for item in data[:count]:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            urls.append(url)
            continue
        encoded = item.get("b64_json")
        if isinstance(encoded, str) and encoded:
            urls.append(_save_base64_image(encoded, story.id))
    if not urls:
        raise RuntimeError("image API returned no usable images")
    return urls


def _save_base64_image(encoded: str, story_id: int) -> str:
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 12 * 1024 * 1024:
        raise RuntimeError("generated image is too large")
    if raw.startswith(b"\x89PNG"):
        suffix = "png"
    elif raw.startswith(b"\xff\xd8\xff"):
        suffix = "jpg"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        suffix = "webp"
    else:
        raise RuntimeError("generated image format is not supported")
    folder = GENERATED_IMAGE_DIR / f"story-{int(story_id)}"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{suffix}"
    (folder / filename).write_bytes(raw)
    return f"/generated/story-{int(story_id)}/{filename}"


def _local_svg(story: Any, index: int) -> str:
    """Create an attractive, deterministic illustration without external services."""
    palettes = (("#FFE4B5", "#FF9A9E", "#8EC5FC"), ("#D4FC79", "#96E6A1", "#84FAB0"), ("#C2FFD8", "#465EFB", "#B8C6DB"))
    sky, hill, accent = palettes[index % len(palettes)]
    title = html.escape(str(story.title)[:22])
    category = html.escape(str(story.category))
    cx = 170 + index * 70
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" role="img" aria-label="{title}">
<defs><linearGradient id="sky" x1="0" y1="0" x2="1" y2="1"><stop stop-color="{sky}"/><stop offset="1" stop-color="{accent}"/></linearGradient></defs>
<rect width="800" height="450" rx="28" fill="url(#sky)"/><circle cx="680" cy="90" r="48" fill="#FFF6B7" opacity=".9"/>
<path d="M0 330 Q140 220 280 330 T560 320 T800 300 V450 H0Z" fill="{hill}" opacity=".9"/>
<path d="M0 370 Q180 290 360 370 T800 350 V450 H0Z" fill="#FFFFFF" opacity=".36"/>
<g transform="translate({cx},220)"><circle r="66" fill="#FFF8E7"/><circle cx="-23" cy="-8" r="8" fill="#394867"/><circle cx="23" cy="-8" r="8" fill="#394867"/><path d="M-24 22 Q0 42 24 22" fill="none" stroke="#E07A5F" stroke-width="7" stroke-linecap="round"/><path d="M-52 -48 L-78 -92 L-20 -64 M52 -48 L78 -92 L20 -64" fill="#FFF8E7" stroke="#E07A5F" stroke-width="6"/></g>
<text x="36" y="52" font-family="sans-serif" font-size="23" font-weight="700" fill="#394867">{title}</text><text x="38" y="84" font-family="sans-serif" font-size="15" fill="#394867" opacity=".8">{category} · TinyLLM-Story</text>
</svg>'''
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


def generate_story_images(story: Any, count: int = 3) -> tuple[list[str], str]:
    count = max(1, min(int(count), 8))
    if IMAGE_API_URL and IMAGE_API_MODEL:
        try:
            return _remote_images(story, count), "openai-compatible"
        except Exception as exc:
            print(f"image API unavailable, using local illustrations: {exc}")
    return [_local_svg(story, index) for index in range(count)], "local-svg-illustration"
