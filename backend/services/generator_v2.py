"""Text generation with remote, local-model, and dependency-free providers."""

from __future__ import annotations

import random
import re
from typing import Any, Optional

import requests

from backend.config import AI_REQUEST_TIMEOUT, MODEL_PATH, TEXT_API_KEY, TEXT_API_MODEL, TEXT_API_URL
from backend.schemas import GenerateRequest

_HF_TOKENIZER: Any = None
_HF_MODEL: Any = None
_HF_MODEL_NAME: Optional[str] = None
_CATEGORIES = ("动物故事", "童话故事", "成长故事", "冒险故事", "温馨故事", "友谊故事", "想象故事")
_CATEGORY_HINTS = {
    "动物故事": ("动物", "小猫", "小狗", "兔", "狐狸", "熊", "鸟", "森林"),
    "童话故事": ("童话", "公主", "王子", "城堡", "魔法", "精灵", "仙女"),
    "成长故事": ("成长", "学校", "考试", "勇敢", "自信", "学习", "第一次"),
    "冒险故事": ("冒险", "探险", "宝藏", "海岛", "宇宙", "山洞", "航行"),
    "温馨故事": ("温馨", "家", "妈妈", "爸爸", "爷爷", "奶奶", "陪伴"),
    "友谊故事": ("友谊", "朋友", "伙伴", "分享", "合作", "和好"),
    "想象故事": ("想象", "会飞", "未来", "梦", "星球", "时间", "奇妙"),
}
_TARGET_LENGTHS = {"short": (180, 320), "medium": (420, 650), "long": (750, 1100)}


def _chat_endpoint(url: str) -> str:
    return url if url.rstrip("/").endswith("/chat/completions") else url.rstrip("/") + "/chat/completions"


def normalize_category(category: str, theme: str) -> str:
    cleaned = category.strip()
    if cleaned in _CATEGORIES:
        return cleaned
    for candidate, hints in _CATEGORY_HINTS.items():
        if any(hint in theme for hint in hints):
            return candidate
    return "想象故事"


def _payload_with_category(payload: GenerateRequest) -> GenerateRequest:
    category = normalize_category(payload.category, payload.theme)
    return payload if category == payload.category else payload.model_copy(update={"category": category})


def _build_prompt(payload: GenerateRequest) -> str:
    min_chars, max_chars = _TARGET_LENGTHS[payload.length]
    return (
        "你是一位优秀的中文儿童文学作家。请只输出故事正文，不要解释创作过程，也不要使用 Markdown 标题。\n"
        f"故事类型：{payload.category}\n主题：{payload.theme.strip()}\n主角：{payload.character.strip() or '小主角'}\n"
        f"篇幅：约 {min_chars}—{max_chars} 个中文字符\n额外要求：{payload.extra.strip() or '无'}\n"
        "要求：情节完整，有清晰的开端、转折、解决和温暖结尾；语言生动但易懂，适合 3—10 岁儿童；"
        "避免恐怖、暴力、危险模仿和说教口吻；角色行为前后一致；自然分段，并把道理融入行动。"
    )


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^(?:故事正文|故事|正文)\s*[：:]\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < 80:
        raise RuntimeError("model returned content that is too short")
    return text.strip()


def _generate_with_text_api(payload: GenerateRequest) -> str:
    if not (TEXT_API_URL and TEXT_API_MODEL):
        raise RuntimeError("text API is not configured")
    headers = {"Content-Type": "application/json"}
    if TEXT_API_KEY:
        headers["Authorization"] = f"Bearer {TEXT_API_KEY}"
    response = requests.post(
        _chat_endpoint(TEXT_API_URL),
        headers=headers,
        json={
            "model": TEXT_API_MODEL,
            "messages": [
                {"role": "system", "content": "你专门创作安全、温暖、有想象力的中文儿童故事。"},
                {"role": "user", "content": _build_prompt(payload)},
            ],
            "temperature": 0.82,
            "top_p": 0.92,
            "max_tokens": {"short": 500, "medium": 1000, "long": 1800}[payload.length],
        },
        timeout=AI_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("text API returned an unsupported response") from exc
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return _clean_generated_text(str(content))


def _generate_with_huggingface(payload: GenerateRequest) -> str:
    global _HF_TOKENIZER, _HF_MODEL, _HF_MODEL_NAME
    if not MODEL_PATH:
        raise RuntimeError("TINYLLM_MODEL_PATH is not configured")
    if _HF_MODEL is None or _HF_MODEL_NAME != MODEL_PATH:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _HF_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _HF_MODEL = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _HF_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")
        _HF_MODEL.eval()
        _HF_MODEL_NAME = MODEL_PATH
    prompt = _build_prompt(payload)
    if getattr(_HF_TOKENIZER, "chat_template", None):
        prompt = _HF_TOKENIZER.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    inputs = _HF_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(_HF_MODEL.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    eos_token_id = _HF_TOKENIZER.eos_token_id
    pad_token_id = _HF_TOKENIZER.pad_token_id if _HF_TOKENIZER.pad_token_id is not None else eos_token_id
    output = _HF_MODEL.generate(
        **inputs,
        max_new_tokens={"short": 320, "medium": 700, "long": 1200}[payload.length],
        do_sample=True,
        temperature=0.82,
        top_p=0.92,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    generated = _HF_TOKENIZER.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return _clean_generated_text(generated)


def _generate_local_story(payload: GenerateRequest) -> str:
    rng = random.SystemRandom()
    character = payload.character.strip() or rng.choice(("小雨", "安安", "米粒", "团团", "小星"))
    theme = payload.theme.strip()
    settings = {
        "动物故事": ("晨雾还没散去的森林", "铺满蒲公英的河岸", "热闹的橡果村"),
        "童话故事": ("云朵托起的城堡", "月光闪烁的魔法花园", "会唱歌的玻璃小镇"),
        "成长故事": ("阳光落进窗台的教室", "放学后的安静操场", "第一次独自出发的清晨"),
        "冒险故事": ("藏着旧地图的海边灯塔", "星光照亮的山谷入口", "漂浮在云海上的小岛"),
        "温馨故事": ("飘着饭菜香的小屋", "雨点轻敲窗户的傍晚", "开满向日葵的院子"),
        "友谊故事": ("热闹的森林运动会", "新学期开学的第一天", "小河边的木工坊"),
        "想象故事": ("钟表会倒着走的城市", "离月亮很近的屋顶", "藏在口袋里的微型星球"),
    }
    helpers = ("爱观察的松鼠点点", "总带着放大镜的小禾", "说话慢吞吞的乌龟阿稳", "会修东西的机器人小咔")
    objects = ("一枚温热的蓝色纽扣", "一张只在月光下显字的纸条", "一颗会随心情变色的种子", "一只迷路的纸飞机")
    setting, helper, clue = rng.choice(settings[payload.category]), rng.choice(helpers), rng.choice(objects)
    paragraphs = [
        f"在{setting}，{character}最喜欢研究“{theme}”。别人觉得这只是一个普通念头，{character}却把每天的新发现画进小本子里。那天清晨，{clue}落在了本子的正中央，旁边还留着一行会发光的小字。",
        f"{character}顺着线索出发，很快遇见了{helper}。他们没有急着往前冲，而是先看脚印、听风声，再把各自知道的事情拼在一起。原来，前方的路被一道奇怪的难题挡住了：只有真正理解“{theme}”的人，才能找到看不见的入口。",
        f"第一次尝试失败了。{character}的脸一下子热了起来，甚至想把本子合上。{helper}没有替{character}完成，只指着刚才留下的痕迹说：“失败也会告诉我们下一步往哪里走。”于是他们重新检查细节，发现自己一直忽略了最安静的那条线索。",
        f"这一次，{character}先提出办法，再认真听伙伴补充。他们一个负责观察，一个负责试验，还约定遇到不确定的地方就停下来商量。随着最后一块小机关轻轻转动，眼前出现了一条洒满金色光点的小路，路的尽头正藏着关于“{theme}”的秘密。",
        f"秘密并不是耀眼的宝箱，而是一份需要大家共同完成的礼物。{character}终于明白，勇敢不是从来不害怕，而是害怕时仍愿意想办法；聪明也不是一个人知道所有答案，而是懂得和伙伴合作。",
        f"回去的路上，他们把经历讲给沿途的朋友听，还把有用的线索留给后来的人。夕阳把影子拉得长长的，{character}在小本子最后写道：“今天得到的最好礼物，是我已经比出发时更勇敢、更懂得倾听。”",
        f"夜色降临时，他们在灯塔下举办了一场小小的分享会。{character}把那份礼物拆成许多份，送给每一位愿意倾听的人。有人带来了新问题，有人画下了更远的地图，原本只属于两个人的秘密，变成了大家一起守护的约定。",
        f"第二天，{character}再次翻开小本子，发现最后一页多了一行字：真正的“{theme}”，会在你愿意迈出第一步、也愿意为别人留一盏灯时发生。{character}笑了起来，决定把这段旅程继续写下去。",
    ]
    if payload.extra.strip():
        paragraphs.insert(4, f"{payload.extra.strip().rstrip('。！？')}。这个特别的约定没有让旅程变得轻松，却让{character}想出了一个更有创意、也更照顾伙伴的办法。")
    min_chars, max_chars = _TARGET_LENGTHS[payload.length]
    chosen: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(chosen + [paragraph])
        if len(candidate) > max_chars and len("\n\n".join(chosen)) >= min_chars:
            break
        chosen.append(paragraph)
        if len(candidate) >= min_chars:
            break
    text = "\n\n".join(chosen or paragraphs)
    if len(text) < min_chars:
        text = "\n\n".join(paragraphs)
    text = text[:max_chars].rstrip("，；：")
    return text if text.endswith(("。", "！", "？")) else text + "。"


def generate_content(payload: GenerateRequest) -> tuple[str, str]:
    payload = _payload_with_category(payload)
    providers = []
    if TEXT_API_URL and TEXT_API_MODEL:
        providers.append((_generate_with_text_api, "openai-compatible"))
    if MODEL_PATH:
        providers.append((_generate_with_huggingface, "huggingface"))
    for provider, name in providers:
        try:
            return provider(payload), name
        except Exception as exc:
            print(f"{name} generation unavailable, using next provider: {exc}")
    return _generate_local_story(payload), "local-story-engine"
