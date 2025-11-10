# app.py
import os
import io
import re
import json
import uuid
import base64
import tempfile
from typing import Optional, List, Dict, Any

import torch
import torchaudio
import httpx
import asyncio

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer, WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token

from flow_inference import AudioDecoder

# =========================
# Config
# =========================
HOST = "0.0.0.0"
PORT = 8888
DEVICE = "cuda"
SAMPLE_RATE = 22050
VLLM_STREAM_URL = "http://localhost:10000/generate_stream"

FLOW_PATH = "/root/model/glm4-voice/glm-4-voice-decoder"
MODEL_PATH = "/root/model/glm4-voice/glm-4-voice-9b"
TOKENIZER_PATH = "/root/model/glm4-voice/glm-4-voice-tokenizer"

AUDIO_CHUNK_TOKENS = 64  # 累积多少个音频 token 解一次码，越小首帧越快

# =========================
# App & Static
# =========================
app = FastAPI(title="GLM-4-Voice (FastAPI + HTML + WS Streaming)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请收敛
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("static/outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================
# Globals (lazy init on startup)
# =========================
audio_decoder: Optional[AudioDecoder] = None
glm_tokenizer: Optional[AutoTokenizer] = None
whisper_model: Optional[WhisperVQEncoder] = None
feature_extractor: Optional[WhisperFeatureExtractor] = None

# special tokens / ids
EOS_ID: Optional[int] = None
ASSISTANT_ID: Optional[int] = None
USER_ID: Optional[int] = None

# 可选：在请求里作为 stop_token_ids 传给 vLLM（按你的模板调整）
EXTRA_STOP_TOKENS: List[str] = [
    # "<|assistant|>",   # 举例：若你的解码会在助手起始标记前停止，可放开
    # "<|user|>",        # 谨慎启用，避免提前截断
]


@app.on_event("startup")
def _startup() -> None:
    global audio_decoder, glm_tokenizer, whisper_model, feature_extractor
    global EOS_ID, ASSISTANT_ID, USER_ID

    # 1) 文本 tokenizer（GLM-4-Voice）
    glm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    EOS_ID = getattr(glm_tokenizer, "eos_token_id", None)
    ASSISTANT_ID = glm_tokenizer.convert_tokens_to_ids("<|assistant|>")
    USER_ID = glm_tokenizer.convert_tokens_to_ids("<|user|>")

    # 2) Flow + HiFT 语音解码器
    audio_decoder = AudioDecoder(
        config_path=os.path.join(FLOW_PATH, "config.yaml"),
        flow_ckpt_path=os.path.join(FLOW_PATH, "flow.pt"),
        hift_ckpt_path=os.path.join(FLOW_PATH, "hift.pt"),
        device=DEVICE,
    )

    # 3) Whisper VQ 编码器（把波形 -> 音频离散 token）
    whisper_model = WhisperVQEncoder.from_pretrained(TOKENIZER_PATH).eval().to(DEVICE)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(TOKENIZER_PATH)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/clear")
def clear_state():
    # 如果需要清理服务端缓存状态，可在这里补充
    return JSONResponse({"ok": True})


def wav_bytes_from_tensor(
    wave_tensor: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> bytes:
    """
    将 (T,) 或 (1, T) 的张量保存为内存中的 WAV bytes
    """
    if wave_tensor.ndim == 1:
        wave = wave_tensor.detach().cpu().unsqueeze(0)  # (1, T)
    elif wave_tensor.ndim == 2 and wave_tensor.shape[0] in (1, 2):
        wave = wave_tensor.detach().cpu()
    else:
        wave = wave_tensor.detach().cpu().unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, wave, sample_rate, format="wav")
    return buf.getvalue()


def parse_sse_or_json_line(raw_line: bytes) -> Optional[Dict[str, Any]]:
    """
    兼容两种常见流式格式：
    1) 纯 NDJSON：  b'{"token_id": 123, ...}'
    2) SSE：        b'data: {"token_id": 123, ...}'
    其余行（空行/注释）返回 None
    """
    if not raw_line:
        return None
    line = raw_line.decode("utf-8", errors="ignore").strip()
    if not line:
        return None
    if line.startswith("data:"):
        line = line[len("data:"):].strip()
    if not line or line in ("[DONE]", "DONE"):
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def token_is_audio(token_id: int) -> Optional[int]:
    """
    更稳妥地识别音频 token：
    通过反查 token 字符串，匹配 <|audio_{n}|> 模式，并返回 n（int）。
    若不是音频 token 返回 None。
    """
    if glm_tokenizer is None:
        return None
    try:
        s = glm_tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        return None
    if not s or not isinstance(s, str):
        return None
    # 形如 "<|audio_123|>"
    if s.startswith("<|audio_") and s.endswith("|>"):
        m = re.match(r"^<\|audio_(\d+)\|\>$", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


async def stream_from_vllm(
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> asyncio.Queue:
    """
    异步任务：拉取 vLLM 流式输出，并将解析后的 payload 放入 queue。
    由 WS 协程消费该 queue。
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=0)

    async def _runner():
        stop_ids: List[int] = []
        for t in EXTRA_STOP_TOKENS:
            try:
                tid = glm_tokenizer.convert_tokens_to_ids(t)
                if isinstance(tid, int) and tid >= 0:
                    stop_ids.append(tid)
            except Exception:
                pass
        if EOS_ID is not None and EOS_ID not in stop_ids:
            # 不把 EOS 放 stop 里也没问题，后面消费端也会识别 EOS 提前结束
            pass

        payload = {
            "prompt": prompt,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
        }
        if stop_ids:
            payload["stop_token_ids"] = stop_ids

        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream("POST", VLLM_STREAM_URL, json=payload) as r:
                    async for raw_line in r.aiter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        if not line or line in ("[DONE]", "DONE"):
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        await queue.put(data)
            except Exception as e:
                await queue.put({"__error__": str(e)})

        await queue.put({"__closed__": True})

    asyncio.create_task(_runner())
    return queue


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        # 1) 初始化消息（文本 json）
        init_msg = await ws.receive_text()
        init = json.loads(init_msg)
        mode = init.get("mode", "text")
        temperature = float(init.get("temperature", 0.2))
        top_p = float(init.get("top_p", 0.8))
        max_new_tokens = int(init.get("max_new_tokens", 2000))
        prev_input = init.get("previous_input_tokens", "") or ""
        prev_completion = init.get("previous_completion_tokens", "") or ""
        text_input = init.get("text") or None

        # 2) 构造 user_input & system_prompt
        system_prompt = (
            "User will provide you with an instruction. "
            "Respond in an interleaved manner: after ~13 text tokens, emit ~26 audio tokens, then continue."
        )

        if mode == "audio":
            # 紧接着接收一帧二进制音频
            bin_frame = await ws.receive_bytes()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                f.write(bin_frame)
                audio_path = f.name
            try:
                audio_tokens = extract_speech_token(
                    whisper_model, feature_extractor, [audio_path]
                )[0]
            except Exception as e:
                await ws.send_text(
                    json.dumps({"type": "error", "message": f"extract_speech_token failed: {e}"})
                )
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
                await ws.close()
                return
            finally:
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

            if not audio_tokens:
                await ws.send_text(json.dumps({"type": "error", "message": "No audio tokens extracted."}))
                await ws.close()
                return

            audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            user_input = f"<|begin_of_audio|>{audio_tokens_str}<|end_of_audio|>"
        else:
            if not text_input:
                await ws.send_text(json.dumps({"type": "error", "message": "No text input."}))
                await ws.close()
                return
            user_input = text_input

        # 3) 拼装 prompt（含历史）
        inputs = (prev_input + prev_completion).strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        # 4) 异步启动 vLLM 流式拉取
        queue = await stream_from_vllm(
            prompt=inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        # 5) 消费 queue，分类处理 token
        text_token_ids: List[int] = []
        audio_token_acc: List[int] = []
        complete_token_ids: List[int] = []

        async def flush_audio(finalize: bool):
            if not audio_token_acc:
                return
            with torch.no_grad():
                tts_token = torch.tensor(audio_token_acc, device=DEVICE).unsqueeze(0)
                wav, _ = audio_decoder.token2wav(
                    tts_token,
                    uuid=str(uuid.uuid4()),
                    prompt_token=torch.zeros(1, 0, dtype=torch.int64, device=DEVICE),
                    prompt_feat=torch.zeros(1, 0, 80, device=DEVICE),
                    finalize=finalize,
                )
            wav_bytes = wav_bytes_from_tensor(wav.squeeze(0))
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            await ws.send_text(json.dumps({"type": "audio_chunk", "wav_b64": b64}))
            audio_token_acc.clear()

        stop_now = False
        while True:
            item = await queue.get()
            if "__error__" in item:
                await ws.send_text(json.dumps({"type": "error", "message": item["__error__"]}))
                break
            if "__closed__" in item:
                # vLLM HTTP 流结束；把尾部音频 flush + 退出
                await flush_audio(finalize=True)
                break

            # 兼容不同字段名；至少需要 token_id
            token_id = item.get("token_id", None)
            if token_id is None:
                continue

            # 结束条件：遇到 eos
            if EOS_ID is not None and token_id == EOS_ID:
                await flush_audio(finalize=True)
                stop_now = True
                break

            complete_token_ids.append(token_id)

            # 判断是否为音频 token
            audio_idx = token_is_audio(token_id)
            if audio_idx is not None:
                audio_token_acc.append(audio_idx)
                if len(audio_token_acc) >= AUDIO_CHUNK_TOKENS:
                    await flush_audio(finalize=False)
                continue

            # 文本 token：增量解码回推
            try:
                delta = glm_tokenizer.decode([token_id], spaces_between_special_tokens=False)
            except Exception:
                delta = ""
            if delta:
                text_token_ids.append(token_id)
                await ws.send_text(json.dumps({"type": "text_delta", "delta": delta}))

        # 6) 收尾，把完整文本回给前端（可选）
        try:
            complete_text = glm_tokenizer.decode(
                complete_token_ids, spaces_between_special_tokens=False
            )
        except Exception:
            complete_text = ""

        await ws.send_text(
            json.dumps(
                {
                    "type": "done",
                    "input_tokens": inputs,
                    "completion_tokens": complete_text,
                }
            )
        )
        await ws.close()

    except WebSocketDisconnect:
        return
    except Exception as e:
        # 最后兜底
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
