# -*- coding: utf-8 -*-
# 兼容“模块方式运行”和“直接运行脚本”的导入兜底
import os, sys, json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

try:
    # 模块方式（python -m app.app / uvicorn app.app:app）
    from app.inference_core import initialize, infer_once, infer_stream, _objs
except Exception:
    # 直接执行（python app/app.py）
    from inference_core import initialize, infer_once, infer_stream, _objs

from fastapi import FastAPI, UploadFile, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

APP_DIR = CURRENT_DIR
ROOT_DIR = PROJECT_ROOT
STATIC_DIR = os.path.join(ROOT_DIR, "static")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
MEDIA_DIR = os.path.join(APP_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# -------- FastAPI ----------
app = FastAPI(title="GLM-4-Voice (FastAPI + HTML)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# === 启动时初始化（注意路径：glm4-voice 没有中划线） ===
@app.on_event("startup")
def _startup():
    initialize(
        flow_path="/root/model/glm4-voice/glm-4-voice-decoder",
        model_path="/root/model/glm4-voice/glm-4-voice-9b",
        tokenizer_path="/root/model/glm4-voice/glm-4-voice-tokenizer",
    )

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---- 文本一次性接口（也可给音频文件路径，直接复用 infer_once） ----
@app.post("/api/infer")
async def api_infer(
    input_mode: str = Form(...),            # "audio" or "text"
    temperature: float = Form(0.2),
    top_p: float = Form(0.8),
    max_new_tokens: int = Form(2000),
    input_text: str | None = Form(None),
    previous_input_tokens: str | None = Form(""),
    previous_completion_tokens: str | None = Form(""),
    audio: UploadFile | None = None,
):
    try:
        audio_path = None
        if input_mode == "audio" and audio is not None:
            up_dir = os.path.join(MEDIA_DIR, "uploads"); os.makedirs(up_dir, exist_ok=True)
            dst = os.path.join(up_dir, audio.filename or "input.wav")
            with open(dst, "wb") as f:
                f.write(await audio.read())
            audio_path = dst

        result = infer_once(
            input_mode=input_mode,
            audio_path=audio_path,
            input_text=input_text,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            previous_input_tokens=previous_input_tokens or "",
            previous_completion_tokens=previous_completion_tokens or "",
            sampler_url="http://localhost:10000/generate_stream",
            media_dir=MEDIA_DIR,
        )

        wav_abs = result.get("wav_path")
        wav_url = f"/media/{os.path.basename(wav_abs)}" if wav_abs else None
        return {
            "ok": True,
            "text": result["text"],
            "audio_url": wav_url,
            "sample_rate": result["sample_rate"],
            "input_tokens": result["input_tokens"],
            "completion_tokens": result["completion_tokens"],
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---- 上传录音文件（前端“上传并对话”先调它） ----
@app.post("/api/upload_audio")
async def upload_audio(audio: UploadFile):
    try:
        up_dir = os.path.join(MEDIA_DIR, "uploads"); os.makedirs(up_dir, exist_ok=True)
        dst = os.path.join(up_dir, audio.filename or "record.webm")
        with open(dst, "wb") as f:
            f.write(await audio.read())
        return {"ok": True, "audio_path": dst}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- WebSocket：实时返回音频分块 ----
@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    try:
        # 兜底：若被热重启/异常影响导致对象丢失，重跑初始化
        if _objs.get("audio_decoder") is None or _objs.get("feature_extractor") is None:
            initialize(
                flow_path="/root/model/glm4-voice/glm-4-voice-decoder",
                model_path="/root/model/glm4-voice/glm-4-voice-9b",
                tokenizer_path="/root/model/glm4-voice/glm-4-voice-tokenizer",
            )

        params = await ws.receive_json()
        gen = infer_stream(
            input_mode=params.get("input_mode", "audio"),
            audio_path=params.get("audio_path"),
            input_text=params.get("input_text"),
            temperature=float(params.get("temperature", 0.2)),
            top_p=float(params.get("top_p", 0.8)),
            max_new_tokens=int(params.get("max_new_tokens", 2000)),
            previous_input_tokens=params.get("previous_input_tokens", ""),
            previous_completion_tokens=params.get("previous_completion_tokens", ""),
            sampler_url="http://localhost:10000/generate_stream",
            media_dir=MEDIA_DIR,
        )
        for msg in gen:
            await ws.send_text(json.dumps(msg))
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(json.dumps({"kind": "error", "error": str(e)}))
        await ws.close()

# 允许 python 直接运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=False, log_level="info")
