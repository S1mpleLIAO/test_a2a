# -*- coding: utf-8 -*-
import os
import json
import uuid
import torch
import torchaudio
import requests

from transformers import AutoTokenizer, WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token

from flow_inference import AudioDecoder
from audio_process import AudioStreamProcessor

# -------- 全局对象（单进程内共享） ----------
_objs = {
    "audio_decoder": None,
    "feature_extractor": None,
    "whisper_model": None,
    "glm_tokenizer": None,
    "device": "cuda",
    "config": {},
}


def _ensure_dir(path: str, name: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def initialize(
    flow_path: str = "/root/model/glm4-voice/glm-4-voice-decoder",
    model_path: str = "/root/model/glm4-voice/glm-4-voice-9b",
    tokenizer_path: str = "/root/model/glm4-voice/glm-4-voice-tokenizer",
):
    """
    初始化：文本 tokenizer、语音分词器(WhisperVQ + FeatureExtractor)、Flow/Hifi 解码器
    注意 tokenizer_path 要指向 “glm-4-voice-tokenizer” 目录，
    该目录里需包含 preprocessor_config.json / tokenizer_config.json 等文件。
    """
    if _objs["audio_decoder"] is not None:
        return

    _ensure_dir(flow_path, "flow_path")
    _ensure_dir(model_path, "model_path")
    _ensure_dir(tokenizer_path, "tokenizer_path")

    # 文本 tokenizer —— 优先走 tokenizer_path
    _objs["glm_tokenizer"] = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )

    # 语音分词器（Whisper VQ + 特征）
    preproc = os.path.join(tokenizer_path, "preprocessor_config.json")
    if not os.path.exists(preproc):
        raise FileNotFoundError(
            f"Missing preprocessor_config.json in {tokenizer_path}. "
            f"Please point tokenizer_path to glm-4-voice-tokenizer (NOT glm-4-voice-9b)."
        )

    _objs["whisper_model"] = (
        WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(_objs["device"])
    )

    _objs["feature_extractor"] = WhisperFeatureExtractor.from_pretrained(
        tokenizer_path, local_files_only=True
    )

    # Flow&Hifi 解码器
    flow_config = os.path.join(flow_path, "config.yaml")
    flow_ckpt = os.path.join(flow_path, "flow.pt")
    hift_ckpt = os.path.join(flow_path, "hift.pt")
    _objs["audio_decoder"] = AudioDecoder(
        config_path=flow_config,
        flow_ckpt_path=flow_ckpt,
        hift_ckpt_path=hift_ckpt,
        device=_objs["device"],
    )

    _objs["config"] = {
        "flow_path": flow_path,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
    }


def _build_prompt(system_prompt: str, user_input: str, previous: str = "") -> str:
    inputs = (previous or "").strip()
    if "<|system|>" not in inputs:
        inputs += f"<|system|>\n{system_prompt}"
    inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
    return inputs


def _sys_prompt_text() -> str:
    return (
        "User will provide you with a text instruction. "
        "Do it step by step. First, think about the instruction and respond in an interleaved manner, "
        "with 13 text token followed by 26 audio tokens."
    )


def _sys_prompt_audio() -> str:
    return (
        "User will provide you with a speech instruction. "
        "Do it step by step. First, think about the instruction and respond in an interleaved manner, "
        "with 13 text token followed by 26 audio tokens."
    )


# ----------------- 一次性推理（文本正常/音频不流式） -----------------
def infer_once(
    *,
    input_mode: str,
    audio_path: str | None,
    input_text: str | None,
    temperature: float = 0.2,
    top_p: float = 0.8,
    max_new_tokens: int = 2000,
    previous_input_tokens: str = "",
    previous_completion_tokens: str = "",
    sampler_url: str = "http://localhost:10000/generate_stream",
    media_dir: str = "./media",
) -> dict:
    assert (
        _objs["audio_decoder"] is not None
    ), "audio_decoder is None. initialize() not done?"
    assert (
        _objs["glm_tokenizer"] is not None
    ), "glm_tokenizer is None. initialize() not done?"

    glm_tokenizer = _objs["glm_tokenizer"]
    device = _objs["device"]
    audio_decoder: AudioDecoder = _objs["audio_decoder"]

    # 组装输入
    if input_mode == "audio":
        assert (
            _objs["whisper_model"] is not None
        ), "whisper_model is None. initialize() not done?"
        assert _objs["feature_extractor"] is not None, (
            "feature_extractor is None. "
            "Check tokenizer_path points to glm-4-voice-tokenizer and contains preprocessor_config.json."
        )
        assert audio_path, "audio mode requires audio_path"
        audio_tokens = extract_speech_token(
            _objs["whisper_model"], _objs["feature_extractor"], [audio_path]
        )[0]
        if len(audio_tokens) == 0:
            raise RuntimeError("No audio tokens extracted")
        audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        user_input = "<|begin_of_audio|>" + audio_tokens_str + "<|end_of_audio|>"
        system_prompt = _sys_prompt_audio()
    else:
        assert input_text, "text mode requires input_text"
        user_input = input_text
        system_prompt = _sys_prompt_text()

    inputs = _build_prompt(
        system_prompt, user_input, previous_input_tokens + previous_completion_tokens
    )

    with torch.no_grad():
        resp = requests.post(
            sampler_url,
            data=json.dumps(
                {
                    "prompt": inputs,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_new_tokens": int(max_new_tokens),
                }
            ),
            stream=True,
            timeout=600,
        )

        text_tokens, audio_tokens = [], []
        audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        end_token_id = glm_tokenizer.convert_tokens_to_ids("<|user|>")
        complete_tokens = []

        prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
        this_uuid = str(uuid.uuid4())
        tts_speechs, tts_mels = [], []
        prev_mel = None
        is_finalize = False

        block_size_list = [25, 50, 100, 150, 200]
        block_size_idx = 0
        block_size = block_size_list[block_size_idx]

        for line in resp.iter_lines():
            if not line:
                continue
            token_id = json.loads(line)["token_id"]
            if token_id == end_token_id:
                is_finalize = True

            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                if block_size_idx < len(block_size_list) - 1:
                    block_size_idx += 1
                    block_size = block_size_list[block_size_idx]

                tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
                if prev_mel is not None and len(tts_mels) > 0:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                tts_speech, tts_mel = audio_decoder.token2wav(
                    tts_token,
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(device),
                    prompt_feat=prompt_speech_feat.to(device),
                    finalize=is_finalize,
                )
                prev_mel = tts_mel
                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)
                flow_prompt_speech_token = torch.cat(
                    (flow_prompt_speech_token, tts_token), dim=-1
                )
                audio_tokens = []

            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

        if not tts_speechs:
            raise RuntimeError("No audio was produced.")

        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        complete_text = glm_tokenizer.decode(
            complete_tokens, spaces_between_special_tokens=False
        )
        completion_raw_text = glm_tokenizer.decode(
            text_tokens, ignore_special_tokens=False
        )

        os.makedirs(media_dir, exist_ok=True)
        out_path = os.path.join(media_dir, f"{uuid.uuid4().hex}.wav")
        torchaudio.save(out_path, tts_speech.unsqueeze(0), 22050, format="wav")
        # final_text = _objs["glm_tokenizer"].decode(text_tokens, skip_special_tokens=True, spaces_between_special_tokens=False)
        final_text = _objs["glm_tokenizer"].decode(
            text_tokens, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        # 2) 完整 token 序列（含 <audio_*>），用于 Debug 区展示
        all_tokens_text = _objs["glm_tokenizer"].decode(
            complete_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        return {
            "text": final_text,
            "wav_path": os.path.abspath(out_path),
            "sample_rate": 22050,
            "input_tokens": inputs,
            "completion_tokens": all_tokens_text,
        }


# ----------------- 流式推理（WebSocket 用） -----------------
def infer_stream(
    *,
    input_mode: str,
    audio_path: str | None,
    input_text: str | None,
    temperature: float = 0.2,
    top_p: float = 0.8,
    max_new_tokens: int = 2000,
    previous_input_tokens: str = "",
    previous_completion_tokens: str = "",
    sampler_url: str = "http://localhost:10000/generate_stream",
    media_dir: str = "./media",
):
    import base64

    assert (
        _objs["audio_decoder"] is not None
    ), "audio_decoder is None. initialize() not done?"
    assert (
        _objs["glm_tokenizer"] is not None
    ), "glm_tokenizer is None. initialize() not done?"

    glm_tokenizer = _objs["glm_tokenizer"]
    device = _objs["device"]
    audio_decoder: AudioDecoder = _objs["audio_decoder"]

    # 组装输入
    if input_mode == "audio":
        assert (
            _objs["whisper_model"] is not None
        ), "whisper_model is None. initialize() not done?"
        assert _objs["feature_extractor"] is not None, (
            "feature_extractor is None. "
            "Check tokenizer_path points to glm-4-voice-tokenizer and contains preprocessor_config.json."
        )
        assert audio_path, "audio mode requires audio_path"
        audio_tokens0 = extract_speech_token(
            _objs["whisper_model"], _objs["feature_extractor"], [audio_path]
        )[0]
        if len(audio_tokens0) == 0:
            raise RuntimeError("No audio tokens extracted")
        audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens0])
        user_input = "<|begin_of_audio|>" + audio_tokens_str + "<|end_of_audio|>"
        system_prompt = _sys_prompt_audio()
    else:
        assert input_text, "text mode requires input_text"
        user_input = input_text
        system_prompt = _sys_prompt_text()

    inputs = _build_prompt(
        system_prompt, user_input, previous_input_tokens + previous_completion_tokens
    )

    with torch.no_grad():
        resp = requests.post(
            sampler_url,
            data=json.dumps(
                {
                    "prompt": inputs,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_new_tokens": int(max_new_tokens),
                }
            ),
            stream=True,
            timeout=600,
        )

        text_tokens, audio_tokens = [], []
        audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        end_token_id = glm_tokenizer.convert_tokens_to_ids("<|user|>")
        complete_tokens = []

        prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
        this_uuid = str(uuid.uuid4())
        tts_speechs, tts_mels = [], []
        prev_mel = None
        is_finalize = False

        block_size_list = [25, 50, 100, 150, 200]
        block_size_idx = 0
        block_size = block_size_list[block_size_idx]

        audio_processor = AudioStreamProcessor()

        for line in resp.iter_lines():
            if not line:
                continue
            token_id = json.loads(line)["token_id"]
            if token_id == end_token_id:
                is_finalize = True

            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                if block_size_idx < len(block_size_list) - 1:
                    block_size_idx += 1
                    block_size = block_size_list[block_size_idx]

                tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
                if prev_mel is not None and len(tts_mels) > 0:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                tts_speech, tts_mel = audio_decoder.token2wav(
                    tts_token,
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(device),
                    prompt_feat=prompt_speech_feat.to(device),
                    finalize=is_finalize,
                )
                prev_mel = tts_mel
                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)

                audio_bytes = audio_processor.process(
                    tts_speech.clone().cpu().numpy()[0], last=is_finalize
                )
                if audio_bytes:
                    yield {
                        "kind": "chunk",
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                    }

                flow_prompt_speech_token = torch.cat(
                    (flow_prompt_speech_token, tts_token), dim=-1
                )
                audio_tokens = []

            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

        if not tts_speechs:
            raise RuntimeError("No audio was produced.")

        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        complete_text = glm_tokenizer.decode(
            complete_tokens, spaces_between_special_tokens=False
        )
        completion_raw_text = glm_tokenizer.decode(
            text_tokens, ignore_special_tokens=False
        )

        os.makedirs(media_dir, exist_ok=True)
        out_path = os.path.join(media_dir, f"{uuid.uuid4().hex}.wav")
        torchaudio.save(out_path, tts_speech.unsqueeze(0), 22050, format="wav")
        wav_url = f"/media/{os.path.basename(out_path)}"
        final_text = _objs["glm_tokenizer"].decode(
            text_tokens, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        # 2) 完整 token 序列（含 <audio_*>），用于 Debug 区展示
        all_tokens_text = _objs["glm_tokenizer"].decode(
            complete_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        yield {
            "kind": "final",
            "text": final_text,  # ← 只含文字
            "wav_url": wav_url,
            "sample_rate": 22050,
            "input_tokens": inputs,
            "completion_tokens": all_tokens_text,  # ← Debug 用，含 <audio_*>
        }
