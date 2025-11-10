// --------- 工具与全局状态 ----------
const qs = (sel) => document.querySelector(sel);
const state = {
  previous_input_tokens: "",
  previous_completion_tokens: "",
  recordedBlob: null,
  mediaRecorder: null,
  chunks: []
};

function appendBubble(role, content) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.innerHTML = content;
  qs("#chat").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}
function showError(err) { qs("#detailed_error").value = String(err || ""); }
function clearAll() {
  qs("#chat").innerHTML = "";
  qs("#input_tokens").value = "";
  qs("#completion_tokens").value = "";
  qs("#detailed_error").value = "";
  qs("#stream_player").src = "";
  qs("#final_player").src = "";
  state.previous_input_tokens = "";
  state.previous_completion_tokens = "";
}
function getMode(){ return document.querySelector('input[name="input_mode"]:checked').value; }
function setMode(mode){
  if(mode==="audio"){ qs("#audio_box").classList.remove("hidden"); qs("#text_box").classList.add("hidden"); }
  else { qs("#audio_box").classList.add("hidden"); qs("#text_box").classList.remove("hidden"); }
}

// --------- 文本一次性提交 ----------
async function submitText() {
  try {
    const text = qs("#input_text").value.trim();
    if(!text){ alert("请输入文本"); return; }
    appendBubble("user", `<strong>User:</strong><br>${text.replace(/\n/g,"<br>")}`);

    const fd = new FormData();
    fd.append("input_mode","text");
    fd.append("input_text", text);
    fd.append("temperature", String(parseFloat(qs("#temperature").value || "0.2")));
    fd.append("top_p", String(parseFloat(qs("#top_p").value || "0.8")));
    fd.append("max_new_tokens", String(parseInt(qs("#max_new_tokens").value || "2000",10)));
    fd.append("previous_input_tokens", state.previous_input_tokens || "");
    fd.append("previous_completion_tokens", state.previous_completion_tokens || "");

    const res = await fetch("/api/infer",{method:"POST",body:fd});
    const data = await res.json();
    if(!data.ok){ showError(data.error||"Unknown error"); appendBubble("assistant", `<strong>Error:</strong> ${data.error||"Unknown error"}`); return; }

    qs("#input_tokens").value = data.input_tokens || "";
    qs("#completion_tokens").value = data.completion_tokens || "";
    state.previous_input_tokens = data.input_tokens || "";
    state.previous_completion_tokens = data.completion_tokens || "";

    if(data.audio_url){
      qs("#stream_player").src = data.audio_url;  // 立即播放
      qs("#final_player").src  = data.audio_url;
    }
    appendBubble("assistant", `<strong>Assistant:</strong><br>${(data.text||"").replace(/\n/g,"<br>")}`);
  } catch(err){ console.error(err); showError(err); appendBubble("assistant", `<strong>Error:</strong> ${String(err)}`); }
}

// --------- 录音：开始 / 停止 / 上传并对话 ----------
async function startRecording(){
  if(getMode()!=="audio") return;
  if(!navigator.mediaDevices){ alert("此浏览器不支持录音"); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio:true });

  state.chunks = [];
  state.mediaRecorder = new MediaRecorder(stream, { mimeType:"audio/webm;codecs=opus" });
  state.mediaRecorder.ondataavailable = ev => { if(ev.data.size>0) state.chunks.push(ev.data); };
  state.mediaRecorder.start(200);

  qs("#start_btn").disabled = true;
  qs("#stop_btn").disabled = false;
  qs("#upload_btn").disabled = true;
  state.recordedBlob = null;
  qs("#record_preview").src = "";
}

async function stopRecording(){
  if(!state.mediaRecorder || state.mediaRecorder.state!=="recording") return;
  const done = new Promise(res => state.mediaRecorder.onstop = res);
  state.mediaRecorder.stop();
  await done;

  // 组合 Blob，允许预听
  const blob = new Blob(state.chunks, { type:"audio/webm;codecs=opus" });
  state.recordedBlob = blob;
  qs("#record_preview").src = URL.createObjectURL(blob);

  qs("#start_btn").disabled = false;
  qs("#stop_btn").disabled = true;
  qs("#upload_btn").disabled = false;
}

async function uploadAndTalk(){
  try{
    if(!state.recordedBlob){ alert("请先录音并停止"); return; }
    appendBubble("user", `<strong>User (audio):</strong><br><small>${new Date().toLocaleTimeString()}</small>`);

    // 1) 上传录音，获得后端本地路径
    const fd = new FormData();
    fd.append("audio", state.recordedBlob, "record.webm");
    const up = await fetch("/api/upload_audio", { method:"POST", body:fd });
    const uj = await up.json();
    if(!uj.ok) throw new Error(uj.error||"upload failed");
    const audio_path = uj.audio_path;

    // 2) 建立 WS，实时接收分块并播放
    const ws = new WebSocket(`ws://${location.host}/ws/infer`);
    ws.onopen = () => {
      ws.send(JSON.stringify({
        input_mode: "audio",
        audio_path,
        temperature: parseFloat(qs("#temperature").value || "0.2"),
        top_p: parseFloat(qs("#top_p").value || "0.8"),
        max_new_tokens: parseInt(qs("#max_new_tokens").value || "2000",10),
        previous_input_tokens: state.previous_input_tokens || "",
        previous_completion_tokens: state.previous_completion_tokens || ""
      }));
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if(msg.kind === "chunk"){
        // 简化：每个分块当独立 WAV 片段播放（近实时）
        const bytes = Uint8Array.from(atob(msg.audio_b64), c => c.charCodeAt(0));
        const blob = new Blob([bytes], { type:"audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = qs("#stream_player");
        audio.src = url; audio.play().catch(()=>{});
      }else if(msg.kind === "final"){
        qs("#final_player").src = msg.wav_url || "";
        qs("#input_tokens").value = msg.input_tokens || "";
        qs("#completion_tokens").value = msg.completion_tokens || "";
        state.previous_input_tokens = msg.input_tokens || "";
        state.previous_completion_tokens = msg.completion_tokens || "";
        appendBubble("assistant", `<strong>Assistant:</strong><br>${(msg.text||"").replace(/\n/g,"<br>")}`);
      }else if(msg.kind === "error"){
        showError(msg.error);
        appendBubble("assistant", `<strong>Error:</strong> ${msg.error}`);
      }
    };
    ws.onerror = () => { showError("WebSocket error"); };
  }catch(err){ console.error(err); showError(err); appendBubble("assistant", `<strong>Error:</strong> ${String(err)}`); }
}

// --------- 初始化绑定 ----------
window.addEventListener("DOMContentLoaded", () => {
  // 模式切换
  document.querySelectorAll('input[name="input_mode"]').forEach(r => {
    r.addEventListener("change", e => setMode(e.target.value));
  });
  setMode(getMode());

  // 文本
  qs("#submit_text_btn").addEventListener("click", submitText);
  qs("#clear_btn").addEventListener("click", clearAll);

  // 音频：开始/停止/上传
  qs("#start_btn").addEventListener("click", startRecording);
  qs("#stop_btn").addEventListener("click", stopRecording);
  qs("#upload_btn").addEventListener("click", uploadAndTalk);
});
