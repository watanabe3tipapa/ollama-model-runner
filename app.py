"""
GradioベースのMVP UIで、Ollamaモデル（ローカル）を実行します。
- ローカルでOllamaが起動している必要があります（デフォルト: http://localhost:11434）
- デフォルトモデル: llama3.2
"""

import json
import os
import time
from typing import Dict, List

import gradio as gr
import httpx

DEFAULT_MODEL = "qwen3.5"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ollama_models() -> List[str]:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return [DEFAULT_MODEL]


def refresh_models():
    return gr.Dropdown(choices=get_ollama_models())


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_ollama_models = get_ollama_models()
SESSION_HISTORY: Dict[str, List[Dict]] = {}
_active_requests: Dict[str, Dict] = {}


# ---- Helpers ----
def make_session_id() -> str:
    return str(int(time.time() * 1000))


def add_history(user_id: str, item: dict):
    SESSION_HISTORY.setdefault(user_id, []).insert(0, item)
    SESSION_HISTORY[user_id] = SESSION_HISTORY[user_id][:50]


def get_history(user_id: str):
    return SESSION_HISTORY.get(user_id, [])


def call_ollama_model(model_id: str, prompt: str, params: dict) -> tuple[bool, str]:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "num_predict": params.get("max_new_tokens", 128),
        },
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
    except httpx.RequestError as e:
        return False, f"[NETWORK_ERROR] {e}"
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        return False, f"[API_ERROR {resp.status_code}] {err}"
    try:
        data = resp.json()
        if "response" in data:
            return True, data["response"]
        return True, json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return False, f"[PARSE_ERROR] {e}"


def stream_chunks(
    text: str, session_id: str, chunk_size: int = 80, delay: float = 0.03
):
    buf = ""
    for i in range(0, len(text), chunk_size):
        if session_id in _active_requests and _active_requests[session_id].get(
            "cancelled"
        ):
            yield "[CANCELLED]"
            return
        chunk = text[i : i + chunk_size]
        buf += chunk
        yield buf
        time.sleep(delay)
    yield "[DONE]"


# ---- Gradio callbacks ----
SAMPLES = [
    {"title": "フリートーク", "prompt": ""},
    {"title": "要約", "prompt": "以下を200文字以内で日本語で要約してください：\n\n"},
    {"title": "質問応答", "prompt": "ユーザー: {question}\nアシスタント:"},
    {"title": "メール下書き", "prompt": "宛先: {name}\n件名: {subject}\n本文:\n"},
    {"title": "メッセージ下書き", "prompt": "送信先: {recipient}\n內容:\n"},
]


def on_sample_change(choice):
    for s in SAMPLES:
        if s["title"] == choice:
            return s["prompt"]
    return ""


def start_generation(
    model_id,
    prompt,
    temperature,
    top_p,
    max_new_tokens,
    save_history,
    user_id,
):
    if not model_id:
        yield "", "モデルを選択してください", "{}", make_session_id()
        return
    session_id = make_session_id()
    _active_requests[session_id] = {"cancelled": False}
    params = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
    }
    ok, result = call_ollama_model(model_id, prompt, params)
    meta = json.dumps(
        {
            "model": model_id,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        ensure_ascii=False,
    )
    if not ok:
        _active_requests.pop(session_id, None)
        yield "", f"{result}", meta, session_id  # error -> show in output box
        return
    # Stream simulated
    for partial in stream_chunks(result, session_id):
        if partial == "[CANCELLED]":
            yield partial, "", meta, session_id
            _active_requests.pop(session_id, None)
            return
        if partial == "[DONE]":
            # finalization: save history if requested
            if save_history:
                add_history(
                    user_id,
                    {
                        "model": model_id,
                        "prompt": prompt,
                        "output": result,
                        "meta": meta,
                        "time": time.time(),
                    },
                )
            _active_requests.pop(session_id, None)
            yield result, "", meta, session_id
            return
        yield partial, "", meta, session_id


def do_cancel(session_id):
    if session_id in _active_requests:
        _active_requests[session_id]["cancelled"] = True
        return "キャンセル要求を送信しました"
    return "実行中のリクエストはありません"


def do_clear():
    return "", "", "", make_session_id()


def do_download(text):
    if not text:
        return None
    b = text.encode("utf-8")
    return ("output.txt", b)


def refresh_history(user_id):
    items = get_history(user_id)
    if not items:
        return "履歴はありません"
    lines = []
    for it in items:
        lines.append(
            f"### {it['model']}\n**Prompt:**\n```\n{it['prompt']}\n```\n**Output (先頭):**\n```\n{it['output'][:500]}\n```\n"
        )
    return "\n\n".join(lines)


# ---- Gradio UI ----
with gr.Blocks(title="Ollama Model Runner") as demo:
    gr.Markdown("## Ollama Model Runner (Local)")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=_ollama_models,
                    value=_ollama_models[0] if _ollama_models else DEFAULT_MODEL,
                )
                refresh_btn = gr.Button("🔄", size="sm")
            sample_dropdown = gr.Dropdown(
                [s["title"] for s in SAMPLES],
                label="サンプル",
                value=SAMPLES[0]["title"],
            )
            prompt_box = gr.Textbox(label="Prompt", lines=8, value=SAMPLES[0]["prompt"])
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, label="temperature"
                )
                top_p_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, label="top_p"
                )
            max_tokens = gr.Number(label="max_new_tokens", value=128, precision=0)
            with gr.Row():
                clear_btn = gr.Button("クリア")
                cancel_btn = gr.Button("キャンセル", variant="stop")
                run_btn = gr.Button("実行")
            save_switch = gr.Checkbox(label="履歴に保存", value=True)
            gr.Markdown("### サンプル一覧")
            gr.Markdown(
                "\n".join([f"- **{s['title']}**: {s['prompt']}" for s in SAMPLES])
            )

        with gr.Column(scale=3):
            stream_output = gr.Textbox(label="出力（ストリーミング表示）", lines=16)
            output_area = gr.HTML(label="完全出力")
            meta_info = gr.Textbox(label="メタ情報", lines=3)
            with gr.Row():
                download_btn = gr.Button("ダウンロード")
                history_btn = gr.Button("履歴を更新")

    # hidden state
    session_id_state = gr.Textbox(value=make_session_id(), visible=False)
    user_id_state = gr.Textbox(value="default_user", visible=False)
    history_box = gr.Markdown()

    # events
    sample_dropdown.change(
        fn=on_sample_change, inputs=sample_dropdown, outputs=prompt_box
    )
    run_btn.click(
        fn=start_generation,
        inputs=[
            model_dropdown,
            prompt_box,
            temp_slider,
            top_p_slider,
            max_tokens,
            save_switch,
            user_id_state,
        ],
        outputs=[stream_output, output_area, meta_info, session_id_state],
    )
    refresh_btn.click(fn=refresh_models, outputs=[model_dropdown])
    cancel_btn.click(fn=do_cancel, inputs=[session_id_state], outputs=[meta_info])
    clear_btn.click(
        fn=do_clear, outputs=[prompt_box, stream_output, output_area, session_id_state]
    )
    download_btn.click(
        fn=do_download, inputs=[output_area], outputs=[gr.File(label="ダウンロード")]
    )
    history_btn.click(fn=refresh_history, inputs=[user_id_state], outputs=[history_box])

demo.launch()
