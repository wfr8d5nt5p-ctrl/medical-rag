# -*- coding: utf-8 -*-
"""
医知助手 · 医疗知识 Agent 网页后端（纯 Python 标准库，零第三方 Web 依赖）

前端:   index.html（同目录，浏览器打开）
运行:   cd medical-rag/web  &&  python server.py
先确保: 01 已建库、.env 已配置 DEEPSEEK_API_KEY
访问:   浏览器打开 http://127.0.0.1:8000

接口:
  GET  /            →  返回首页 index.html
  POST /api/chat    →  body  {"messages":[{"role":"user","content":"..."}]}
                        返回 {answer, sources:[{text,score}]}
  POST /api/ocr     →  body  {"image":"data:image/png;base64,...."}
                        返回 {text}
说明: DeepSeek Key 只在本后端 .env，不经过浏览器，安全。
"""
import os
import sys
import json
import base64
import mimetypes
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # medical-rag 根目录
DB_DIR = os.path.join(BASE, "chroma_db")
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("AGENT_PORT", "8000"))

sys.path.insert(0, BASE)  # 让 web/ 下的代码能 import 根目录模块
from agent_tools import build_tools, get_retriever, reset_flag, was_retrieved, MIN_SOURCE_SCORE

load_dotenv(os.path.join(BASE, ".env"))

_AGENT = None


def _build_agent(api_key=None):
    """ReAct Agent：多工具（知识检索/紧急分级/剂量计算/单位换算）+ DeepSeek + 提示词。
    api_key 不传时优先用环境变量 DEEPSEEK_API_KEY；传了则用该用户的 key。"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.prebuilt import create_react_agent

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是专业的健康顾问 Agent。你有以下工具：\n"
                "1) query_medical_kb：涉及疾病、症状、饮食、用药等医疗健康事实时调用，检索知识库获取准确资料；\n"
                "2) emergency_triage：用户描述胸痛、呼吸困难、昏迷等急症症状时调用，做紧急程度分级；\n"
                "3) calc_dosage：需要按体重计算用药剂量时调用；\n"
                "4) convert_units：需要进行 mg/g/kg、ml/L 等医疗单位换算时调用。\n"
                "寒暄、自我介绍等非医疗问题无需调用工具，可直接回答。"
                "资料中没有的内容要明确说明，不要凭空编造。回答请结构化分段，末尾给出免责声明。",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    return create_react_agent(llm, build_tools(), prompt=prompt)


_AGENTS = {}  # api_key -> agent 缓存，避免每个请求都重建 agent


def _get_agent(api_key=None):
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or ""
    if not key.strip():
        raise ValueError("未配置 API Key，请在网页右上角输入你的 DeepSeek API Key 后再提问。")
    if key not in _AGENTS:
        _AGENTS[key] = _build_agent(api_key)
    return _AGENTS[key]


def _sources_for(question: str):
    """对最近一个问题做一次检索，返回来源片段 + 相关度打分（供前端展示置信度）。复用全局检索器，不再重复加载模型。
    用 MIN_SOURCE_SCORE 过滤掉低相关来源，避免把无关片段(如其他患者的化验单)展示给用户。"""
    try:
        out = []
        for text, score in get_retriever().search_with_scores(question, k=3, min_score=MIN_SOURCE_SCORE):
            out.append({"text": text.strip()[:220], "score": score})
        return out
    except Exception:
        return []


def _run_chat(messages, api_key=None):
    """调用 Agent 出回答。messages 为 [{'role','content'}, ...]（后端会自动补 system）。
    返回 (answer, retrieved) —— retrieved 表示本次 Agent 是否真正调用了检索工具。
    api_key 由前端传入；不传则用环境变量。"""
    reset_flag()
    agent = _get_agent(api_key)
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content, was_retrieved()


def _ocr_image(data_url: str) -> str:
    from rapidocr_onnxruntime import RapidOCR

    # data_url:  data:image/png;base64,xxxx
    if data_url.startswith("data:") and "," in data_url:
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url
    raw = base64.b64decode(b64)

    # 用 Pillow 从字节解码为 RGB 数组，可自动识别 png/jpeg/webp 等格式，
    # 从而绕开"临时文件后缀猜错导致 PIL 无法识别"的问题。
    from PIL import Image
    import io
    import numpy as np
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)  # HxWx3，RapidOCR 直接接受 ndarray

    engine = RapidOCR()
    out = engine(arr)
    result = out[0] if isinstance(out, tuple) else out
    lines = []
    if result:
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = item[1]
                if isinstance(txt, (list, tuple)):
                    txt = txt[0]
                lines.append(str(txt))
            else:
                lines.append(str(item))
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 关闭默认访问日志，保持终端干净

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        else:
            # 兜底：尝试在同目录找静态文件
            candidate = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))
            if candidate.startswith(WEB_DIR) and os.path.isfile(candidate):
                ctype = mimetypes.guess_type(candidate)[0] or "application/octet-stream"
                self._serve_file(candidate, ctype + "; charset=utf-8" if ctype.startswith("text/") else ctype)
            else:
                self._send(404, b"Not Found", "text/plain")

    def _serve_file(self, path, ctype):
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

            if path == "/api/chat":
                messages = body.get("messages", [])
                api_key = body.get("api_key") or None
                if not messages:
                    return self._send_json({"error": "messages 不能为空"}, 400)
                last_user = next(
                    (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
                )
                try:
                    answer, retrieved = _run_chat(messages, api_key)
                except ValueError as e:
                    # 缺 API Key：提示用户而非当做系统错误，前端可据此突出展示
                    return self._send_json({"error": str(e), "need_key": True}, 400)
                except Exception as e:
                    traceback.print_exc()
                    return self._send_json({"error": f"Agent 调用失败：{e}"}, 500)
                # 仅当 Agent 真正调用了检索工具时，才附带知识来源；否则前端不显示来源/置信度
                sources = (_sources_for(last_user) if (last_user and retrieved) else [])
                return self._send_json({"answer": answer, "sources": sources})

            if path == "/api/ocr":
                image = body.get("image", "")
                if not image:
                    return self._send_json({"error": "缺少 image"}, 400)
                try:
                    text = _ocr_image(image)
                except Exception as e:
                    traceback.print_exc()
                    return self._send_json({"error": f"OCR 失败:{e}"}, 500)
                return self._send_json({"text": text})

            return self._send_json({"error": f"未知接口 {path}"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._send_json({"error": f"请求处理失败：{e}"}, 500)


if __name__ == "__main__":
    print(f"医知助手后端已启动：http://127.0.0.1:{PORT}  （Ctrl+C 停止）")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()