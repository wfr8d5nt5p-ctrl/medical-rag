"""
04 医疗 Agent · ReAct 工作流（LangGraph 正规版）

运行：python 04_react_agent.py   （先确保 01 已建库）

本文件演示 Agent 与 "RAG"(02) 的本质区别：
  02 是固定流程：问题 → 永远检索 → 回答
  04 是智能决策：LLM 自己判断"这个问题要不要查知识库"，查完不够还能再查。

LangChain 提供"零件"：LLM、提示词、检索器工具
LangGraph 负责"调度"：让 ReAct 循环（想 → 调工具 → 看结果 → 再想）能真正跑起来
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
sys.path.insert(0, BASE_DIR)
from agent_tools import build_tools

# ---- 记忆窗口配置 ----
RECENT_TURNS = 3       # 始终保留最近几轮的"原文"
TRIGGER_TURNS = 6      # 历史超过多少轮后，把最早的压缩成摘要


def compress_history(llm, messages) -> str:
    """把较早的历史压缩成一句话摘要：丢掉废话、保留大意。"""
    text = "\n".join(f"{role}: {content}" for role, content in messages)
    summary = llm.invoke(
        "请用简洁的中文，把下面这段多轮对话压缩成一句摘要，"
        "只保留关键病情、问题和结论，丢掉寒暄与冗余：\n\n" + text
    )
    return summary.content.strip()


def main():
    # ---- 2. LangChain 提供"大脑"：DeepSeek 大模型 ----
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    # ---- 3. ReAct 提示词：给 Agent 立行为准则（多工具）----
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
                "资料中没有的内容要明确说明，不要凭空编造。",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # ---- 4. LangGraph 把"大脑 + 多工具 + 提示词"编排成能循环的 Agent 图 ----
    agent = create_react_agent(llm, build_tools(), prompt=prompt)

    # ---- 5. 交互式对话：累积历史 + 多轮记忆 + 摘要压缩 + 截断保底 ----
    print("=== 医疗 Agent（ReAct·多轮记忆版）已启动；输入 exit 退出 ===\n")
    memory = []  # 累积历史，元素为 ("user"/"ai", 文本)
    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("exit", "quit", "退出"):
            print("已退出。")
            break

        memory.append(("user", user_input))

        # 记忆压缩：历史超过阈值时，把最早的压成摘要，仅保留最近几轮原文
        if len(memory) > TRIGGER_TURNS * 2:
            old, recent = memory[: len(memory) - RECENT_TURNS * 2], memory[-RECENT_TURNS * 2:]
            summary = compress_history(llm, old)
            memory = [("system", f"[更早对话摘要] {summary}")] + recent
            print(f"\n[记忆压缩] 已把较早 {len(old)} 条对话压缩为摘要，保留最近 {RECENT_TURNS} 轮。\n")

        result = agent.invoke({"messages": memory})
        answer = result["messages"][-1].content
        memory.append(("ai", answer))
        print("\nAgent：", answer, "\n")


if __name__ == "__main__":
    main()