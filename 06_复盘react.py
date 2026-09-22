# -*- coding: utf-8 -*-
"""
06 复盘：这个 Agent 到底是不是"真 ReAct"？
运行：python -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple langchain langgraph  (必要时)
     set HF_ENDPOINT=https://hf-mirror.com
     python 06_复盘react.py      (medical-rag 目录下)

它会对一组测试问题逐一运行 Agent，并把每一次的"轨迹"打印出来：
    每轮循环里，AI 是"调用了工具 query_medical_kb" 还是 "直接给出最终答案"。

重点结论：
  - 若所有问题都调用工具  => 说明提示词"务必先查库"把它变成了固定流程（伪 ReAct）
  - 若某些问题不调用工具  => 说明它在做真正的决策（真 ReAct）

判断标准：问不需要查库的问题（寒暄/自我介绍），真 ReAct 不会查。
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


def build_agent(llm):
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
    return create_react_agent(llm, build_tools(), prompt=prompt)


def replay(agent, question: str):
    """运行一次，并打印轨迹：判断 AI 是否调用了工具、调了几次。"""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    msgs = result["messages"]

    print("\n" + "=" * 56)
    print(f"问题：{question}")
    print("-" * 56)
    tool_calls = 0
    for m in msgs:
        role = m.__class__.__name__
        if role == "AIMessage" and getattr(m, "tool_calls", None):
            tc = m.tool_calls[0]["name"]
            tool_calls += 1
            print(f"  [AI] 调用了工具 -> {tc}")
        elif role == "ToolMessage":
            short = (m.content or "")[:80].replace("\n", " ")
            print(f"  [工具返回] {short}")
        elif role == "AIMessage":
            # 这是最终回答（也可能是"还需要继续思考"）
            print(f"  [AI 输出] {m.content[:100]}")
    verdict = "✅ 本轮调用了工具（医疗类）" if tool_calls else "✅ 本轮未调用工具（自主决策）"
    print(f"  调用工具次数：{tool_calls}  →  {verdict}")
    return tool_calls


def main():
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )
    agent = build_agent(llm)

    tests = [
        "你好，你是谁？",                              # 寒暄，不应调任何工具
        "我胸口突然很痛，喘不上气怎么办？",             # 急症 → emergency_triage
        "高血压饮食要注意什么？",                       # 医疗 → query_medical_kb
        "我体重70公斤，医生让按每公斤5mg吃，一天两次，每次该吃多少？",  # 剂量 → calc_dosage
        "250mg等于多少克？",                            # 换算 → convert_units
    ]

    print("开始复盘（共 %d 个问题）..." % len(tests))
    for q in tests:
        replay(agent, q)

    print("\n" + "=" * 56)
    print("总结：观察每个问题是否调用了"正确的"工具、还是完全不调用。")
    print("  - 急症问题 → 应调 emergency_triage")
    print("  - 医疗事实 → 应调 query_medical_kb")
    print("  - 剂量/换算 → 应调 calc_dosage / convert_units")
    print("  - 寒暄 → 不应调用任何工具")


if __name__ == "__main__":
    main()