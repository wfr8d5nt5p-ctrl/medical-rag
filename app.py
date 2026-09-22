"""
医疗 Agent 网页版（Streamlit）

运行：python -m streamlit run app.py   （先确保 01 已建库、.env 已配 Key）
说明：复用 04 的 ReAct Agent + 多轮记忆，换成网页聊天界面。
      Key 只存在于后端 .env，不经过浏览器，安全。
"""
import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

st.set_page_config(page_title="医疗健康 Agent", page_icon="🩺", layout="centered")


# ---- 缓存：只初始化一次模型/向量库/Agent，避免每轮重复加载 ----
@st.cache_resource
def build_agent():
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.prebuilt import create_react_agent

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(BASE_DIR, "chroma_db")

    @tool
    def query_medical_kb(question: str) -> str:
        """当用户询问疾病、症状、饮食、用药等医疗健康问题时，调用本工具检索医疗知识库，
        返回与问题最相关的资料原文。"""
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        docs = db.as_retriever(search_kwargs={"k": 3}).invoke(question)
        return "\n\n".join(d.page_content for d in docs) or "知识库中没有相关资料。"

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是专业的健康顾问 Agent。你可以使用工具查询医疗知识库以获得准确资料。"
                "涉及疾病、症状、饮食、用药等问题时，请务必先调用工具获取资料再回答，"
                "不要凭空编造；资料中没有的内容要明确说明。",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    agent = create_react_agent(llm, [query_medical_kb], prompt=prompt)
    return agent


agent = build_agent()


st.title("🩺 医疗健康 Agent")
st.caption("我只是健康科普助手，不能替代医生诊断。紧急情况请及时就医。")

# ---- 多轮记忆：保存在页面会话状态里，跨轮积累 ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示已有对话
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# ---- 输入框 ----
user_input = st.chat_input("输入你的健康问题…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            msgs = [{"role": r, "content": c} for r, c in st.session_state.messages]
            try:
                result = agent.invoke({"messages": msgs})
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"出错了：{e}"
        st.markdown(answer)
    st.session_state.messages.append(("assistant", answer))

    # 简单截断保底：只保留最近 40 条，避免无限制增长
    if len(st.session_state.messages) > 40:
        st.session_state.messages = st.session_state.messages[-40:]