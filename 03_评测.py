"""
03 RAG 效果评测（检索召回率 + 答题准确率）

运行：python 03_评测.py   （先确保已跑过 01 建库、02 能问答）

原理：
  准备一份 "测试题目集"，每题包含（问题、期望回答必须出现的关键词）。
  对每题执行完整 RAG（检索+回答），用自动关键词匹配判断“这题算不算对”。
  最后统计两个指标：
    答题准确率(Accuracy)：回答里包含期望关键词的题目占比
    检索召回率(Recall)  ：检索结果里至少出现一条含期望关键词片段的题目占比
这两个数字就是你在简历/面试里能讲的“效果提升数据”。
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")


# ---- 1. 测试题目集（可自行扩充；"expected" 是期望回答里出现的关键词，多个用列表，命中任意一个即算对） ----
TEST_CASES = [
    {"question": "高血压患者饮食上要注意什么？", "expected": ["钠盐", "食盐"]},
    {"question": "高血压患者该怎么锻炼？",         "expected": ["有氧运动", "快走", "慢跑", "150 分钟"]},
    {"question": "糖尿病患者饮食有什么原则？",     "expected": ["热量", "低升糖", "主食", "燕麦"]},
    {"question": "糖尿病患者低血糖时怎么办？",     "expected": ["糖果", "心慌", "血糖"]},
    {"question": "常用降压药有哪些类型？",         "expected": ["钙通道阻滞剂", "ACEI", "利尿剂"]},
]


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})


def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )


def check(expected: list, text: str) -> bool:
    """text 里是否出现 expected 中的任意一个关键词。"""
    return any(k in text for k in expected)


def main():
    retriever = get_retriever()
    llm = get_llm()

    total = len(TEST_CASES)
    hit = 0      # 检索召回命中数（检索结果里出现含期望关键词的片段）
    correct = 0  # 答题答对数（回答里出现期望关键词）

    print(f"评测开始，共 {total} 题...\n")
    for i, case in enumerate(TEST_CASES, 1):
        q = case["question"]
        exp = case["expected"]

        # ---- 检索 ----
        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content for d in docs)

        # 检索是否命中：检索出的片段文本里包含任一期望关键词
        if check(exp, context):
            hit += 1
            hit_ok = "✔"
        else:
            hit_ok = "✘"

        # ---- 回答 ----
        prompt = f"""你是专业的健康顾问。请只依据下面提供的资料回答用户问题。
如果资料里没有相关信息，请明确说"资料中未提及"，不要自行编造。

【资料】
{context}

【问题】
{q}

请给出简洁、准确的回答。"""
        answer = llm.invoke(prompt).content

        if check(exp, answer):
            correct += 1
            ans_ok = "✔"
        else:
            ans_ok = "✘"

        print(f"[{i}/{total}] 题目：{q}")
        print(f"        检索命中：{hit_ok}   答题正确：{ans_ok}")
        print(f"        回答：{answer[:80]}...\n")

    # ---- 汇总 ----
    print("=" * 50)
    print(f"检索召回率 Recall ：{hit}/{total} = {hit/total:.0%}")
    print(f"答题准确率 Acc    ：{correct}/{total} = {correct/total:.0%}")


if __name__ == "__main__":
    main()