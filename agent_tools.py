# -*- coding: utf-8 -*-
"""
医知助手 · 多工具集（检索型 + 计算型）
  query_medical_kb  检索医疗知识库（混合检索）
  emergency_triage  症状紧急分级（规则判断）
  calc_dosage       按体重计算用药剂量
  convert_units     医疗单位换算

所有入口（04 / server.py / 06）统一 from agent_tools import build_tools 使用，
保证命令行与网页版行为一致。
"""
import os

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
# 重排置信度阈值：低于该值的来源片段视为不相关，不提供给 Agent，也不展示。
# 依据实测：本库相关片段重排分 >0.5，无关 <0.1，取 0.2 可干净分裂。保持适度，避免误删相关。
MIN_SOURCE_SCORE = 0.2

_RETRIEVER = None
_FLAG = {"retrieved": False}


def _get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        from hybrid_retriever import HybridRetriever

        _RETRIEVER = HybridRetriever(DB_DIR)
    return _RETRIEVER


def get_retriever():
    return _get_retriever()


def reset_flag():
    _FLAG["retrieved"] = False


def was_retrieved():
    return _FLAG["retrieved"]


from langchain.tools import tool


@tool
def query_medical_kb(question: str) -> str:
    """检索医疗知识库，返回与问题最相关的资料原文，可用于核实疾病、症状、饮食、用药等健康事实。"""
    _FLAG["retrieved"] = True
    print(f">>> [Agent决策] 调用了查询知识库工具 ← 问题：{question!r}")
    docs = _get_retriever().search(question, k=3, min_score=MIN_SOURCE_SCORE)
    return "\n\n".join(docs) or "知识库中暂时没有足够相关的资料。"


@tool
def emergency_triage(symptom: str) -> str:
    """根据患者描述的症状做紧急程度分级（规则判断），返回"紧急/次紧急/不紧急"及就医建议。"""
    critical = [
        "胸痛", "胸闷", "呼吸困难", "喘不上气", "昏迷", "意识不清", "抽搐",
        "大出血", "呕血", "便血不止", "剧烈头痛", "不能说话", "半边身体麻木",
    ]
    urgent = [
        "持续高热", "反复呕吐", "剧烈腹痛", "头晕眼花", "心悸明显", "外伤出血",
    ]
    hit = [k for k in critical if k in symptom]
    if hit:
        return (
            f"⚠️ 紧急：检测到高危关键词【{'、'.join(hit)}】，"
            "请立即拨打 120 或前往急诊就医，不要自行处理。"
        )
    hit2 = [k for k in urgent if k in symptom]
    if hit2:
        return (
            f"🟠 次紧急：检测到关键词【{'、'.join(hit2)}】，"
            "建议尽快就医；前往途中保持观察，必要时呼叫急救。"
        )
    return (
        "🟢 不紧急：未检测到危急关键词。建议先休息观察；"
        "若症状持续或加重，请及时就医。"
    )


@tool
def calc_dosage(weight_kg: float, dose_per_kg: float, times_per_day: int = 1) -> str:
    """按体重计算用药剂量：单次剂量 = 体重 × 每公斤剂量；每日总量 = 单次剂量 × 每日次数。"""
    single = weight_kg * dose_per_kg
    daily = single * times_per_day
    return (
        f"按体重 {weight_kg:g} kg、剂量 {dose_per_kg:g} mg/kg、每日 {times_per_day} 次计算：\n"
        f"单次剂量 ≈ {single:.1f} mg，每日总剂量 ≈ {daily:.1f} mg。\n"
        "⚠️ 仅供参考，请以医生处方为准。"
    )


_MASS = {"mg": 1.0, "g": 1000.0, "kg": 1_000_000.0}
_VOLUME = {"ml": 1.0, "l": 1000.0, "毫升": 1.0, "升": 1000.0}


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """常用医疗单位换算：质量 mg/g/kg，体积 ml/L。例如 convert_units(250, 'mg', 'g')。"""
    f = str(from_unit).lower()
    t = str(to_unit).lower()
    if f in _MASS and t in _MASS:
        result = value * _MASS[f] / _MASS[t]
        return f"{value:g} {from_unit} = {result:g} {to_unit}"
    if f in _VOLUME and t in _VOLUME:
        result = value * _VOLUME[f] / _VOLUME[t]
        return f"{value:g} {from_unit} = {result:g} {to_unit}"
    return f"暂不支持 {from_unit} → {to_unit} 的换算，目前支持：mg/g/kg、ml/L。"


def build_tools():
    """返回 Agent 使用的全部工具列表。"""
    return [query_medical_kb, emergency_triage, calc_dosage, convert_units]
