"""
05 图片 OCR → 存入知识库

作用：把一张含文字的图片（如化验单、报告、截图）OCR 提取成文本，
     追加保存到 data/ 下的 markdown 文件；之后再跑 `01_建知识库.py` 重建向量库，
     你的 Agent 就能检索到图片里的内容了。

用法：
  1）先安装：pip install paddleocr paddlepaddle
  2）python 05_ocr入库.py  <图片路径>  [备注名]
     例：python 05_ocr入库.py "D:/报告.jpg" "患者张三血常规"

  之后重新跑：python 01_建知识库.py
"""
import sys
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_MD = os.path.join(DATA_DIR, "ocr_uploads.md")


def ocr_image(img_path: str) -> str:
    # 用 RapidOCR（onnxruntime，模型随包自带、无需联网下载），
    # 替代 PaddleOCR——它更稳、无 paddle/PIR/onednn 兼容坑。
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    out = engine(img_path)
    # engine() 返回 (识别结果列表, 耗时) 元组；兼容只返回列表的情况
    result = out[0] if isinstance(out, tuple) else out

    lines = []
    if result:
        for item in result:
            # 每条： [文本框坐标, 文本, 置信度]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = item[1]
                if isinstance(txt, (list, tuple)):  # 个别版本文本是 (text, score)
                    txt = txt[0]
                lines.append(str(txt))
            else:
                lines.append(str(item))
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法：python 05_ocr入库.py <图片路径> [备注名]")
        return
    img_path = sys.argv[1]
    note = sys.argv[2] if len(sys.argv) > 2 else "OCR导入"

    if not os.path.exists(img_path):
        print(f"❌ 找不到图片：{img_path}")
        return

    print(f"🔍 正在识别图片：{img_path} ...")
    text = ocr_image(img_path)
    if not text.strip():
        print("⚠️ 未识别到任何文字（可能不是文字图片）。")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    header = (
        f"\n## {note}（OCR 识别，{datetime.now():%Y-%m-%d %H:%M}）\n"
        f"> 来源图片：{os.path.basename(img_path)}\n"
    )
    body = header + text + "\n"
    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write(body)

    print(f"✅ 已追加到：{OUT_MD}")
    print(f"   下一步请重跑：python 01_建知识库.py 让 Agent 能检索到这段内容")
    print("---- 识别到的文字预览 ----")
    print(text[:800])


if __name__ == "__main__":
    main()