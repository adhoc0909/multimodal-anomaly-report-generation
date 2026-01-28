from __future__ import annotations
from datetime import datetime
from typing import Any, Dict
import random

class EchoMLLM:
    """Step 1 placeholder.

    - `answer_mcq`: heuristic answer based on whether a defect exists.
    - `generate_report`: deterministic Korean/English template report.
    """

    def __init__(self, language: str = "ko", seed: int = 0):
        self.language = language
        random.seed(seed)

    def answer_mcq(self, *, question: Dict[str, Any], structured: Dict[str, Any]) -> str:
        opts = question.get("options", [])
        if not opts:
            return "A"
        has_defect = bool(structured.get("has_defect", False))
        anomaly_words = ["anomaly", "defect", "abnormal", "결함", "이상", "불량"]
        normal_words = ["normal", "good", "정상", "양품"]

        def score_opt(text: str) -> int:
            t = text.lower()
            s = 0
            for w in anomaly_words:
                if w in t:
                    s += 2
            for w in normal_words:
                if w in t:
                    s -= 1
            return s

        scored = [(i, score_opt(o.get("text",""))) for i, o in enumerate(opts)]
        idx = max(scored, key=lambda x: x[1])[0] if has_defect else min(scored, key=lambda x: x[1])[0]
        return opts[idx].get("label", "A")

    def generate_report(self, *, structured: Dict[str, Any]) -> Dict[str, Any]:
        has_defect = bool(structured.get("has_defect", False))
        decision = "anomaly" if has_defect else "normal"
        conf = float(structured.get("confidence", 0.6))
        loc = structured.get("location", "none")
        area = float(structured.get("area_ratio", 0.0))
        shape = structured.get("shape", "none")
        severity = structured.get("severity", "low")

        if self.language == "ko":
            summary = f"검사 결과: {'불량(이상)' if has_defect else '정상'}로 판단됩니다. 위치={loc}, 면적비율={area*100:.2f}%, 형태={shape}, 심각도={severity}."
            impact = "제품 기능/신뢰성 저하 가능성이 있습니다." if has_defect else "특이사항 없습니다."
            reco = "라인에서 분리 후 재검 및 원인 분석을 권고합니다." if has_defect else "출하 가능합니다."
        else:
            summary = f"Decision={decision}. loc={loc}, area={area:.4f}, shape={shape}, severity={severity}."
            impact = "Potential functional risk." if has_defect else "No issue."
            reco = "Remove and re-inspect." if has_defect else "OK to ship."

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "confidence": conf,
            "defect": {"location": loc, "area_ratio": area, "shape": shape, "severity": severity},
            "summary": summary,
            "impact": impact,
            "recommendation": reco,
        }
