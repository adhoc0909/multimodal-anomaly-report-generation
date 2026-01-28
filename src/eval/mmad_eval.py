from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
from tqdm import tqdm

from ..datasets.mmad import load_mmad_samples, get_templates
from ..common.io import imread_bgr

def parse_questions(meta: dict) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Parse MMAD mmad.json format:
    meta["conversation"] is list of dicts with keys:
      - Question
      - Answer  (letter)
      - Options (dict: {A:..., B:...})
      - type
    """
    conv = meta.get("conversation", [])
    questions, answers, qtypes = [], [], []
    for item in conv:
        q_text = item.get("Question") or item.get("question")
        ans = item.get("Answer") or item.get("answer")
        opts = item.get("Options") or item.get("options") or {}
        qt = item.get("type") or item.get("question_type") or "unknown"
        if q_text is None or ans is None:
            continue
        # normalize options
        norm_opts = []
        if isinstance(opts, dict):
            # keep key ordering A,B,C...
            for k in sorted(opts.keys()):
                norm_opts.append({"label": str(k).strip(), "text": str(opts[k])})
        elif isinstance(opts, list):
            for j,o in enumerate(opts):
                norm_opts.append({"label": chr(ord("A")+j), "text": str(o)})
        else:
            norm_opts = []
        questions.append({"text": str(q_text), "options": norm_opts})
        answers.append(str(ans).strip())
        qtypes.append(str(qt))
    return questions, answers, qtypes

def evaluate_mmad(*, runtime_cfg, anomaly_model, mllm_client, few_shot_k: int, use_similar: bool, max_images: int | None, out_json: Path) -> dict:
    samples = load_mmad_samples(runtime_cfg.paths.mmad_json)
    if max_images is not None:
        samples = samples[:int(max_images)]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    records = []
    total = correct = 0
    by_type: Dict[str, List[int]] = {}

    for s in tqdm(samples, desc="MMAD eval"):
        image_abs = str(Path(runtime_cfg.paths.data_root) / s.image_rel)
        # templates (not used by Dummy model yet, but kept for Step 2)
        template_rels = get_templates(s.meta, k=few_shot_k, use_similar=use_similar)
        template_abs = [str(Path(runtime_cfg.paths.data_root)/p) for p in template_rels]

        try:
            img = imread_bgr(image_abs)
        except FileNotFoundError:
            # Dataset images may not be present in some setups (e.g., only json/metadata downloaded).
            # Skip safely so you can still verify the pipeline/code structure.
            continue
        templates_bgr = []
        for p in template_abs:
            try:
                templates_bgr.append(imread_bgr(p))
            except FileNotFoundError:
                pass
        ar = anomaly_model.infer(img, templates_bgr=templates_bgr if templates_bgr else None)

        # structured info for MLLM answer (Step 1 is heuristic)
        # minimal: defect exists if score above a tiny threshold
        structured = {
            "has_defect": bool(ar.score > 0.01),
            "confidence": max(0.5, min(0.99, float(ar.score) * 3.0)),
        }

        qs, ans, qtypes = parse_questions(s.meta)
        for q, a, qt in zip(qs, ans, qtypes):
            pred = mllm_client.answer_mcq(question=q, structured=structured)
            ok = str(pred).strip().upper() == str(a).strip().upper()
            total += 1
            correct += int(ok)
            by_type.setdefault(qt, [0,0])
            by_type[qt][0] += 1
            by_type[qt][1] += int(ok)
            records.append({
                "image": s.image_rel,
                "templates": template_rels,
                "question_type": qt,
                "question": q,
                "correct_answer": a,
                "pred_answer": pred,
                "is_correct": ok,
            })

    acc = correct / total if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "by_type": {k: {"total": v[0], "correct": v[1], "acc": (v[1]/v[0] if v[0] else 0.0)} for k,v in by_type.items()}
    }
    out_json.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
