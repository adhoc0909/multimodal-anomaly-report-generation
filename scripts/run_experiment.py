"""Experiment Runner — run MMAD evaluation with YAML config + CLI overrides.

Usage:
    # Use YAML config
    python scripts/run_experiment.py

    # CLI overrides
    python scripts/run_experiment.py --llm gpt-4o --ad-model null
    python scripts/run_experiment.py --llm qwen --ad-model patchcore
    python scripts/run_experiment.py --llm qwen --max-images 5

    # RAG comparison
    python scripts/run_experiment.py --llm internvl3.5-2b              # baseline
    python scripts/run_experiment.py --llm internvl3.5-2b --rag        # with RAG

    # List available models
    python scripts/run_experiment.py --list-models

    # Custom config file
    python scripts/run_experiment.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm

# Suppress noisy C++ library warnings (CUDA/XLA/absl) — must be set before any imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Load .env (HF_TOKEN, API keys 등)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
warnings.filterwarnings('ignore')

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.config.experiment import ExperimentConfig, load_experiment_config
from src.ad import load_ad_predictions_file, normalize_image_key, to_llm_ad_info
from src.mllm.factory import MODEL_REGISTRY, get_llm_client, list_llm_models
from src.mllm.base import format_ad_info
from src.eval.metrics import calculate_accuracy_mmad
from src.utils.log import setup_logger


def stratified_sample(image_paths: list[str], n_per_folder: int, seed: int = 42) -> list[str]:
    """폴더(dataset/category/split)별 N장 샘플링.

    이미지 경로를 {dataset}/{category}/{split} 기준으로 그룹핑 후
    각 그룹에서 최대 n_per_folder장을 랜덤 추출한다.
    """
    rng = random.Random(seed)

    folders = defaultdict(list)
    for path in image_paths:
        parts = path.split("/")
        if len(parts) >= 4:
            key = f"{parts[0]}/{parts[1]}/{parts[3]}"  # dataset/category/split(good|bad)
        elif len(parts) >= 2:
            key = f"{parts[0]}/{parts[1]}"
        else:
            key = "unknown"
        folders[key].append(path)

    sampled = []
    for key in sorted(folders.keys()):
        imgs = folders[key]
        sampled.extend(rng.sample(imgs, min(n_per_folder, len(imgs))))

    n_good = sum(1 for s in sampled if "/good/" in s)
    n_bad = len(sampled) - n_good
    print(f"Stratified sampling: {n_per_folder}장/폴더, {len(folders)}폴더")
    print(f"  Total: {len(image_paths)} -> Sampled: {len(sampled)} (normal={n_good}, anomaly={n_bad})")

    return sampled


def load_mmad_data(json_path: str) -> dict:
    """Load MMAD dataset JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ad_predictions(ad_output_path: str) -> dict:
    """Load AD predictions JSON and index by normalized image path."""
    return load_ad_predictions_file(ad_output_path)


def run_ad_inference(cfg: ExperimentConfig, data_root: str, mmad_json: str) -> str:
    """Run AD model inference using run_ad_inference.py.

    If cfg.ad_output already points to an existing file, skip inference.
    Otherwise run scripts/run_ad_inference.py via subprocess.
    """
    # ad.output를 명시적으로 지정한 경우에만 스킵 (사용자가 의도적으로 재사용)
    if cfg.ad_output and Path(cfg.ad_output).exists():
        print(f"Using existing AD predictions (ad.output): {cfg.ad_output}")
        return cfg.ad_output

    # Determine output path — 항상 새로 실행
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ad_output = str(output_dir / f"{cfg.ad_model}_predictions.json")

    # Find inference script
    inference_script = str(PROJ_ROOT / "scripts" / "run_ad_inference.py")
    if not Path(inference_script).exists():
        print(f"Error: inference script not found: {inference_script}")
        sys.exit(1)

    # Checkpoint dir
    checkpoint_dir = cfg.ad_checkpoint_dir
    if not checkpoint_dir:
        print("Error: ad.checkpoint_dir is not set in experiment.yaml")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, inference_script,
        "--checkpoint-dir", checkpoint_dir,
        "--data-root", data_root,
        "--mmad-json", mmad_json,
        "--output", ad_output,
        "--output-format", "report",
    ]

    ad_config = cfg.ad_config or "configs/anomaly.yaml"
    if Path(ad_config).exists():
        cmd.extend(["--config", ad_config])

    if cfg.ad_threshold is not None:
        cmd.extend(["--threshold", str(cfg.ad_threshold)])

    if cfg.ad_version is not None:
        cmd.extend(["--version", str(cfg.ad_version)])

    device = "cuda"
    cmd.extend(["--device", device])

    version_str = f"v{cfg.ad_version}" if cfg.ad_version is not None else "latest"
    print("=" * 60)
    print("Running AD Model Inference")
    print("=" * 60)
    print(f"Script:       {inference_script}")
    print(f"Config:       {ad_config}")
    print(f"Checkpoint:   {checkpoint_dir}")
    print(f"Version:      {version_str}")
    print(f"Data root:    {data_root}")
    print(f"MMAD JSON:    {mmad_json}")
    print(f"Output:       {ad_output}")
    print()

    # 실시간 출력 (버퍼링 없이 바로 표시)
    process = subprocess.Popen(
        cmd, cwd=str(PROJ_ROOT),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    process.wait()

    if process.returncode != 0:
        print(f"Error: AD inference failed (exit code {process.returncode})")
        sys.exit(1)

    if not Path(ad_output).exists():
        print(f"Error: AD inference did not produce output: {ad_output}")
        print(f"Check the inference log above for details.")
        sys.exit(1)

    return ad_output


def resolve_paths(cfg: ExperimentConfig) -> tuple[str, str]:
    """Resolve data_root and mmad_json from config, env vars, or defaults."""
    data_root = cfg.data_root or os.environ.get("MMAD_DATA_ROOT")
    mmad_json = cfg.mmad_json or os.environ.get("MMAD_JSON_PATH")

    if not data_root:
        candidates = [
            PROJ_ROOT / "datasets" / "MMAD",
            Path("/Users/leehw/Documents/likelion/final_project/MMAD/dataset/MMAD"),
        ]
        for c in candidates:
            if c.exists():
                data_root = str(c)
                break

    if not mmad_json and data_root:
        mmad_json = str(Path(data_root) / "mmad.json")

    return data_root, mmad_json


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run a single experiment and return the answers JSON path."""
    data_root, mmad_json = resolve_paths(cfg)

    # Validate paths
    if not mmad_json or not Path(mmad_json).exists():
        print(f"Error: mmad.json not found at {mmad_json}")
        print("Set --data-root, MMAD_DATA_ROOT, or data_root in YAML")
        sys.exit(1)

    if not data_root or not Path(data_root).exists():
        print(f"Error: Data root not found at {data_root}")
        print("Set --data-root, MMAD_DATA_ROOT, or data_root in YAML")
        sys.exit(1)

    # Setup output
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name="Experiment", log_prefix="experiment", console_logging=False)

    # Load dataset & sampling (AD inference 전에 먼저 수행)
    mmad_data = load_mmad_data(mmad_json)
    image_paths = list(mmad_data.keys())
    total_available = len(image_paths)

    # Stratified sampling (폴더별 N장)
    if cfg.sample_per_folder:
        image_paths = stratified_sample(image_paths, cfg.sample_per_folder, cfg.sample_seed)

    # max_images는 샘플링 이후에 적용
    if cfg.max_images:
        image_paths = image_paths[:cfg.max_images]

    template_type = "Similar_template" if cfg.similar_template else "Random_template"
    ad_suffix = f"_with_{cfg.ad_model}" if cfg.ad_model else ""
    version_suffix = f"_v{cfg.ad_version}" if cfg.ad_model and cfg.ad_version is not None else ""
    rag_suffix = "_rag" if cfg.rag else ""
    llm_safe = cfg.llm.replace("/", "_").replace("\\", "_")
    img_count = f"_{len(image_paths)}img"
    output_name = f"answers_{cfg.few_shot}_shot_{llm_safe}_{template_type}{ad_suffix}{version_suffix}{rag_suffix}{img_count}"
    answers_json_path = output_dir / f"{output_name}.json"

    # Initialize RAG retriever if enabled
    rag_retriever = None
    if cfg.rag:
        from src.rag import Indexer, Retrievers
        from src.rag.prompt import rag_prompt

        rag_json = cfg.rag_json_path or str(Path(data_root) / "domain_knowledge.json")
        if not Path(rag_json).exists():
            print(f"Error: domain_knowledge.json not found at {rag_json}")
            sys.exit(1)

        rag_persist_dir = getattr(cfg, "rag_persist_dir", None) or "vectorstore/domain_knowledge"
        indexer = Indexer(json_path=rag_json, persist_dir=rag_persist_dir)
        vectorstore = indexer.get_or_create()
        rag_retriever = Retrievers(vectorstore)
        print(f"RAG enabled: {vectorstore._collection.count()} documents indexed")

    print("=" * 60)
    print("MMAD Experiment Runner")
    print("=" * 60)
    print(f"Experiment:  {cfg.experiment_name}")
    print(f"LLM:         {cfg.llm}")
    print(f"AD model:    {cfg.ad_model or 'none'}")
    print(f"RAG:         {'enabled' if cfg.rag else 'disabled'}")
    print(f"Few-shot:    {cfg.few_shot}")
    print(f"Template:    {template_type}")
    print(f"Image size:  {cfg.max_image_size}")
    print(f"Images:      {len(image_paths)} / {total_available}")
    print(f"Data root:   {data_root}")
    print(f"Output:      {answers_json_path}")
    print("=" * 60)
    print()

    logger.info(
        f"[Config] experiment={cfg.experiment_name} | llm={cfg.llm} | "
        f"ad={cfg.ad_model or 'none'} | rag={cfg.rag} | "
        f"few_shot={cfg.few_shot} | images={len(image_paths)}"
    )

    # 샘플링된 이미지만으로 필터링된 MMAD json 생성 (AD inference용)
    sampled_mmad_json = mmad_json
    if len(image_paths) < total_available:
        sampled_data = {k: mmad_data[k] for k in image_paths}
        sampled_json_path = output_dir / "_sampled_mmad.json"
        with open(sampled_json_path, "w", encoding="utf-8") as f:
            json.dump(sampled_data, f, ensure_ascii=False)
        sampled_mmad_json = str(sampled_json_path)
        print(f"Filtered MMAD json: {len(sampled_data)} images -> {sampled_json_path}")
        print()

    # Run AD inference if ad_model is set (샘플링된 이미지만 처리)
    ad_predictions = None
    if cfg.ad_model:
        ad_output_path = run_ad_inference(cfg, data_root, sampled_mmad_json)
        ad_predictions = load_ad_predictions(ad_output_path)
        print(f"Loaded {len(ad_predictions)} AD predictions")
        print()

    # Load existing results if resuming
    all_answers = []
    existing_images = set()

    if cfg.resume and answers_json_path.exists():
        with open(answers_json_path, "r", encoding="utf-8") as f:
            all_answers = json.load(f)
        existing_images = {a["image"] for a in all_answers}
        print(f"Resuming from {len(existing_images)} existing images")

    # Initialize LLM client
    try:
        llm_client = get_llm_client(cfg.llm, max_image_size=cfg.max_image_size)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Warm up local model before timing starts (excludes cold-start from per-image latency)
    if hasattr(llm_client, 'load_model'):
        print("Warming up model (pre-loading weights)...")
        llm_client.load_model()
        print("Warming up with dummy forward pass...")
        try:
            warmup_rel = image_paths[0]
            warmup_path = str(Path(data_root) / warmup_rel)
            warmup_meta = mmad_data[warmup_rel]
            if cfg.batch_mode:
                llm_client.generate_answers_batch(warmup_path, warmup_meta, [], ad_info=None, instruction=None)
            else:
                llm_client.generate_answers(warmup_path, warmup_meta, [], ad_info=None, instruction=None)
        except Exception:
            pass
        print("Model ready.")
        print()

    # Track statistics
    total_correct = 0
    total_questions = 0
    processed = 0
    errors = 0
    start_time = time.time()
    class_latencies: dict = defaultdict(list)

    # Evaluate with progress bar
    pbar = tqdm(image_paths, desc="Evaluating", ncols=100)
    for image_rel in pbar:
        if image_rel in existing_images:
            continue

        meta = mmad_data[image_rel]

        # Get templates
        if cfg.similar_template:
            few_shot = meta.get("similar_templates", [])[:cfg.few_shot]
        else:
            few_shot = meta.get("random_templates", [])[:cfg.few_shot]

        # Build absolute paths
        query_image_path = str(Path(data_root) / image_rel)
        few_shot_paths = [str(Path(data_root) / p) for p in few_shot]

        # Check if image exists
        if not Path(query_image_path).exists():
            errors += 1
            continue

        # Get AD prediction for this image
        ad_info = None
        if ad_predictions is not None:
            image_key = normalize_image_key(image_rel)
            ad_raw = ad_predictions.get(image_key)
            if ad_raw is not None:
                ad_info = to_llm_ad_info(ad_raw)

        # Build RAG instruction if enabled
        instruction = None
        if rag_retriever is not None:
            parts = image_rel.split("/")
            ds_name = parts[0] if len(parts) > 0 else ""
            cat_name = parts[1] if len(parts) > 1 else ""

            # defect_type 유출 방지: ground-truth defect_type을 쿼리/필터에 사용하지 않음
            # 프로덕션과 동일한 generic 쿼리 사용 (category 기반)
            query = rag_retriever.build_generic_query(cat_name)
            docs = rag_retriever.retrieve(query, category=cat_name, defect_type=None, k=cfg.rag_k)
            domain_knowledge = rag_retriever.format_context(docs)

            ad_info_str = format_ad_info(ad_info) if ad_info else ""
            instruction = rag_prompt(ad_info=ad_info_str, domain_knowledge=domain_knowledge)

        # Generate answers
        img_start = time.time()
        if cfg.batch_mode:
            questions, answers, predicted, q_types = llm_client.generate_answers_batch(
                query_image_path, meta, few_shot_paths, ad_info=ad_info, instruction=instruction
            )
        else:
            questions, answers, predicted, q_types = llm_client.generate_answers(
                query_image_path, meta, few_shot_paths, ad_info=ad_info, instruction=instruction
            )
        img_latency = time.time() - img_start

        if predicted is None or len(predicted) != len(answers):
            errors += 1
            continue

        # Accumulate per-class latency (성공한 이미지만)
        parts = image_rel.split("/")
        cls_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else parts[0]
        class_latencies[cls_key].append(img_latency)

        # Calculate accuracy for this image
        correct = sum(1 for p, a in zip(predicted, answers) if p == a)
        total_correct += correct
        total_questions += len(answers)
        processed += 1

        # Update progress bar with running accuracy
        running_acc = total_correct / total_questions if total_questions > 0 else 0
        pbar.set_postfix({"acc": f"{running_acc:.1%}", "done": processed, "err": errors}, refresh=False)

        # Store results
        for q, a, pred, qt in zip(questions, answers, predicted, q_types):
            all_answers.append({
                "image": image_rel,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": pred,
            })

        # Save incrementally
        with open(answers_json_path, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, indent=4, ensure_ascii=False)

    pbar.close()

    elapsed = time.time() - start_time
    final_acc = total_correct / total_questions if total_questions > 0 else 0

    # Print & log class-level latency table
    if class_latencies:
        print()
        print("=" * 60)
        print("Latency by Class")
        print("=" * 60)
        print(f"  {'Class':<32} {'N':>4}  {'Avg(s)':>7}  {'Total(s)':>9}")
        print("-" * 60)
        for cls_key in sorted(class_latencies.keys()):
            lats = class_latencies[cls_key]
            avg_lat = sum(lats) / len(lats)
            total_lat = sum(lats)
            print(f"  {cls_key:<32} {len(lats):>4}  {avg_lat:>7.2f}  {total_lat:>9.2f}")
            logger.info(f"[Latency] {cls_key} | n={len(lats)} | avg={avg_lat:.3f}s | total={total_lat:.1f}s")
        print("-" * 60)
        overall_avg = elapsed / processed if processed > 0 else 0
        print(f"  {'Overall':<32} {processed:>4}  {overall_avg:>7.2f}  {elapsed:>9.1f}")
        print()

    # Save metadata
    meta_path = answers_json_path.with_suffix(".meta.json")
    meta_info = {
        "experiment_name": cfg.experiment_name,
        "llm": cfg.llm,
        "ad_model": cfg.ad_model,
        "ad_version": cfg.ad_version,
        "rag": cfg.rag,
        "few_shot": cfg.few_shot,
        "similar_template": cfg.similar_template,
        "sample_per_folder": cfg.sample_per_folder,
        "max_images": cfg.max_images,
        "total_images": len(image_paths),
        "processed": processed,
        "errors": errors,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "accuracy": round(final_acc * 100, 2),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "answers_file": str(answers_json_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 60)
    print("Calculating Metrics")
    print("=" * 60)
    if total_questions > 0:
        calculate_accuracy_mmad(str(answers_json_path))
    else:
        print("No valid answers to evaluate")
    print()
    print(f"Results saved to: {answers_json_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Elapsed: {elapsed:.1f}s")

    overall_avg = elapsed / processed if processed > 0 else 0
    logger.info(
        f"[Result] acc={round(final_acc * 100, 2)}% | "
        f"questions={total_questions} | correct={total_correct} | "
        f"elapsed={elapsed:.1f}s | avg={overall_avg:.3f}s/img"
    )

    return answers_json_path


def main():
    parser = argparse.ArgumentParser(description="MMAD Experiment Runner")

    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Experiment config YAML (default: configs/experiment.yaml)")

    # CLI overrides
    parser.add_argument("--llm", type=str, default=None,
                        help="Override LLM model name")
    parser.add_argument("--ad-model", type=str, default=None,
                        help="Override AD model (null = no AD)")
    parser.add_argument("--ad-output", type=str, default=None,
                        help="Path to existing AD predictions JSON (skip inference)")
    parser.add_argument("--ad-version", type=int, default=None,
                        help="Override AD checkpoint version (null = latest)")
    parser.add_argument("--few-shot", type=int, default=None,
                        help="Override few-shot count")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Override max images")
    parser.add_argument("--sample-per-folder", type=int, default=None,
                        help="Override sample count per folder (stratified sampling)")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="Override sampling seed")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--mmad-json", type=str, default=None,
                        help="Override mmad json path (default: {data_root}/mmad.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--batch-mode", type=str, default=None, choices=["true", "false"],
                        help="Override batch mode (default: auto per model)")
    parser.add_argument("--resume", action="store_true", default=None,
                        help="Resume from existing results")

    # RAG
    parser.add_argument("--rag", action="store_true", default=None,
                        help="Enable RAG domain knowledge injection")
    parser.add_argument("--rag-json", type=str, default=None,
                        help="Path to domain_knowledge.json (default: {data_root}/domain_knowledge.json)")
    parser.add_argument("--rag-persist-dir", type=str, default=None,
                        help="Chroma vectorstore 경로 (default: vectorstore/domain_knowledge). "
                             "Config A/B/C 비교 실험 시 각각 다른 경로 지정.")
    parser.add_argument("--rag-top-k", type=int, default=None,
                        help="RAG 검색 문서 수 (default: 3)")

    # Utility
    parser.add_argument("--list-models", action="store_true",
                        help="List available LLM models and exit")

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("Available LLM models:")
        print("-" * 40)
        for name in list_llm_models():
            info = MODEL_REGISTRY[name]
            print(f"  {name:<20s} ({info['type']}) {info['model']}")
        return

    # Load config from YAML
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_experiment_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        cfg = ExperimentConfig()

    # Apply CLI overrides
    if args.llm is not None:
        cfg.llm = args.llm
    if args.ad_model is not None:
        cfg.ad_model = None if args.ad_model.lower() == "null" else args.ad_model
    if args.ad_output is not None:
        cfg.ad_output = args.ad_output
    if args.ad_version is not None:
        cfg.ad_version = args.ad_version
    if args.few_shot is not None:
        cfg.few_shot = args.few_shot
    if args.max_images is not None:
        cfg.max_images = args.max_images
    if args.sample_per_folder is not None:
        cfg.sample_per_folder = args.sample_per_folder
    if args.sample_seed is not None:
        cfg.sample_seed = args.sample_seed
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.mmad_json is not None:
        cfg.mmad_json = args.mmad_json
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.batch_mode is not None:
        cfg.batch_mode = args.batch_mode == "true"
    if args.resume is not None:
        cfg.resume = args.resume
    if args.rag is not None:
        cfg.rag = args.rag
    if args.rag_json is not None:
        cfg.rag_json_path = args.rag_json
    if args.rag_persist_dir is not None:
        cfg.rag_persist_dir = args.rag_persist_dir
    if args.rag_top_k is not None:
        cfg.rag_k = args.rag_top_k

    # 모델별 기본값 적용 (CLI로 명시하지 않은 경우)
    _BATCH_MODE_DEFAULTS = {
        "internvl": True,
        "internvl3.5-8b": True,
        "internvl2.5-8b": True,
        "internvl-8b": True,
        "llava": False,
        "llava-onevision": False,
    }
    if args.batch_mode is None and cfg.llm in _BATCH_MODE_DEFAULTS:
        cfg.batch_mode = _BATCH_MODE_DEFAULTS[cfg.llm]

    run_experiment(cfg)


if __name__ == "__main__":
    main()
