from __future__ import annotations

import inspect
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


class AdService:
    """PatchCore checkpoint inference service for uploaded images."""

    def __init__(
        self,
        checkpoint_dir: str,
        output_root: str,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_cache: Dict[str, Any] = {}
        self.input_size = self._resolve_input_size()
        self.default_score_threshold = self._resolve_default_threshold()

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)
        except Exception:
            return float(default)

    def _resolve_input_size(self) -> Tuple[int, int]:
        default_size = (384, 384)

        raw_env = os.getenv("AD_INPUT_SIZE", "").strip()
        if raw_env:
            parts = [p for p in raw_env.replace("x", ",").split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    h = int(parts[0].strip())
                    w = int(parts[1].strip())
                    if h > 0 and w > 0:
                        return (h, w)
                except ValueError:
                    logger.warning("Invalid AD_INPUT_SIZE=%s, fallback to config/default", raw_env)

        cfg_path = Path(
            os.getenv(
                "ANOMALY_CONFIG_PATH",
                str(Path(__file__).resolve().parents[2] / "configs" / "anomaly.yaml"),
            )
        )
        try:
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                image_size = (cfg.get("data") or {}).get("image_size")
                if (
                    isinstance(image_size, (list, tuple))
                    and len(image_size) == 2
                    and int(image_size[0]) > 0
                    and int(image_size[1]) > 0
                ):
                    return (int(image_size[0]), int(image_size[1]))
        except Exception:
            logger.exception("Failed to read anomaly config image_size from %s", cfg_path)

        return default_size

    def _resolve_default_threshold(self) -> float:
        raw = os.getenv("AD_DEFAULT_THRESHOLD", "0.5").strip()
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid AD_DEFAULT_THRESHOLD=%s, using 0.5", raw)
            return 0.5

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Match existing PatchCore script path: resize -> RGB float [0,1] -> CHW tensor."""
        np_img = np.asarray(image, dtype=np.uint8)
        h, w = self.input_size
        resized = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1))
        return tensor

    def _iter_checkpoint_roots(self) -> List[Path]:
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if patchcore_dir.exists():
            return [patchcore_dir, self.checkpoint_dir]
        return [self.checkpoint_dir]

    @staticmethod
    def _select_latest_ckpt(category_dir: Path) -> Path | None:
        if not category_dir.exists() or not category_dir.is_dir():
            return None

        versions: List[Tuple[int, Path]] = []
        for v_dir in category_dir.iterdir():
            if not v_dir.is_dir() or not v_dir.name.startswith("v"):
                continue
            try:
                version = int(v_dir.name[1:])
            except ValueError:
                continue
            ckpt = v_dir / "model.ckpt"
            if ckpt.exists():
                versions.append((version, ckpt))
        if not versions:
            return None
        return max(versions, key=lambda x: x[0])[1]

    def _find_checkpoint(self, category_key: str) -> Tuple[Path, str] | None:
        key = category_key.strip().strip("/")
        if not key:
            return None

        if "/" in key:
            candidates: List[str] = [key]
        else:
            candidates = [key]

        for root in self._iter_checkpoint_roots():
            for cand in candidates:
                ckpt = self._select_latest_ckpt(root / cand)
                if ckpt is not None:
                    return ckpt, cand

            # category only key일 때 root/*/{category}/v*/model.ckpt 탐색
            if "/" not in key:
                ds_dirs = sorted(
                    [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")],
                    key=lambda p: p.name,
                ) if root.exists() else []
                for ds_dir in ds_dirs:
                    if not ds_dir.is_dir() or ds_dir.name.startswith("."):
                        continue
                    ckpt = self._select_latest_ckpt(ds_dir / key)
                    if ckpt is not None:
                        return ckpt, f"{ds_dir.name}/{key}"
        return None

    def get_model(self, dataset: str, category: str):
        del dataset  # kept for API compatibility
        found = self._find_checkpoint(category)
        if found is None:
            raise FileNotFoundError(f"Model not found for category key: {category}")
        model_path, resolved_key = found
        model_key = str(model_path)
        if model_key in self.model_cache:
            return self.model_cache[model_key]

        from anomalib.models import Patchcore

        load_kwargs: Dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": False,
            "pre_trained": False,
        }

        try:
            model = Patchcore.load_from_checkpoint(str(model_path), **load_kwargs)
        except TypeError as exc:
            if "pre_trained" not in str(exc):
                raise
            load_kwargs.pop("pre_trained", None)
            model = Patchcore.load_from_checkpoint(str(model_path), **load_kwargs)
        except Exception:
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            hyper = ckpt.get("hyper_parameters", {})
            sig = inspect.signature(Patchcore.__init__)
            allowed = set(sig.parameters.keys()) - {"self"}
            init_kwargs = {k: v for k, v in hyper.items() if k in allowed}
            init_kwargs["pre_trained"] = False
            model = Patchcore(**init_kwargs)
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(self.device)
        self.model_cache[model_key] = model
        logger.info("Loaded AD checkpoint: %s (resolved_category=%s)", model_path, resolved_key)
        return model

    @torch.no_grad()
    def predict_batch(
        self,
        upload_files: List[Any],
        category: str,
        dataset: str,
        threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        model = self.get_model(dataset, category)
        inputs = []
        originals: List[Image.Image] = []
        request_ids: List[str] = []

        for f in upload_files:
            req_id = str(uuid.uuid4())[:8]
            img = Image.open(f.file).convert("RGB")
            inputs.append(self._preprocess(img))
            originals.append(img)
            request_ids.append(req_id)

        batch = torch.stack(inputs).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(batch)

        if hasattr(outputs, "anomaly_map") and hasattr(outputs, "pred_score"):
            batch_maps = outputs.anomaly_map
            batch_scores = outputs.pred_score
            batch_labels = getattr(outputs, "pred_label", None)
        elif isinstance(outputs, (list, tuple)):
            batch_maps, batch_scores = outputs
            batch_labels = None
        else:
            batch_maps = outputs
            batch_scores = outputs.view(len(upload_files), -1).max(dim=1).values
            batch_labels = None

        score_threshold = float(self.default_score_threshold if threshold is None else threshold)
        post = getattr(model, "post_processor", None)
        if post is not None:
            if hasattr(post, "normalized_image_threshold"):
                score_threshold = self._to_float(getattr(post, "normalized_image_threshold"), score_threshold)
            elif hasattr(post, "image_threshold"):
                score_threshold = self._to_float(getattr(post, "image_threshold"), score_threshold)
        # run_ad_inference와 동일하게 defect location threshold도 image threshold를 사용한다.
        map_threshold = float(score_threshold)

        results: List[Dict[str, Any]] = []
        for i in range(len(upload_files)):
            amap = batch_maps[i].squeeze().cpu().numpy().astype(np.float32)

            w_orig, h_orig = originals[i].size
            if amap.shape != (h_orig, w_orig):
                amap_final = cv2.resize(amap, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            else:
                amap_final = amap

            orig_path = self.output_root / f"orig_{request_ids[i]}.png"
            heatmap_path = self.output_root / f"heat_{request_ids[i]}.png"
            mask_path = self.output_root / f"mask_{request_ids[i]}.png"
            overlay_path = self.output_root / f"overlay_{request_ids[i]}.png"
            score = round(float(batch_scores[i].cpu().item()), 6)
            if batch_labels is not None:
                model_is_anomaly = bool(batch_labels[i].detach().cpu().item())
            else:
                model_is_anomaly = bool(score > score_threshold)

            originals[i].save(orig_path)
            loc = self._save_visuals(
                orig_img=originals[i],
                amap=amap_final,
                h_path=heatmap_path,
                m_path=mask_path,
                o_path=overlay_path,
                threshold=map_threshold,
                anomaly_score=score,
            )

            results.append(
                {
                    "ad_score": score,
                    "model_is_anomaly": model_is_anomaly,
                    "model_threshold": float(score_threshold),
                    "mask_threshold": float(map_threshold),
                    "original_path": str(orig_path),
                    "heatmap_path": str(heatmap_path),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "region": loc.get("region"),
                    "has_defect": loc.get("has_defect", False),
                    "area_ratio": loc.get("area_ratio"),
                    "bbox": loc.get("bbox"),
                    "center": loc.get("center"),
                    "confidence": loc.get("confidence"),
                }
            )
        return results

    def _save_visuals(
        self,
        *,
        orig_img: Image.Image,
        amap: np.ndarray,
        h_path: Path,
        m_path: Path,
        o_path: Path,
        threshold: float = 0.5,
        anomaly_score: float | None = None,
    ) -> Dict[str, Any]:
        input_img = np.array(orig_img)
        h, w = amap.shape
        amap_raw = amap.astype(np.float32)
        amap_norm = (amap_raw - amap_raw.min()) / (amap_raw.max() - amap_raw.min() + 1e-8)
        amap_8bit = (amap_norm * 255).astype(np.uint8)
        mask_threshold = float(threshold)

        heatmap = cv2.applyColorMap(amap_8bit, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        Image.fromarray(cv2.addWeighted(input_img, 0.5, heatmap_rgb, 0.5, 0)).save(h_path)

        # run_ad_inference와 동일하게 raw anomaly_map을 thresholding.
        mask = (amap_raw > mask_threshold).astype(np.uint8) * 255
        loc: Dict[str, Any] = {
            "has_defect": False,
            "region": "none",
            "bbox": None,
            "center": None,
            "area_ratio": 0.0,
            "confidence": 0.0,
        }

        def infer_region(cx: float, cy: float) -> str:
            ry = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
            rx = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
            if ry == "middle" and rx == "center":
                return "center"
            return f"{ry}-{rx}"

        if np.any(mask > 0):
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = int(coords[0].min()), int(coords[0].max())
                x_min, x_max = int(coords[1].min()), int(coords[1].max())

                center_y = (y_min + y_max) / 2 / h
                center_x = (x_min + x_max) / 2 / w

                region = infer_region(center_x, center_y)

                area_ratio = round(float(np.sum(mask > 0)) / (h * w), 4)
                confidence = round(float(amap_raw[mask > 0].max()), 4)
                loc.update(
                    {
                        "has_defect": True,
                        "region": region,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "center": [round(center_x, 3), round(center_y, 3)],
                        "area_ratio": area_ratio,
                        "confidence": confidence,
                    }
                )
        elif anomaly_score is not None and float(anomaly_score) > mask_threshold:
            # Score는 NG인데 마스크가 비는 케이스 보정:
            # 최대 응답 지점 주변 최소 영역을 결함 후보로 시각화한다.
            y_peak, x_peak = np.unravel_index(int(np.argmax(amap_raw)), amap_raw.shape)
            radius = max(2, min(h, w) // 30)
            cv2.circle(mask, (int(x_peak), int(y_peak)), int(radius), 255, thickness=-1)
            x_min = max(0, int(x_peak) - radius)
            x_max = min(w - 1, int(x_peak) + radius)
            y_min = max(0, int(y_peak) - radius)
            y_max = min(h - 1, int(y_peak) + radius)
            center_x = float(x_peak) / float(max(1, w))
            center_y = float(y_peak) / float(max(1, h))
            area_ratio = round(float(np.sum(mask > 0)) / float(max(1, h * w)), 4)
            confidence = round(float(amap_raw[y_peak, x_peak]), 4)
            loc.update(
                {
                    "has_defect": True,
                    "region": infer_region(center_x, center_y),
                    "bbox": [x_min, y_min, x_max, y_max],
                    "center": [round(center_x, 3), round(center_y, 3)],
                    "area_ratio": area_ratio,
                    "confidence": confidence,
                }
            )

        Image.fromarray(mask).save(m_path)

        # Overlay는 예측 마스크 윤곽선을 원본 이미지 위에 빨간색으로 그린다.
        contour_overlay = input_img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 반투명 채우기 (원본 70% + 색 30%)
            fill_layer = contour_overlay.copy()
            cv2.drawContours(fill_layer, contours, -1, (255, 0, 0), thickness=cv2.FILLED)
            contour_overlay = cv2.addWeighted(contour_overlay, 0.7, fill_layer, 0.3, 0)
            # 윤곽선 테두리 (두께 4px)
            cv2.drawContours(contour_overlay, contours, -1, (255, 0, 0), thickness=4)
        Image.fromarray(contour_overlay).save(o_path)

        # [BACKUP] bbox 기반 overlay (필요 시 복구)
        # bbox_overlay = input_img.copy()
        # bbox = loc.get("bbox")
        # if isinstance(bbox, list) and len(bbox) == 4:
        #     x0, y0, x1, y1 = [int(v) for v in bbox]
        #     x0 = max(0, min(w - 1, x0))
        #     x1 = max(0, min(w - 1, x1))
        #     y0 = max(0, min(h - 1, y0))
        #     y1 = max(0, min(h - 1, y1))
        #     if x1 > x0 and y1 > y0:
        #         fill = bbox_overlay.copy()
        #         cv2.rectangle(fill, (x0, y0), (x1, y1), (255, 0, 0), thickness=-1)
        #         bbox_overlay = cv2.addWeighted(bbox_overlay, 0.88, fill, 0.12, 0)
        #         cv2.rectangle(bbox_overlay, (x0, y0), (x1, y1), (255, 0, 0), thickness=2)
        #         cv2.putText(
        #             bbox_overlay,
        #             "AD BBox",
        #             (x0, max(18, y0 - 6)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             (255, 0, 0),
        #             1,
        #             cv2.LINE_AA,
        #         )
        # Image.fromarray(bbox_overlay).save(o_path)
        return loc
