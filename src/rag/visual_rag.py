from __future__ import annotations

import logging
import pickle
import warnings
import numpy as np
import torch
import torchvision.transforms as T

from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from src.datasets.dataloader import iter_image_files
from src.utils.device import get_device

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# train dataset root to good
TRAIN_IMAGE_ROOTS: Dict[str, str] = {
    "GoodsAD":    "train/good",
    "MVTec-LOCO": "train/good",
}


class VisualRAG:
    def __init__(self, index_dir: str | Path, model_name: str = 'dinov2_vits14',):
        self.index_dir = Path(index_dir)
        self.device = get_device()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(256), # 518
            T.CenterCrop(224), # 518
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.index_cache: Dict[str, Dict[str, Any]] = {}

    # Embedding
    def extract_embedding(self, image_path: str | Path) -> np.ndarray:
        """이미지 파일 → DINOv2 feature vector.

        Returns:
            np.ndarray of shape (1, D)
        """
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_t)
        return embedding.cpu().numpy()

    # Index build
    def build_index(self, category: str, image_paths: List[str | Path]) -> None:
        """image_paths의 임베딩을 추출해 {category}.pkl로 저장.

        Args:
            category: 카테고리 이름 (예: "carpet", "drink_bottle").
            image_paths: 정상 이미지 절대 경로 리스트.
        """
        features, paths = [], []
        for p in image_paths:
            try:
                features.append(self.extract_embedding(p))
                paths.append(str(p))
            except Exception as e:
                logger.warning("Skipping %s: %s", p, e)

        if not features:
            logger.warning("No images processed for category '%s'. Skipping.", category)
            return

        data = {
            "features": np.vstack(features),  # (N, D)
            "paths": paths,
        }
        self.index_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = self.index_dir / f"{category.replace('/', '_')}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("Built visual index for '%s': %d images → %s", category, len(paths), pkl_path)

    def build_all(self, dataset_root: str | Path, datasets: Optional[List[str]] = None,) -> None:
        """전체 dataset을 스캔해 카테고리별 인덱스를 빌드.

        Args:
            dataset_root: MMAD 루트 디렉토리 (dataset/MMAD).
            datasets: 빌드할 데이터셋 이름 리스트. None이면 전체.
        """
        root = Path(dataset_root)
        targets = datasets or list(TRAIN_IMAGE_ROOTS.keys())

        for dataset_name in targets:
            train_subdir = TRAIN_IMAGE_ROOTS[dataset_name]
            dataset_dir = root / dataset_name

            if not dataset_dir.exists():
                logger.warning("Dataset dir not found: %s", dataset_dir)
                continue

            for category_dir in sorted(dataset_dir.iterdir()):
                if not category_dir.is_dir():
                    continue

                train_dir = category_dir / train_subdir
                image_paths = list(iter_image_files(train_dir))
                if not image_paths:
                    logger.warning("No images in %s", train_dir)
                    continue

                logger.info(
                    "[%s] Building index for '%s' (%d images)",
                    dataset_name, category_dir.name, len(image_paths),
                )
                self.build_index(category_dir.name, image_paths)

    # Index load
    def load_index(self, category: str) -> Dict[str, Any]:
        """카테고리 인덱스를 로드하고 캐싱.

        Returns:
            {"features": np.ndarray, "paths": List[str]}
        """
        key = category.replace('/', '_')
        if key in self.index_cache:
            return self.index_cache[key]

        pkl_path = self.index_dir / f"{key}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Visual index not found for category '{category}': {pkl_path}\n"
                "Run build_all() or build_index() to create the index first."
            )

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.index_cache[key] = data
        logger.info("Loaded visual index for '%s': %d images", category, len(data["paths"]))
        return data

    # Search
    def search(self, query_image_path: str | Path, category: str, k: int = 1,) -> List[Dict[str, Any]]:
        """쿼리 이미지와 가장 유사한 정상 이미지를 반환.

        Args:
            query_image_path: 검색 대상 이미지 경로.
            category: 인덱스를 조회할 카테고리 이름.
            k: 반환할 결과 수.

        Returns:
            [{"path": str, "score": float}, ...] (유사도 내림차순)
        """
        query_feat = self.extract_embedding(query_image_path)
        idx_data = self.load_index(category)

        similarities = cosine_similarity(query_feat, idx_data["features"])[0]
        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            {"path": idx_data["paths"][i], "score": float(similarities[i])}
            for i in top_indices
        ]
