from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CnnPrediction:
    defect_type: str
    defect_confidence: float
    grade: str
    grade_confidence: float


class CnnModelStatus:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    @property
    def exists(self) -> bool:
        return self.model_path.exists()

    @property
    def size_bytes(self) -> int:
        if not self.exists:
            return 0
        return self.model_path.stat().st_size

    @property
    def is_loadable_placeholder(self) -> bool:
        return self.exists and self.size_bytes > 0


class RelayGradeCnn:
    def __init__(self, model_path: Path, metadata_path: Path | None = None) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path or model_path.with_name(
            f"{model_path.stem}.metadata.json"
        )
        self._model: Any | None = None
        self._metadata: dict[str, Any] | None = None
        self._device: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            if self.metadata_path.exists():
                self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            else:
                self._metadata = {}
        return self._metadata

    @property
    def input_size(self) -> int:
        return int(self.metadata.get("input_size", 224))

    @property
    def defect_labels(self) -> list[str]:
        return list(self.metadata.get("defect_labels", ["other"]))

    @property
    def grade_labels(self) -> list[str]:
        return list(self.metadata.get("grade_labels", ["A+", "A", "B+", "B", "C", "D"]))

    def predict(self, image: Any) -> CnnPrediction:
        torch = _import_torch()
        model = self._load_model()
        tensor = self._preprocess(image).to(self._device)

        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            defect_probs = torch.softmax(outputs["defect_logits"], dim=1)[0]
            grade_probs = torch.softmax(outputs["grade_logits"], dim=1)[0]

        defect_id = int(torch.argmax(defect_probs).item())
        grade_id = int(torch.argmax(grade_probs).item())
        return CnnPrediction(
            defect_type=self.defect_labels[defect_id],
            defect_confidence=float(defect_probs[defect_id].item()),
            grade=self.grade_labels[grade_id],
            grade_confidence=float(grade_probs[grade_id].item()),
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            raise RuntimeError(f"CNN checkpoint is missing or empty: {self.model_path}")

        torch = _import_torch()
        checkpoint = torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        architecture = checkpoint.get(
            "architecture", self.metadata.get("architecture", "mobilenetv3_small_100")
        )

        model = _RelayGradeModule(
            architecture=architecture,
            num_defects=len(self.defect_labels),
            num_grades=len(self.grade_labels),
            input_size=self.input_size,
        )
        model.load_state_dict(state_dict)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.to(self._device)
        return self._model

    def _preprocess(self, image: Any) -> Any:
        torch = _import_torch()
        size = self.input_size
        image = image.convert("RGB").resize((size, size))
        pixels = list(image.getdata())

        tensor = torch.tensor(pixels, dtype=torch.float32).view(size, size, 3)
        tensor = tensor.permute(2, 0, 1).div(255.0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor.sub(mean).div(std).unsqueeze(0)


class _RelayGradeModule:
    def __new__(
        cls,
        architecture: str,
        num_defects: int,
        num_grades: int,
        input_size: int,
    ) -> Any:
        torch = _import_torch()
        timm = _import_timm()

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = timm.create_model(
                    architecture,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, input_size, input_size)
                    features = self.backbone(dummy).shape[1]
                self.defect_head = torch.nn.Linear(features, num_defects)
                self.grade_head = torch.nn.Linear(features, num_grades)

            def forward(self, x: Any) -> dict[str, Any]:
                features = self.backbone(x)
                return {
                    "defect_logits": self.defect_head(features),
                    "grade_logits": self.grade_head(features),
                }

        return Module()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CNN grading disabled — relay-ml runs bedrock_only and ships without "
            "torch. Set GRADING_MODE=cnn and install torch/timm to use the local CNN."
        ) from exc
    return torch


def _import_timm() -> Any:
    _define_torchvision_nms_stub()
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "CNN grading disabled — relay-ml runs bedrock_only and ships without "
            "timm. Set GRADING_MODE=cnn and install torch/timm to use the local CNN."
        ) from exc
    return timm


def _define_torchvision_nms_stub() -> None:
    torch = _import_torch()
    try:
        torch.library.Library("torchvision", "DEF").define(
            "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
        )
    except Exception:
        pass
