from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = False

# This order must exactly match the order used while training the checkpoint.
LABELS = ("AD", "SD", "CD")
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"


@dataclass(frozen=True)
class Prediction:
    predicted_class: str | None
    confidence: float
    probabilities: dict[str, float]
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "status": self.status,
            "message": self.message,
            "is_retrieval_hint": self.status == "usable_as_retrieval_hint",
        }


class SkinClassifier:
    """Load and serve the trained ConvNeXtV2-Tiny AD/SD/CD checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str | None = None,
        confidence_threshold: float = 0.80,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.labels = LABELS
        self.model = self._build_model()
        self.model.eval().to(self.device)

        # Match the 384x384 input resolution used by the ConvNeXtV2 model name.
        self.transform = transforms.Compose(
            [
                transforms.Resize(384),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _build_model(self) -> nn.Module:
        model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=len(self.labels),
        )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint),
        ) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a readable state dictionary")

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized_key = str(key)
            for prefix in ("module.", "model.", "net."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
            cleaned[normalized_key] = value

        try:
            model.load_state_dict(cleaned, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "The checkpoint does not match ConvNeXtV2-Tiny with three outputs "
                "in the AD/SD/CD order. Verify MODEL_NAME, class order, and checkpoint format."
            ) from exc
        return model

    def predict(self, image_bytes: bytes) -> Prediction:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as exc:
            raise ValueError("The uploaded file is not a valid readable image") from exc

        with torch.inference_mode():
            probabilities_tensor = torch.softmax(self.model(tensor), dim=1)[0].cpu()

        probabilities = {
            label: float(probabilities_tensor[index])
            for index, label in enumerate(self.labels)
        }
        best_index = int(torch.argmax(probabilities_tensor))
        confidence = float(probabilities_tensor[best_index])
        predicted_class = self.labels[best_index]

        if confidence >= self.confidence_threshold:
            status = "usable_as_retrieval_hint"
            message = "Prediction may be used as a soft retrieval hint; it is not a diagnosis."
        else:
            status = "uncertain"
            predicted_class = None
            message = "Confidence is below the configured threshold; do not use this prediction to filter evidence."

        return Prediction(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            status=status,
            message=message,
        )
