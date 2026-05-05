"""
Transformer-based emotion classification for user utterances (Phase 5).

Uses a small English DistilRoBERTa model fine-tuned on emotion labels.
This is **not** scripture classification — it only labels the user's message
to steer tone and to populate the structured ``emotion`` field honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Go-to compact model: 7-way English emotions, strong for short social text.
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


@dataclass(frozen=True)
class EmotionPrediction:
    """Argmax label plus full softmax distribution (for debugging / UI)."""

    label: str
    confidence: float
    scores: dict[str, float]


class EmotionClassifier:
    """
    Hugging Face sequence classification wrapper.

    Loads tokenizer + model once; ``predict`` is deterministic (argmax on logits).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMOTION_MODEL,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_name: HF hub id or local path with ``id2label`` in config.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (pick CUDA if available).
        """
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @torch.inference_mode()
    def predict(self, text: str) -> EmotionPrediction:
        """
        Return the highest-probability emotion label for ``text``.

        Empty / whitespace input yields ``neutral`` with zero confidence (no inference).
        """
        t = text.strip()
        if not t:
            return EmotionPrediction(label="neutral", confidence=0.0, scores={})

        enc = self._tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        logits = self._model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        id2label = self._model.config.id2label
        label = str(id2label[int(idx.item())])
        scores = {
            str(id2label[i]): float(probs[i].item()) for i in range(len(id2label))
        }
        return EmotionPrediction(
            label=label,
            confidence=float(conf.item()),
            scores=scores,
        )
