"""
Duck Counting & Detection Package
Modular dataset extraction, YOLO model training, and density counting pipelines.
"""

__version__ = "1.0.0"

from src.models.counter import DuckCounter, DuckCountResult
from src.models.trainer import train_duck_model
from src.models.evaluator import evaluate_duck_model
from src.data.objects365 import Objects365DuckExtractor
from src.data.imagenet import ImageNetDuckExtractor

__all__ = [
    "DuckCounter",
    "DuckCountResult",
    "train_duck_model",
    "evaluate_duck_model",
    "Objects365DuckExtractor",
    "ImageNetDuckExtractor",
]
