"""
Models Module

BERT-style models for NLP tasks:
- NER: Entity extraction (dslim/bert-base-NER)
- Semantic Relevance: Document-event similarity (all-mpnet-base-v2)
- Dependency Classifier: Multi-label classification
- Signal Classifier: Type and direction classification
"""

from .model_manager import ModelManager, get_model_manager
from .ner import NERExtractor, ExtractedEntity
from .semantic_relevance import SemanticRelevanceScorer
from .dependency_classifier import DependencyClassifier
from .signal_classifier import SignalClassifier

__all__ = [
    "ModelManager",
    "get_model_manager",
    "NERExtractor",
    "ExtractedEntity",
    "SemanticRelevanceScorer",
    "DependencyClassifier",
    "SignalClassifier"
]
