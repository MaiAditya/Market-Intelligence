"""
Dependency Classifier Module

Classifies documents by which event dependencies they affect.
Uses zero-shot approach with sentence similarity.

Dependencies:
- training
- compute  
- safety
- regulation
- executive_statement
- public_narrative
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from .model_manager import get_model_manager
from .semantic_relevance import SemanticRelevanceScorer

logger = logging.getLogger(__name__)


# Dependency descriptions for zero-shot classification
DEPENDENCY_DESCRIPTIONS = {
    "training": (
        "Information about model training progress, training completion, "
        "dataset preparation, training infrastructure, compute usage for training, "
        "training milestones, model convergence, and pre-training or fine-tuning status."
    ),
    "compute": (
        "Information about computational resources, GPU availability, TPU clusters, "
        "data center infrastructure, cloud computing capacity, hardware procurement, "
        "chip supply, and computational bottlenecks."
    ),
    "safety": (
        "Information about AI safety testing, alignment research, red teaming, "
        "safety evaluations, risk assessments, responsible AI practices, "
        "jailbreak resistance, and potential harms or misuse concerns."
    ),
    "regulation": (
        "Information about regulatory requirements, government oversight, "
        "legal compliance, policy frameworks, legislation, AI governance, "
        "regulatory approval, and legal challenges or restrictions."
    ),
    "executive_statement": (
        "Official statements, announcements, or communications from company executives, "
        "leadership, CEOs, CTOs, press releases, official blog posts, "
        "and authoritative company communications."
    ),
    "public_narrative": (
        "Public perception, media coverage, social media sentiment, "
        "community reactions, public discourse, hype, criticism, "
        "and general public opinion about the event."
    )
}


class DependencyClassifier:
    """
    Multi-label dependency classifier using zero-shot approach.
    
    Uses sentence similarity between document content and
    dependency descriptions to classify which dependencies
    a document affects.
    """
    
    def __init__(self):
        """Initialize dependency classifier."""
        self.model_manager = get_model_manager()
        self.relevance_scorer = SemanticRelevanceScorer()
        self.threshold = self.model_manager.get_threshold("dependency_threshold")
        
        # Cache dependency embeddings
        self._dependency_embeddings: Optional[np.ndarray] = None
        self._dependency_order: List[str] = list(DEPENDENCY_DESCRIPTIONS.keys())
    
    def _get_dependency_embeddings(self) -> Optional[np.ndarray]:
        """Get or compute dependency description embeddings."""
        if self._dependency_embeddings is not None:
            return self._dependency_embeddings
        
        descriptions = [
            DEPENDENCY_DESCRIPTIONS[dep]
            for dep in self._dependency_order
        ]
        
        self._dependency_embeddings = self.relevance_scorer.encode(descriptions)
        return self._dependency_embeddings
    
    def classify(self, document_text: str) -> Dict[str, float]:
        """
        Classify document by dependencies.
        
        Args:
            document_text: Document content text
        
        Returns:
            Dictionary mapping dependency names to scores (0.0-1.0)
        """
        dep_embeddings = self._get_dependency_embeddings()
        
        if dep_embeddings is None:
            # Fallback to keyword-based classification
            return self._keyword_classify(document_text)
        
        # Encode document
        doc_embedding = self.relevance_scorer.encode([document_text])
        
        if doc_embedding is None:
            return self._keyword_classify(document_text)
        
        doc_embedding = doc_embedding[0]
        
        # Compute similarity to each dependency
        scores = {}
        for i, dep_name in enumerate(self._dependency_order):
            similarity = self.relevance_scorer.cosine_similarity(
                doc_embedding,
                dep_embeddings[i]
            )
            # Normalize to 0-1 range (cosine can be negative)
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[dep_name] = round(normalized, 3)
        
        return scores
    
    def _keyword_classify(self, text: str) -> Dict[str, float]:
        """
        Fallback keyword-based classification.
        
        Uses keyword presence to estimate dependency relevance.
        """
        text_lower = text.lower()
        
        keyword_map = {
            "training": [
                "training", "trained", "pre-training", "fine-tuning", "fine-tune",
                "dataset", "epochs", "convergence", "loss", "gradient"
            ],
            "compute": [
                "gpu", "tpu", "compute", "infrastructure", "hardware", "chip",
                "data center", "cluster", "nvidia", "h100", "a100", "cloud"
            ],
            "safety": [
                "safety", "alignment", "red team", "risk", "harm", "misuse",
                "jailbreak", "responsible", "ethical", "dangerous", "threat"
            ],
            "regulation": [
                "regulation", "regulatory", "compliance", "law", "legislation",
                "government", "policy", "legal", "oversight", "eu ai act", "ban"
            ],
            "executive_statement": [
                "announced", "announcement", "statement", "ceo", "executive",
                "press release", "blog post", "official", "confirms", "revealed"
            ],
            "public_narrative": [
                "twitter", "reddit", "reaction", "opinion", "sentiment",
                "hype", "criticism", "community", "public", "viral", "trending"
            ]
        }
        
        scores = {}
        for dep_name, keywords in keyword_map.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Normalize: more matches = higher score, max at ~1.0
            score = min(1.0, matches / 3)
            scores[dep_name] = round(score, 3)
        
        return scores
    
    def get_top_dependencies(
        self,
        document_text: str,
        min_score: Optional[float] = None
    ) -> List[tuple]:
        """
        Get top dependencies for a document.
        
        Args:
            document_text: Document text
            min_score: Minimum score threshold
        
        Returns:
            List of (dependency_name, score) tuples, sorted by score
        """
        min_score = min_score or self.threshold
        scores = self.classify(document_text)
        
        # Filter and sort
        filtered = [
            (name, score)
            for name, score in scores.items()
            if score >= min_score
        ]
        
        return sorted(filtered, key=lambda x: x[1], reverse=True)
    
    def classify_batch(
        self,
        documents: List[str]
    ) -> List[Dict[str, float]]:
        """
        Classify multiple documents.
        
        Args:
            documents: List of document texts
        
        Returns:
            List of dependency score dictionaries
        """
        return [self.classify(doc) for doc in documents]
    
    def classify_batch_optimized(
        self,
        document_embeddings: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Optimized batch classification using pre-computed document embeddings.
        
        This method is designed to work with embeddings from score_batch_optimized()
        to avoid redundant encoding.
        
        Args:
            document_embeddings: Pre-computed document embeddings (N x embedding_dim)
        
        Returns:
            List of dependency score dictionaries
        """
        dep_embeddings = self._get_dependency_embeddings()
        
        if dep_embeddings is None or document_embeddings is None:
            logger.warning("Embeddings not available, cannot classify batch")
            return [{dep: 0.0 for dep in self._dependency_order} for _ in range(len(document_embeddings))]
        
        results = []
        
        # Process each document embedding
        for doc_embedding in document_embeddings:
            scores = {}
            for i, dep_name in enumerate(self._dependency_order):
                similarity = self.relevance_scorer.cosine_similarity(
                    doc_embedding,
                    dep_embeddings[i]
                )
                # Normalize to 0-1 range (cosine can be negative)
                normalized = max(0.0, min(1.0, (similarity + 1) / 2))
                scores[dep_name] = round(normalized, 3)
            
            results.append(scores)
        
        return results

