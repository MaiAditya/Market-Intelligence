"""
Signal Classifier Module

Classifies documents by signal type and direction using zero-shot approach.

Signal Types:
- training_progress
- delay
- rumor
- official_confirmation
- narrative_shift
- regulation_update
- executive_statement

Directions:
- positive
- negative
- neutral
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model_manager import get_model_manager
from .semantic_relevance import SemanticRelevanceScorer

logger = logging.getLogger(__name__)


# Signal type descriptions for zero-shot classification
SIGNAL_TYPE_DESCRIPTIONS = {
    "training_progress": (
        "Report of progress in AI model training, completion of training phases, "
        "training milestones achieved, successful model convergence, or training updates."
    ),
    "delay": (
        "Information about delays, postponements, setbacks, or extended timelines "
        "for AI model releases, product launches, or project completion."
    ),
    "rumor": (
        "Unconfirmed information, speculation, leaks from unnamed sources, "
        "insider reports without official confirmation, or hearsay about AI developments."
    ),
    "official_confirmation": (
        "Official announcement, press release, confirmed statement from company leadership, "
        "or authoritative confirmation of AI-related events or milestones."
    ),
    "narrative_shift": (
        "Change in public perception, shift in media coverage tone, "
        "emerging new perspective, or significant change in discourse about AI."
    ),
    "regulation_update": (
        "Update on AI regulations, new legislation, policy changes, "
        "compliance requirements, or government oversight announcements."
    ),
    "executive_statement": (
        "Statement from a CEO, CTO, or senior executive about AI strategy, "
        "product direction, company plans, or official company position."
    )
}

# Direction descriptions
DIRECTION_DESCRIPTIONS = {
    "positive": (
        "Good news, progress, success, achievement, advancement, improvement, "
        "optimistic outlook, beneficial development, or favorable outcome."
    ),
    "negative": (
        "Bad news, setback, failure, problem, concern, delay, risk, "
        "pessimistic outlook, harmful development, or unfavorable outcome."
    ),
    "neutral": (
        "Factual report without clear positive or negative framing, "
        "balanced coverage, informational update, or objective description."
    )
}


class SignalTypeClassifier:
    """
    Classifies documents by signal type using zero-shot approach.
    """
    
    def __init__(self):
        """Initialize signal type classifier."""
        self.relevance_scorer = SemanticRelevanceScorer()
        self._type_embeddings: Optional[np.ndarray] = None
        self._type_order: List[str] = list(SIGNAL_TYPE_DESCRIPTIONS.keys())
    
    def _get_type_embeddings(self) -> Optional[np.ndarray]:
        """Get or compute signal type description embeddings."""
        if self._type_embeddings is not None:
            return self._type_embeddings
        
        descriptions = [
            SIGNAL_TYPE_DESCRIPTIONS[t]
            for t in self._type_order
        ]
        
        self._type_embeddings = self.relevance_scorer.encode(descriptions)
        return self._type_embeddings
    
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify document by signal type.
        
        Args:
            text: Document text
        
        Returns:
            Dictionary mapping signal type to confidence score
        """
        type_embeddings = self._get_type_embeddings()
        
        if type_embeddings is None:
            return self._keyword_classify(text)
        
        doc_embedding = self.relevance_scorer.encode([text])
        if doc_embedding is None:
            return self._keyword_classify(text)
        
        doc_embedding = doc_embedding[0]
        
        scores = {}
        for i, signal_type in enumerate(self._type_order):
            similarity = self.relevance_scorer.cosine_similarity(
                doc_embedding,
                type_embeddings[i]
            )
            # Normalize to 0-1
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[signal_type] = round(normalized, 3)
        
        return scores
    
    def _keyword_classify(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based classification."""
        text_lower = text.lower()
        
        keyword_map = {
            "training_progress": [
                "training complete", "finished training", "training milestone",
                "converged", "pre-training done", "model trained"
            ],
            "delay": [
                "delay", "postpone", "setback", "pushed back", "later than",
                "not ready", "extended timeline"
            ],
            "rumor": [
                "rumor", "leak", "sources say", "reportedly", "unconfirmed",
                "insider", "speculation", "allegedly"
            ],
            "official_confirmation": [
                "announced", "confirms", "official", "press release",
                "we are pleased", "today we"
            ],
            "narrative_shift": [
                "shift", "change in", "new perspective", "rethinking",
                "pivoting", "reconsidering"
            ],
            "regulation_update": [
                "regulation", "law", "policy", "compliance", "legislation",
                "government", "enforcement", "legal"
            ],
            "executive_statement": [
                "ceo said", "cto said", "executive", "leadership",
                "statement from", "according to the ceo"
            ]
        }
        
        scores = {}
        for signal_type, keywords in keyword_map.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = min(1.0, matches / 2)
            scores[signal_type] = round(score, 3)
        
        return scores
    
    def classify_from_embedding(self, doc_embedding) -> Dict[str, float]:
        """
        Classify signal type from a pre-computed document embedding.
        Avoids re-encoding the document text.
        """
        type_embeddings = self._get_type_embeddings()
        if type_embeddings is None:
            return {}
        
        scores = {}
        for i, signal_type in enumerate(self._type_order):
            similarity = self.relevance_scorer.cosine_similarity(
                doc_embedding, type_embeddings[i]
            )
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[signal_type] = round(normalized, 3)
        return scores

    def get_best_type(self, text: str) -> Tuple[str, float]:
        """Get the best matching signal type."""
        scores = self.classify(text)
        if not scores:
            return "rumor", 0.5  # Default
        
        best_type = max(scores, key=scores.get)
        return best_type, scores[best_type]

    def get_best_type_from_embedding(self, doc_embedding) -> Tuple[str, float]:
        """Get the best matching signal type from a pre-computed embedding."""
        scores = self.classify_from_embedding(doc_embedding)
        if not scores:
            return "rumor", 0.5
        best_type = max(scores, key=scores.get)
        return best_type, scores[best_type]


class DirectionClassifier:
    """
    Classifies document direction (positive/negative/neutral).
    
    Uses semantic similarity with pre-defined direction descriptions.
    Falls back to keyword matching if model is unavailable.
    """
    
    def __init__(self):
        """Initialize direction classifier."""
        self.relevance_scorer = SemanticRelevanceScorer()
        self._direction_embeddings: Optional[np.ndarray] = None
        self._direction_order: List[str] = ["positive", "negative", "neutral"]
        self._model_healthy: bool = False
        self._classification_count: int = 0
        self._fallback_count: int = 0
        
        # Run health check on initialization
        self._check_model_health()
    
    def _check_model_health(self) -> bool:
        """
        Check if the semantic model is working properly.
        
        Runs a simple test classification to verify model output.
        
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            # Test with clearly positive and negative texts
            positive_test = "This is great news! Amazing success and breakthrough achievement."
            negative_test = "This is terrible news. Major failure and serious problems ahead."
            
            pos_embedding = self.relevance_scorer.encode([positive_test])
            neg_embedding = self.relevance_scorer.encode([negative_test])
            
            if pos_embedding is None or neg_embedding is None:
                logger.warning("DirectionClassifier: Model failed to encode test texts")
                self._model_healthy = False
                return False
            
            # Verify embeddings have reasonable shapes
            if pos_embedding.shape[-1] < 100:  # Expect at least 100-dim embeddings
                logger.warning(f"DirectionClassifier: Unexpected embedding dimension: {pos_embedding.shape}")
                self._model_healthy = False
                return False
            
            self._model_healthy = True
            logger.info("DirectionClassifier: Model health check passed")
            return True
            
        except Exception as e:
            logger.error(f"DirectionClassifier: Model health check failed: {e}")
            self._model_healthy = False
            return False
    
    def _get_direction_embeddings(self) -> Optional[np.ndarray]:
        """Get or compute direction description embeddings."""
        if self._direction_embeddings is not None:
            return self._direction_embeddings
        
        descriptions = [
            DIRECTION_DESCRIPTIONS[d]
            for d in self._direction_order
        ]
        
        self._direction_embeddings = self.relevance_scorer.encode(descriptions)
        
        if self._direction_embeddings is not None:
            logger.debug(f"Direction embeddings computed: shape {self._direction_embeddings.shape}")
        else:
            logger.warning("Failed to compute direction embeddings")
            
        return self._direction_embeddings
    
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify document direction using semantic similarity.
        
        Args:
            text: Document text
        
        Returns:
            Dictionary mapping direction to confidence score
        """
        self._classification_count += 1
        
        dir_embeddings = self._get_direction_embeddings()
        
        if dir_embeddings is None:
            self._fallback_count += 1
            logger.debug(f"Direction classification fallback to keywords (model unavailable)")
            return self._keyword_classify(text)
        
        doc_embedding = self.relevance_scorer.encode([text])
        if doc_embedding is None:
            self._fallback_count += 1
            logger.debug(f"Direction classification fallback to keywords (encoding failed)")
            return self._keyword_classify(text)
        
        doc_embedding = doc_embedding[0]
        
        scores = {}
        for i, direction in enumerate(self._direction_order):
            similarity = self.relevance_scorer.cosine_similarity(
                doc_embedding,
                dir_embeddings[i]
            )
            # Normalize to 0-1
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[direction] = round(normalized, 3)
        
        # Log classification quality periodically
        if self._classification_count % 100 == 0:
            fallback_rate = (self._fallback_count / self._classification_count) * 100
            logger.info(
                f"DirectionClassifier stats: {self._classification_count} classifications, "
                f"{fallback_rate:.1f}% fallback rate"
            )
        
        return scores
    
    def _keyword_classify(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based classification."""
        text_lower = text.lower()
        
        positive_keywords = [
            "success", "achievement", "progress", "breakthrough", "good news",
            "impressive", "excellent", "advance", "milestone", "exciting",
            "promising", "optimistic", "positive", "improvement", "ahead of schedule"
        ]
        
        negative_keywords = [
            "failure", "problem", "concern", "delay", "setback", "risk",
            "disappointing", "issue", "challenge", "criticism", "worry",
            "pessimistic", "negative", "behind schedule", "trouble"
        ]
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        # Simple scoring
        if positive_count > negative_count:
            return {
                "positive": min(0.8, 0.5 + positive_count * 0.1),
                "negative": max(0.1, 0.3 - positive_count * 0.05),
                "neutral": 0.3
            }
        elif negative_count > positive_count:
            return {
                "positive": max(0.1, 0.3 - negative_count * 0.05),
                "negative": min(0.8, 0.5 + negative_count * 0.1),
                "neutral": 0.3
            }
        else:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.5}
    
    def classify_from_embedding(self, doc_embedding) -> Dict[str, float]:
        """
        Classify direction from a pre-computed document embedding.
        Avoids re-encoding the document text.
        """
        self._classification_count += 1
        dir_embeddings = self._get_direction_embeddings()
        if dir_embeddings is None:
            self._fallback_count += 1
            return {}
        
        scores = {}
        for i, direction in enumerate(self._direction_order):
            similarity = self.relevance_scorer.cosine_similarity(
                doc_embedding, dir_embeddings[i]
            )
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[direction] = round(normalized, 3)
        return scores

    def get_direction(self, text: str) -> Tuple[str, float]:
        """Get the dominant direction."""
        scores = self.classify(text)
        if not scores:
            return "neutral", 0.5
        
        direction = max(scores, key=scores.get)
        return direction, scores[direction]

    def get_direction_from_embedding(self, doc_embedding) -> Tuple[str, float]:
        """Get the dominant direction from a pre-computed embedding."""
        scores = self.classify_from_embedding(doc_embedding)
        if not scores:
            return "neutral", 0.5
        best_direction = max(scores, key=scores.get)
        return best_direction, scores[best_direction]
    
    def get_health_status(self) -> Dict:
        """
        Get health and quality metrics for the classifier.
        
        Returns:
            Dictionary with health status and metrics
        """
        return {
            "model_healthy": self._model_healthy,
            "total_classifications": self._classification_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": (
                (self._fallback_count / self._classification_count * 100)
                if self._classification_count > 0 else 0
            ),
            "embeddings_cached": self._direction_embeddings is not None,
        }
    
    def validate_output(self, text: str) -> Dict:
        """
        Validate classification output for debugging.
        
        Args:
            text: Test text to classify
        
        Returns:
            Detailed classification results with validation info
        """
        scores = self.classify(text)
        direction, confidence = self.get_direction(text)
        
        # Check for concerning patterns
        warnings = []
        
        # All scores very similar = model may not be discriminating
        if scores:
            score_range = max(scores.values()) - min(scores.values())
            if score_range < 0.1:
                warnings.append("Low score differentiation - model may not be discriminating well")
        
        # Very low confidence on best match
        if confidence < 0.4:
            warnings.append("Low confidence on best match")
        
        # All scores near 0.5 = model may be defaulting
        if scores and all(0.45 < s < 0.55 for s in scores.values()):
            warnings.append("All scores near 0.5 - possible model issue")
        
        return {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "scores": scores,
            "direction": direction,
            "confidence": confidence,
            "used_fallback": not self._model_healthy,
            "warnings": warnings
        }


class SignalClassifier:
    """
    Combined signal classifier for type and direction.
    """
    
    def __init__(self):
        """Initialize combined classifier."""
        self.type_classifier = SignalTypeClassifier()
        self.direction_classifier = DirectionClassifier()
        self.model_manager = get_model_manager()
        self.min_confidence = self.model_manager.get_threshold("signal_confidence")
    
    def classify(self, text: str) -> Dict:
        """
        Classify document for both signal type and direction.
        
        Args:
            text: Document text
        
        Returns:
            Dictionary with type_scores, direction_scores, best_type, direction
        """
        type_scores = self.type_classifier.classify(text)
        direction_scores = self.direction_classifier.classify(text)
        
        best_type, type_confidence = self.type_classifier.get_best_type(text)
        direction, dir_confidence = self.direction_classifier.get_direction(text)
        
        return {
            "type_scores": type_scores,
            "direction_scores": direction_scores,
            "signal_type": best_type,
            "type_confidence": type_confidence,
            "direction": direction,
            "direction_confidence": dir_confidence
        }
