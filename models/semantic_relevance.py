"""
Semantic Relevance Module

Uses sentence-transformers for semantic similarity between
documents and events.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model_manager import get_model_manager

logger = logging.getLogger(__name__)


class SemanticRelevanceScorer:
    """
    Scores document relevance to events using semantic similarity.
    
    Uses sentence-transformers/all-mpnet-base-v2 for embeddings
    and cosine similarity for scoring.
    """
    
    def __init__(self):
        """Initialize semantic relevance scorer."""
        self.model_manager = get_model_manager()
        self._model = None
        self.threshold = self.model_manager.get_threshold("semantic_relevance")
        self.max_length = self.model_manager.get_max_sequence_length()
    
    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            self._model = self.model_manager.get_sentence_transformer()
        return self._model
    
    def _truncate_text(self, text: str, max_chars: int = 2048) -> str:
        """Truncate text to fit model constraints."""
        if len(text) <= max_chars:
            return text
        # Truncate but try to keep sentence boundaries
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars // 2:
            return truncated[:last_period + 1]
        return truncated
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            Numpy array of embeddings or None
        """
        model = self._get_model()
        if model is None:
            logger.warning("Sentence transformer not available")
            return None
        
        try:
            # Truncate texts
            truncated = [self._truncate_text(t) for t in texts]
            
            # Encode
            embeddings = model.encode(
                truncated,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None
    
    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def score_relevance(
        self,
        event_description: str,
        document_text: str
    ) -> Tuple[float, bool]:
        """
        Score document relevance to an event.
        
        Args:
            event_description: Event description text
            document_text: Document content text
        
        Returns:
            Tuple of (relevance_score, passed_threshold)
        """
        embeddings = self.encode([event_description, document_text])
        
        if embeddings is None:
            # Fallback: keyword matching
            return self._keyword_fallback(event_description, document_text)
        
        similarity = self.cosine_similarity(embeddings[0], embeddings[1])
        passed = similarity >= self.threshold
        
        return similarity, passed
    
    def _keyword_fallback(
        self,
        event_description: str,
        document_text: str
    ) -> Tuple[float, bool]:
        """
        Fallback keyword-based relevance when model unavailable.
        
        Uses simple word overlap as a proxy for similarity.
        """
        # Tokenize and normalize
        event_words = set(event_description.lower().split())
        doc_words = set(document_text.lower().split())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'that', 'this', 'these', 'those', 'it', 'its', 'as', 'if'
        }
        
        event_words -= stop_words
        doc_words -= stop_words
        
        if not event_words:
            return 0.0, False
        
        # Calculate Jaccard-like similarity
        intersection = event_words & doc_words
        union = event_words | doc_words
        
        if not union:
            return 0.0, False
        
        similarity = len(intersection) / len(event_words)
        passed = similarity >= 0.3  # Lower threshold for keyword matching
        
        return similarity, passed
    
    def score_batch(
        self,
        event_description: str,
        documents: List[str]
    ) -> List[Tuple[float, bool]]:
        """
        Score multiple documents against an event.
        
        Args:
            event_description: Event description
            documents: List of document texts
        
        Returns:
            List of (score, passed) tuples
        """
        if not documents:
            return []
        
        # Encode all at once for efficiency
        all_texts = [event_description] + documents
        embeddings = self.encode(all_texts)
        
        if embeddings is None:
            # Fallback for each document
            return [
                self._keyword_fallback(event_description, doc)
                for doc in documents
            ]
        
        event_embedding = embeddings[0]
        results = []
        
        for i, doc_embedding in enumerate(embeddings[1:]):
            similarity = self.cosine_similarity(event_embedding, doc_embedding)
            passed = similarity >= self.threshold
            results.append((similarity, passed))
        
        return results
    
    def find_most_relevant(
        self,
        event_description: str,
        documents: List[Dict],
        top_k: int = 10,
        text_key: str = "raw_text"
    ) -> List[Dict]:
        """
        Find most relevant documents for an event.
        
        Args:
            event_description: Event description
            documents: List of document dicts
            top_k: Number of top documents to return
            text_key: Key for text content in document dict
        
        Returns:
            Top k documents with relevance scores added
        """
        if not documents:
            return []
        
        texts = [doc.get(text_key, "") for doc in documents]
        scores = self.score_batch(event_description, texts)
        
        # Add scores to documents
        for doc, (score, passed) in zip(documents, scores):
            doc["relevance_score"] = score
            doc["relevance_passed"] = passed
        
        # Sort by score
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        
        return sorted_docs[:top_k]
    
    def score_batch_optimized(
        self,
        event_description: str,
        documents: List[str],
        cache_event_embedding: bool = True
    ) -> Tuple[List[Tuple[float, bool]], Optional[np.ndarray]]:
        """
        Optimized batch scoring that returns both scores and document embeddings.
        
        This method is designed for maximum performance when you need both
        relevance scores AND document embeddings (e.g., for dependency classification).
        
        Args:
            event_description: Event description
            documents: List of document texts
            cache_event_embedding: If True, cache event embedding for reuse
        
        Returns:
            Tuple of (scores_list, document_embeddings)
            - scores_list: List of (score, passed) tuples
            - document_embeddings: Numpy array of document embeddings (or None if fallback)
        """
        if not documents:
            return [], None
        
        # Encode all at once for efficiency
        all_texts = [event_description] + documents
        embeddings = self.encode(all_texts)
        
        if embeddings is None:
            # Fallback for each document
            scores = [
                self._keyword_fallback(event_description, doc)
                for doc in documents
            ]
            return scores, None
        
        event_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Compute all similarities at once
        results = []
        for doc_embedding in doc_embeddings:
            similarity = self.cosine_similarity(event_embedding, doc_embedding)
            passed = similarity >= self.threshold
            results.append((similarity, passed))
        
        return results, doc_embeddings
