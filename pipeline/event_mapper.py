"""
Event-Document Mapping Pipeline

3-stage mapping process:
1. Hard Entity Gate - Filter by entity presence
2. Semantic Relevance - Score by embedding similarity
3. Dependency Classification - Identify affected dependencies
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import Event, EventRegistry, get_registry
from pipeline.normalizer import DocumentNormalizer, NormalizedDocument
from models.semantic_relevance import SemanticRelevanceScorer
from models.dependency_classifier import DependencyClassifier

logger = logging.getLogger(__name__)


@dataclass
class DocumentMapping:
    """
    Result of mapping a document to an event.
    
    Contains all 3 stages of mapping results.
    """
    doc_id: str
    event_id: str
    
    # Stage 1: Entity Gate
    entity_gate_passed: bool
    primary_entities_matched: List[str]
    secondary_entities_matched: List[str]
    aliases_matched: List[str]
    
    # Stage 2: Semantic Relevance
    relevance_score: float
    relevance_passed: bool
    
    # Stage 3: Dependency Classification
    dependency_scores: Dict[str, float]
    top_dependencies: List[str]
    
    # Overall
    is_relevant: bool
    mapped_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "event_id": self.event_id,
            "entity_gate_passed": self.entity_gate_passed,
            "primary_entities_matched": self.primary_entities_matched,
            "secondary_entities_matched": self.secondary_entities_matched,
            "aliases_matched": self.aliases_matched,
            "relevance_score": self.relevance_score,
            "relevance_passed": self.relevance_passed,
            "dependency_scores": self.dependency_scores,
            "top_dependencies": self.top_dependencies,
            "is_relevant": self.is_relevant,
            "mapped_at": self.mapped_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentMapping":
        """Create from dictionary."""
        mapped_at = _utc_now()
        if data.get("mapped_at"):
            mapped_at = datetime.fromisoformat(data["mapped_at"])
        
        return cls(
            doc_id=data["doc_id"],
            event_id=data["event_id"],
            entity_gate_passed=data["entity_gate_passed"],
            primary_entities_matched=data.get("primary_entities_matched", []),
            secondary_entities_matched=data.get("secondary_entities_matched", []),
            aliases_matched=data.get("aliases_matched", []),
            relevance_score=data["relevance_score"],
            relevance_passed=data["relevance_passed"],
            dependency_scores=data.get("dependency_scores", {}),
            top_dependencies=data.get("top_dependencies", []),
            is_relevant=data["is_relevant"],
            mapped_at=mapped_at
        )


class EntityGate:
    """
    Stage 1: Hard Entity Gate
    
    Document passes if:
    - At least 1 primary entity present (exact or alias match)
    - At least 1 secondary entity OR alias present
    """
    
    def check(
        self,
        doc: NormalizedDocument,
        event: Event
    ) -> Tuple[bool, List[str], List[str], List[str]]:
        """
        Check if document passes entity gate.
        
        Args:
            doc: Normalized document with extracted entities
            event: Event to check against
        
        Returns:
            Tuple of (passed, primary_matches, secondary_matches, alias_matches)
        """
        logger.debug(f"Entity gate check for doc {doc.doc_id} against event {event.event_id}")
        
        # Get document entities (case-insensitive)
        doc_entities = set()
        for entity in doc.extracted_entities:
            entity_text = entity.get("text", "").lower()
            doc_entities.add(entity_text)
            # Also add without spaces/hyphens for flexible matching
            doc_entities.add(entity_text.replace("-", "").replace(" ", ""))
        
        # Also check raw text for entity presence
        doc_text_lower = doc.raw_text.lower()
        
        # Check primary entities
        primary_matches = []
        for entity in event.primary_entities:
            entity_lower = entity.lower()
            if entity_lower in doc_entities or entity_lower in doc_text_lower:
                primary_matches.append(entity)
        
        # Check secondary entities
        secondary_matches = []
        for entity in event.secondary_entities:
            entity_lower = entity.lower()
            if entity_lower in doc_entities or entity_lower in doc_text_lower:
                secondary_matches.append(entity)
        
        # Check aliases
        alias_matches = []
        for alias in event.aliases:
            alias_lower = alias.lower()
            if alias_lower in doc_entities or alias_lower in doc_text_lower:
                alias_matches.append(alias)
        
        # Pass if: at least 1 primary AND (at least 1 secondary OR alias)
        has_primary = len(primary_matches) > 0
        has_secondary_or_alias = len(secondary_matches) > 0 or len(alias_matches) > 0
        
        passed = has_primary and has_secondary_or_alias
        
        logger.debug(
            f"Entity gate result: passed={passed}, "
            f"primary={len(primary_matches)}, secondary={len(secondary_matches)}, aliases={len(alias_matches)}"
        )
        
        return passed, primary_matches, secondary_matches, alias_matches


class EventMapper:
    """
    3-stage event-document mapping pipeline.
    
    Stage 1: Hard Entity Gate
    Stage 2: Semantic Relevance (sentence-transformers)
    Stage 3: Dependency Classification (zero-shot)
    """
    
    def __init__(
        self,
        registry: Optional[EventRegistry] = None,
        normalizer: Optional[DocumentNormalizer] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize event mapper.
        
        Args:
            registry: Event registry
            normalizer: Document normalizer
            output_dir: Directory for mapping results
        """
        logger.info("Initializing EventMapper...")
        self.registry = registry or get_registry()
        self.normalizer = normalizer or DocumentNormalizer()
        
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "data" / "normalized"
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.entity_gate = EntityGate()
        self.relevance_scorer = SemanticRelevanceScorer()
        self.dependency_classifier = DependencyClassifier()
        logger.info("EventMapper initialized")
    
    def map_document_to_event(
        self,
        doc: NormalizedDocument,
        event: Event,
        require_entity_gate: bool = True
    ) -> DocumentMapping:
        """
        Map a single document to an event through all 3 stages.
        
        Args:
            doc: Normalized document
            event: Event to map against
            require_entity_gate: If True, skip Stage 2-3 if Stage 1 fails
        
        Returns:
            DocumentMapping result
        """
        logger.debug(f"Mapping document {doc.doc_id} to event {event.event_id}")
        
        # Stage 1: Entity Gate
        (
            gate_passed,
            primary_matches,
            secondary_matches,
            alias_matches
        ) = self.entity_gate.check(doc, event)
        
        # Early exit if gate fails and required
        if require_entity_gate and not gate_passed:
            logger.debug(f"Document {doc.doc_id} failed entity gate")
            return DocumentMapping(
                doc_id=doc.doc_id,
                event_id=event.event_id,
                entity_gate_passed=False,
                primary_entities_matched=primary_matches,
                secondary_entities_matched=secondary_matches,
                aliases_matched=alias_matches,
                relevance_score=0.0,
                relevance_passed=False,
                dependency_scores={},
                top_dependencies=[],
                is_relevant=False
            )
        
        # Stage 2: Semantic Relevance
        logger.debug(f"Stage 2: Computing semantic relevance for {doc.doc_id}")
        relevance_score, relevance_passed = self.relevance_scorer.score_relevance(
            event.event_description,
            doc.raw_text
        )
        
        # Stage 3: Dependency Classification
        logger.debug(f"Stage 3: Classifying dependencies for {doc.doc_id}")
        dependency_scores = self.dependency_classifier.classify(doc.raw_text)
        top_deps = [
            dep for dep, score in 
            sorted(dependency_scores.items(), key=lambda x: x[1], reverse=True)
            if score >= 0.3
        ][:3]
        
        # Overall relevance: passed entity gate AND semantic threshold
        is_relevant = gate_passed and relevance_passed
        
        logger.debug(
            f"Mapping complete for {doc.doc_id}: "
            f"relevant={is_relevant}, score={relevance_score:.3f}"
        )
        
        return DocumentMapping(
            doc_id=doc.doc_id,
            event_id=event.event_id,
            entity_gate_passed=gate_passed,
            primary_entities_matched=primary_matches,
            secondary_entities_matched=secondary_matches,
            aliases_matched=alias_matches,
            relevance_score=round(relevance_score, 4),
            relevance_passed=relevance_passed,
            dependency_scores=dependency_scores,
            top_dependencies=top_deps,
            is_relevant=is_relevant
        )
    
    def map_document_to_all_events(
        self,
        doc: NormalizedDocument
    ) -> List[DocumentMapping]:
        """
        Map a document to all registered events.
        
        Args:
            doc: Normalized document
        
        Returns:
            List of mappings (one per event)
        """
        mappings = []
        
        for event in self.registry:
            mapping = self.map_document_to_event(doc, event)
            mappings.append(mapping)
        
        return mappings
    
    def map_all_documents_to_event(
        self,
        event: Event,
        documents: Optional[List[NormalizedDocument]] = None
    ) -> List[DocumentMapping]:
        """
        Map all documents to a single event.
        
        Args:
            event: Target event
            documents: Optional list of documents (loads all if None)
        
        Returns:
            List of mappings for relevant documents
        """
        if documents is None:
            documents = self.normalizer.load_all_for_event(event.event_id)
        
        mappings = []
        for doc in documents:
            mapping = self.map_document_to_event(doc, event)
            mappings.append(mapping)
        
        return mappings
    
    def get_relevant_documents(
        self,
        event: Event,
        documents: Optional[List[NormalizedDocument]] = None,
        min_relevance: Optional[float] = None
    ) -> List[Tuple[NormalizedDocument, DocumentMapping]]:
        """
        Get documents relevant to an event.
        
        Args:
            event: Target event
            documents: Documents to check
            min_relevance: Minimum relevance score
        
        Returns:
            List of (document, mapping) tuples for relevant documents
        """
        mappings = self.map_all_documents_to_event(event, documents)
        
        # Build doc lookup
        if documents is None:
            documents = self.normalizer.load_all_for_event(event.event_id)
        doc_lookup = {doc.doc_id: doc for doc in documents}
        
        # Filter relevant
        relevant = []
        for mapping in mappings:
            if not mapping.is_relevant:
                continue
            if min_relevance and mapping.relevance_score < min_relevance:
                continue
            
            doc = doc_lookup.get(mapping.doc_id)
            if doc:
                relevant.append((doc, mapping))
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x[1].relevance_score, reverse=True)
        
        return relevant
    
    def save_mapping(self, mapping: DocumentMapping) -> None:
        """Save a mapping result to disk."""
        from utils.json_utils import dump_json
        mapping_path = self.output_dir / f"{mapping.doc_id}_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            dump_json(mapping.to_dict(), f)
    
    def load_mapping(self, doc_id: str) -> Optional[DocumentMapping]:
        """Load a mapping result from disk."""
        mapping_path = self.output_dir / f"{doc_id}_mapping.json"
        if not mapping_path.exists():
            return None
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return DocumentMapping.from_dict(data)
    
    def process_event(
        self,
        event_id: str,
        save_mappings: bool = True,
        use_batch: bool = True
    ) -> Dict:
        """
        Process all documents for an event through the mapping pipeline.
        
        Args:
            event_id: Event ID to process
            save_mappings: Whether to save mapping results
            use_batch: If True, use optimized batch processing (10-40x faster)
        
        Returns:
            Summary of mapping results
        """
        event = self.registry.get_event(event_id)
        if event is None:
            raise ValueError(f"Event not found: {event_id}")
        
        # Load all documents for this event
        documents = self.normalizer.load_all_for_event(event_id)
        
        logger.info(f"Mapping {len(documents)} documents to event {event_id}")
        
        # Use batch processing by default (much faster!)
        if use_batch:
            mappings = self.map_all_documents_to_event_batch(
                event,
                documents,
                save_mappings=save_mappings
            )
        else:
            # Legacy sequential processing
            mappings = []
            for doc in documents:
                mapping = self.map_document_to_event(doc, event)
                mappings.append(mapping)
                
                if save_mappings:
                    self.save_mapping(mapping)
        
        relevant_count = sum(1 for m in mappings if m.is_relevant)
        
        # Build summary
        summary = {
            "event_id": event_id,
            "total_documents": len(documents),
            "relevant_documents": relevant_count,
            "entity_gate_passed": sum(1 for m in mappings if m.entity_gate_passed),
            "semantic_passed": sum(1 for m in mappings if m.relevance_passed),
            "avg_relevance_score": (
                sum(m.relevance_score for m in mappings) / len(mappings)
                if mappings else 0
            ),
            "dependency_coverage": {}
        }
        
        # Count documents per dependency
        for dep in ["training", "compute", "safety", "regulation", 
                    "executive_statement", "public_narrative"]:
            count = sum(
                1 for m in mappings 
                if m.is_relevant and dep in m.top_dependencies
            )
            summary["dependency_coverage"][dep] = count
        
        logger.info(
            f"Event {event_id}: {relevant_count}/{len(documents)} "
            f"documents relevant"
        )
        
        return summary
    
    def map_all_documents_to_event_batch(
        self,
        event: Event,
        documents: Optional[List[NormalizedDocument]] = None,
        save_mappings: bool = True
    ) -> List[DocumentMapping]:
        """
        OPTIMIZED: Map all documents to an event using batch processing.
        
        This method is 10-40x faster than map_all_documents_to_event because it:
        1. Processes entity gates first (fast filter)
        2. Batches all semantic relevance scoring in one pass
        3. Reuses embeddings for dependency classification
        
        Args:
            event: Target event
            documents: Optional list of documents (loads all if None)
            save_mappings: Whether to save mapping results
        
        Returns:
            List of mappings for all documents
        """
        if documents is None:
            documents = self.normalizer.load_all_for_event(event.event_id)
        
        if not documents:
            logger.warning(f"No documents found for event {event.event_id}")
            return []
        
        logger.info(f"Batch mapping {len(documents)} documents to event {event.event_id}")
        
        # Stage 1: Entity Gate (fast, do all upfront)
        logger.info(f"Stage 1: Running entity gate for {len(documents)} documents...")
        entity_results = []
        passed_docs = []
        passed_indices = []
        
        for i, doc in enumerate(documents):
            (
                gate_passed,
                primary_matches,
                secondary_matches,
                alias_matches
            ) = self.entity_gate.check(doc, event)
            
            entity_results.append({
                'passed': gate_passed,
                'primary': primary_matches,
                'secondary': secondary_matches,
                'aliases': alias_matches
            })
            
            if gate_passed:
                passed_docs.append(doc)
                passed_indices.append(i)
        
        logger.info(f"Entity gate: {len(passed_docs)}/{len(documents)} passed")
        
        # Stage 2 & 3: Batch semantic + dependency for passed documents
        relevance_scores = []
        dependency_scores_list = []
        
        if passed_docs:
            logger.info(f"Stage 2-3: Batch processing {len(passed_docs)} documents...")
            
            # Get all document texts
            doc_texts = [doc.raw_text for doc in passed_docs]
            
            # Batch semantic scoring (returns embeddings too!)
            scores, doc_embeddings = self.relevance_scorer.score_batch_optimized(
                event.event_description,
                doc_texts
            )
            relevance_scores = scores
            
            # Batch dependency classification (reuses embeddings!)
            if doc_embeddings is not None:
                dependency_scores_list = self.dependency_classifier.classify_batch_optimized(
                    doc_embeddings
                )
            else:
                # Fallback if embeddings not available
                logger.warning("Embeddings not available, using fallback dependency classification")
                dependency_scores_list = [
                    self.dependency_classifier.classify(doc.raw_text)
                    for doc in passed_docs
                ]
            
            logger.info(f"Batch processing complete")
        
        # Build final mappings
        mappings = []
        passed_idx = 0
        
        for i, doc in enumerate(documents):
            entity_result = entity_results[i]
            
            if i in passed_indices:
                # Document passed entity gate
                relevance_score, relevance_passed = relevance_scores[passed_idx]
                dependency_scores = dependency_scores_list[passed_idx]
                passed_idx += 1
                
                # Get top dependencies
                top_deps = [
                    dep for dep, score in 
                    sorted(dependency_scores.items(), key=lambda x: x[1], reverse=True)
                    if score >= 0.3
                ][:3]
                
                is_relevant = entity_result['passed'] and relevance_passed
            else:
                # Document failed entity gate
                relevance_score = 0.0
                relevance_passed = False
                dependency_scores = {}
                top_deps = []
                is_relevant = False
            
            mapping = DocumentMapping(
                doc_id=doc.doc_id,
                event_id=event.event_id,
                entity_gate_passed=entity_result['passed'],
                primary_entities_matched=entity_result['primary'],
                secondary_entities_matched=entity_result['secondary'],
                aliases_matched=entity_result['aliases'],
                relevance_score=round(relevance_score, 4),
                relevance_passed=relevance_passed,
                dependency_scores=dependency_scores,
                top_dependencies=top_deps,
                is_relevant=is_relevant
            )
            
            mappings.append(mapping)
            
            if save_mappings:
                self.save_mapping(mapping)
        
        relevant_count = sum(1 for m in mappings if m.is_relevant)
        logger.info(
            f"Batch mapping complete: {relevant_count}/{len(documents)} relevant"
        )
        
        return mappings

