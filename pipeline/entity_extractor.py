"""
Entity Extractor Pipeline

Wrapper for NER extraction that integrates with the document pipeline.
Extracts entities from normalized documents and updates them.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.ner import NERExtractor, ExtractedEntity
from pipeline.normalizer import DocumentNormalizer, NormalizedDocument

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Entity extraction pipeline for documents.
    
    Extracts entities from normalized documents using BERT NER
    and custom regex patterns.
    """
    
    def __init__(
        self,
        normalizer: Optional[DocumentNormalizer] = None
    ):
        """
        Initialize entity extractor.
        
        Args:
            normalizer: DocumentNormalizer instance for loading/saving docs
        """
        logger.info("Initializing EntityExtractor...")
        self.ner = NERExtractor()
        self.normalizer = normalizer or DocumentNormalizer()
        logger.info("EntityExtractor initialized")
    
    def extract_from_document(
        self,
        doc: NormalizedDocument
    ) -> List[Dict]:
        """
        Extract entities from a normalized document.
        
        Args:
            doc: NormalizedDocument to extract from
        
        Returns:
            List of entity dictionaries
        """
        logger.debug(f"Extracting entities from document: {doc.doc_id}")
        
        # Extract from title and text
        title_entities = self.ner.extract(doc.title)
        text_entities = self.ner.extract(doc.raw_text)
        
        # Combine and deduplicate
        all_entities = title_entities + text_entities
        
        # Deduplicate by text + type
        seen = set()
        unique = []
        for entity in all_entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        logger.debug(f"Extracted {len(unique)} unique entities from {doc.doc_id}")
        return [e.to_dict() for e in unique]
    
    def extract_and_update(
        self,
        doc_id: str
    ) -> Optional[NormalizedDocument]:
        """
        Extract entities and update the document.
        
        Args:
            doc_id: Document ID to process
        
        Returns:
            Updated document or None
        """
        doc = self.normalizer.load(doc_id)
        if doc is None:
            logger.warning(f"Document not found: {doc_id}")
            return None
        
        entities = self.extract_from_document(doc)
        doc.extracted_entities = entities
        self.normalizer.save(doc)
        
        logger.info(f"Extracted {len(entities)} entities from {doc_id}")
        return doc
    
    def process_event_documents(
        self,
        event_id: str
    ) -> List[NormalizedDocument]:
        """
        Extract entities from all documents for an event.
        
        Uses batched BERT NER for much faster processing.
        
        Args:
            event_id: Event ID to process
        
        Returns:
            List of updated documents
        """
        logger.info(f"Processing entities for event: {event_id}")
        docs = self.normalizer.load_all_for_event(event_id)
        
        # Filter to docs that need extraction
        needs_extraction = [doc for doc in docs if not doc.extracted_entities]
        
        if not needs_extraction:
            logger.info(f"All {len(docs)} documents already have entities extracted")
            return []
        
        logger.info(f"Batched NER extraction for {len(needs_extraction)} documents...")
        
        # Combine title + text for each doc (matching the original per-doc logic)
        combined_texts = [f"{doc.title}\n{doc.raw_text}" for doc in needs_extraction]
        
        # Batch extract using the new NERExtractor.batch_extract method
        all_entity_lists = self.ner.batch_extract(combined_texts, batch_size=16)
        
        updated = []
        for doc, entities in zip(needs_extraction, all_entity_lists):
            # Deduplicate by text + type (same logic as extract_from_document)
            seen = set()
            unique = []
            for entity in entities:
                key = (entity.text.lower(), entity.entity_type)
                if key not in seen:
                    seen.add(key)
                    unique.append(entity)
            
            doc.extracted_entities = [e.to_dict() for e in unique]
            self.normalizer.save(doc)
            updated.append(doc)
        
        logger.info(
            f"Extracted entities from {len(updated)} documents "
            f"for event {event_id}"
        )
        return updated
    
    def get_entity_summary(
        self,
        docs: List[NormalizedDocument]
    ) -> Dict:
        """
        Get summary of entities across documents.
        
        Args:
            docs: List of documents to summarize
        
        Returns:
            Summary dictionary
        """
        summary = {
            "total_entities": 0,
            "by_type": {},
            "top_organizations": {},
            "top_models": {},
            "dates_mentioned": set()
        }
        
        for doc in docs:
            for entity in doc.extracted_entities:
                summary["total_entities"] += 1
                
                entity_type = entity.get("type", "UNKNOWN")
                summary["by_type"][entity_type] = (
                    summary["by_type"].get(entity_type, 0) + 1
                )
                
                text = entity.get("text", "")
                if entity_type == "ORG":
                    summary["top_organizations"][text] = (
                        summary["top_organizations"].get(text, 0) + 1
                    )
                elif entity_type == "MODEL":
                    summary["top_models"][text] = (
                        summary["top_models"].get(text, 0) + 1
                    )
                elif entity_type == "DATE":
                    summary["dates_mentioned"].add(text)
        
        # Convert set to list for JSON serialization
        summary["dates_mentioned"] = list(summary["dates_mentioned"])
        
        # Sort by frequency
        summary["top_organizations"] = dict(
            sorted(
                summary["top_organizations"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        )
        summary["top_models"] = dict(
            sorted(
                summary["top_models"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )
        
        logger.debug(f"Entity summary: {summary['total_entities']} total entities")
        return summary
