"""
Named Entity Recognition Module

Uses dslim/bert-base-NER for entity extraction.
Extracts:
- Organizations (ORG)
- Model names (via custom regex + MISC)
- Dates (via pattern matching)
- Benchmarks (via pattern matching)
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .model_manager import get_model_manager

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    text: str
    entity_type: str  # ORG, MODEL, DATE, BENCHMARK, PERSON, LOCATION
    confidence: float
    start: int
    end: int
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.entity_type,
            "confidence": float(self.confidence),  # Convert numpy float32 to Python float
            "start": int(self.start),
            "end": int(self.end)
        }


class ModelNameExtractor:
    """
    Custom extractor for AI model names.
    
    Uses regex patterns since BERT NER may not recognize model names.
    """
    
    # Known model name patterns
    MODEL_PATTERNS = [
        # GPT variants
        r'\b(GPT-?[0-9]+(?:\.[0-9]+)?(?:\s*(?:Turbo|Vision|Pro|Mini))?)\b',
        r'\b(ChatGPT(?:-[0-9]+)?)\b',
        r'\b(o[0-9]+(?:-mini|-preview)?)\b',  # o1, o3, etc.
        
        # Gemini variants
        r'\b(Gemini\s*(?:[0-9]+(?:\.[0-9]+)?)?(?:\s*(?:Pro|Ultra|Nano|Flash))?)\b',
        
        # Claude variants
        r'\b(Claude\s*(?:[0-9]+(?:\.[0-9]+)?)?(?:\s*(?:Opus|Sonnet|Haiku|Instant))?)\b',
        
        # LLaMA variants
        r'\b(LLa[Mm][Aa]-?[0-9]*(?:\.[0-9]+)?)\b',
        r'\b(Llama\s*[0-9]+(?:\.[0-9]+)?)\b',
        
        # Mistral variants
        r'\b(Mistral(?:-[A-Za-z0-9]+)?)\b',
        r'\b(Mixtral(?:-[A-Za-z0-9]+)?)\b',
        
        # Other models
        r'\b(BERT(?:-(?:base|large|tiny))?(?:-(?:uncased|cased))?)\b',
        r'\b(PaLM\s*[0-9]*)\b',
        r'\b(Grok(?:-[0-9]+)?)\b',
        r'\b(Phi-?[0-9]+)\b',
        r'\b(Qwen[0-9]*(?:\.[0-9]+)?)\b',
        r'\b(Command\s*R(?:\+)?)\b',
        r'\b(Stable\s*Diffusion(?:\s*[0-9]+(?:\.[0-9]+)?)?)\b',
        r'\b(DALL-?E(?:-?[0-9]+)?)\b',
        r'\b(Midjourney(?:\s*[Vv][0-9]+)?)\b',
        r'\b(Sora)\b',
    ]
    
    @classmethod
    def extract(cls, text: str) -> List[ExtractedEntity]:
        """
        Extract model names from text.
        
        Args:
            text: Text to search
        
        Returns:
            List of extracted model entities
        """
        entities = []
        seen = set()
        
        for pattern in cls.MODEL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                model_name = match.group(1)
                normalized = model_name.lower().replace(" ", "")
                
                if normalized not in seen:
                    seen.add(normalized)
                    entities.append(ExtractedEntity(
                        text=model_name,
                        entity_type="MODEL",
                        confidence=0.9,  # High confidence for regex matches
                        start=match.start(1),
                        end=match.end(1)
                    ))
        
        return entities


class DateExtractor:
    """
    Extract date mentions from text.
    
    Uses patterns for various date formats.
    """
    
    DATE_PATTERNS = [
        # Full dates
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
        
        # Month Year
        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4})\b',
        
        # Quarter Year
        r'\b(Q[1-4]\s*\d{4})\b',
        r'\b((?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?\d{4})\b',
        
        # Year only in context
        r'\b(end\s+of\s+\d{4})\b',
        r'\b(early\s+\d{4})\b',
        r'\b(late\s+\d{4})\b',
        r'\b(mid-?\d{4})\b',
        r'\b(by\s+\d{4})\b',
        
        # Relative dates
        r'\b(this\s+year)\b',
        r'\b(next\s+year)\b',
        r'\b(by\s+year\s*end)\b',
    ]
    
    @classmethod
    def extract(cls, text: str) -> List[ExtractedEntity]:
        """Extract date mentions from text."""
        entities = []
        seen = set()
        
        for pattern in cls.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_text = match.group(1)
                normalized = date_text.lower()
                
                if normalized not in seen:
                    seen.add(normalized)
                    entities.append(ExtractedEntity(
                        text=date_text,
                        entity_type="DATE",
                        confidence=0.85,
                        start=match.start(1),
                        end=match.end(1)
                    ))
        
        return entities


class BenchmarkExtractor:
    """
    Extract AI benchmark mentions from text.
    """
    
    BENCHMARK_PATTERNS = [
        r'\b(MMLU)\b',
        r'\b(HumanEval)\b',
        r'\b(HellaSwag)\b',
        r'\b(ARC(?:-Challenge|-Easy)?)\b',
        r'\b(WinoGrande)\b',
        r'\b(GSM8K)\b',
        r'\b(MATH)\b',
        r'\b(TruthfulQA)\b',
        r'\b(BigBench)\b',
        r'\b(GLUE)\b',
        r'\b(SuperGLUE)\b',
        r'\b(SQuAD(?:\s*[0-9.]+)?)\b',
        r'\b(MT-?Bench)\b',
        r'\b(AlpacaEval)\b',
        r'\b(Chatbot\s*Arena)\b',
        r'\b(LMSYS)\b',
    ]
    
    @classmethod
    def extract(cls, text: str) -> List[ExtractedEntity]:
        """Extract benchmark mentions from text."""
        entities = []
        seen = set()
        
        for pattern in cls.BENCHMARK_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                benchmark = match.group(1)
                normalized = benchmark.lower()
                
                if normalized not in seen:
                    seen.add(normalized)
                    entities.append(ExtractedEntity(
                        text=benchmark,
                        entity_type="BENCHMARK",
                        confidence=0.95,
                        start=match.start(1),
                        end=match.end(1)
                    ))
        
        return entities


class NERExtractor:
    """
    Main NER extraction pipeline.
    
    Combines:
    1. BERT-based NER for organizations, people, locations
    2. Custom regex for model names
    3. Custom regex for dates
    4. Custom regex for benchmarks
    """
    
    def __init__(self):
        """Initialize NER extractor."""
        self.model_manager = get_model_manager()
        self._ner_pipeline = None
        self.min_confidence = self.model_manager.get_threshold("entity_confidence")
    
    def _get_ner_pipeline(self):
        """Lazy load NER pipeline."""
        if self._ner_pipeline is None:
            self._ner_pipeline = self.model_manager.get_ner_pipeline()
        return self._ner_pipeline
    
    def _bert_ner_extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities using BERT NER.
        
        Args:
            text: Text to extract from
        
        Returns:
            List of extracted entities
        """
        pipeline = self._get_ner_pipeline()
        if pipeline is None:
            logger.warning("NER pipeline not available")
            return []
        
        entities = []
        
        try:
            # Truncate text if too long
            max_len = self.model_manager.get_max_sequence_length()
            if len(text) > max_len * 4:  # Approximate character limit
                text = text[:max_len * 4]
            
            # Run NER
            results = pipeline(text)
            
            # Map NER labels to our entity types
            label_map = {
                "ORG": "ORG",
                "B-ORG": "ORG",
                "I-ORG": "ORG",
                "PER": "PERSON",
                "B-PER": "PERSON",
                "I-PER": "PERSON",
                "LOC": "LOCATION",
                "B-LOC": "LOCATION",
                "I-LOC": "LOCATION",
                "MISC": "MISC",
                "B-MISC": "MISC",
                "I-MISC": "MISC",
            }
            
            for result in results:
                entity_type = label_map.get(result.get("entity_group", ""), "MISC")
                confidence = result.get("score", 0.0)
                
                if confidence >= self.min_confidence:
                    entities.append(ExtractedEntity(
                        text=result.get("word", ""),
                        entity_type=entity_type,
                        confidence=confidence,
                        start=result.get("start", 0),
                        end=result.get("end", 0)
                    ))
            
        except Exception as e:
            logger.error(f"BERT NER extraction failed: {e}")
        
        return entities
    
    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract all entity types from text.
        
        Args:
            text: Text to extract from
        
        Returns:
            List of all extracted entities
        """
        all_entities = []
        
        # BERT NER for organizations, people, locations
        bert_entities = self._bert_ner_extract(text)
        all_entities.extend(bert_entities)
        
        # Custom extractors
        model_entities = ModelNameExtractor.extract(text)
        all_entities.extend(model_entities)
        
        date_entities = DateExtractor.extract(text)
        all_entities.extend(date_entities)
        
        benchmark_entities = BenchmarkExtractor.extract(text)
        all_entities.extend(benchmark_entities)
        
        # Deduplicate by text (case-insensitive)
        seen = set()
        unique_entities = []
        for entity in all_entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        # Sort by position
        unique_entities.sort(key=lambda e: e.start)
        
        return unique_entities
    
    def batch_extract(self, texts: List[str], batch_size: int = 16) -> List[List[ExtractedEntity]]:
        """
        Extract entities from multiple texts using batched BERT inference.
        
        Much faster than calling extract() in a loop because the HF pipeline
        processes all texts in GPU-batched chunks.
        
        Args:
            texts: List of texts to extract from
            batch_size: Batch size for BERT inference
        
        Returns:
            List of entity lists, one per input text
        """
        if not texts:
            return []
        
        pipeline = self._get_ner_pipeline()
        max_len = self.model_manager.get_max_sequence_length()
        
        # Truncate all texts
        truncated = [
            t[:max_len * 4] if len(t) > max_len * 4 else t
            for t in texts
        ]
        
        # --- Batched BERT NER ---
        all_bert_results = [[] for _ in range(len(texts))]
        if pipeline is not None:
            try:
                batch_results = pipeline(truncated, batch_size=batch_size)
                # pipeline returns List[List[dict]] when given a list of strings
                if batch_results and isinstance(batch_results[0], dict):
                    # Single text was passed (shouldn't happen, but guard)
                    batch_results = [batch_results]
                all_bert_results = batch_results
            except Exception as e:
                logger.error(f"Batched BERT NER failed: {e}")
        
        label_map = {
            "ORG": "ORG", "B-ORG": "ORG", "I-ORG": "ORG",
            "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
            "LOC": "LOCATION", "B-LOC": "LOCATION", "I-LOC": "LOCATION",
            "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC",
        }
        
        results = []
        for idx, text in enumerate(texts):
            all_entities = []
            
            # BERT entities for this text
            for result in all_bert_results[idx]:
                entity_type = label_map.get(result.get("entity_group", ""), "MISC")
                confidence = result.get("score", 0.0)
                if confidence >= self.min_confidence:
                    all_entities.append(ExtractedEntity(
                        text=result.get("word", ""),
                        entity_type=entity_type,
                        confidence=confidence,
                        start=result.get("start", 0),
                        end=result.get("end", 0)
                    ))
            
            # Custom regex extractors (already fast)
            all_entities.extend(ModelNameExtractor.extract(text))
            all_entities.extend(DateExtractor.extract(text))
            all_entities.extend(BenchmarkExtractor.extract(text))
            
            # Deduplicate
            seen = set()
            unique = []
            for entity in all_entities:
                key = (entity.text.lower(), entity.entity_type)
                if key not in seen:
                    seen.add(key)
                    unique.append(entity)
            unique.sort(key=lambda e: e.start)
            results.append(unique)
        
        return results
    
    def extract_to_dict(self, text: str) -> List[Dict]:
        """Extract entities and return as dictionaries."""
        entities = self.extract(text)
        return [e.to_dict() for e in entities]
    
    def get_organizations(self, text: str) -> List[str]:
        """Get just organization names from text."""
        entities = self.extract(text)
        return [e.text for e in entities if e.entity_type == "ORG"]
    
    def get_models(self, text: str) -> List[str]:
        """Get just model names from text."""
        entities = self.extract(text)
        return [e.text for e in entities if e.entity_type == "MODEL"]
