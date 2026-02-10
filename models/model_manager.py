"""
Model Manager

Handles downloading, caching, and loading of BERT-style models.
All models are from Hugging Face and are open-source.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages BERT-style model loading and caching.
    
    Supported models:
    - NER: dslim/bert-base-NER
    - Sentence Transformer: sentence-transformers/all-mpnet-base-v2
    - Dependency Classifier: bert-base-uncased (for embeddings)
    - Signal Classifier: roberta-base (for embeddings)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            config_path: Path to model_config.json
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "model_config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set cache directory
        cache_dir = self.config.get("cache_dir", "./models/cache")
        if not os.path.isabs(cache_dir):
            cache_dir = Path(__file__).parent.parent / cache_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for transformers cache
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        os.environ["HF_HOME"] = str(self.cache_dir)
        
        # Model cache
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
    
    def _load_config(self) -> Dict:
        """Load model configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_ner_pipeline(self):
        """
        Get NER pipeline using dslim/bert-base-NER.
        
        Returns:
            Hugging Face NER pipeline
        """
        if "ner_pipeline" in self._models:
            return self._models["ner_pipeline"]
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            
            model_name = self.config.get("ner_model", "dslim/bert-base-NER")
            
            logger.info(f"Loading NER model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
            
            self._models["ner_pipeline"] = ner_pipeline
            logger.info("NER model loaded successfully")
            
            return ner_pipeline
            
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            return None
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            return None
    
    def get_sentence_transformer(self):
        """
        Get sentence transformer for semantic similarity.
        
        Returns:
            SentenceTransformer model
        """
        if "sentence_transformer" in self._models:
            return self._models["sentence_transformer"]
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.get(
                "sentence_transformer",
                "sentence-transformers/all-mpnet-base-v2"
            )
            
            logger.info(f"Loading sentence transformer: {model_name}")
            
            model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir)
            )
            
            self._models["sentence_transformer"] = model
            logger.info("Sentence transformer loaded successfully")
            
            return model
            
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            return None
    
    def get_bert_model(self, model_key: str = "dependency_classifier"):
        """
        Get BERT model for classification tasks.
        
        Args:
            model_key: Key in config (dependency_classifier or signal_classifier)
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_key in self._models:
            return self._models[model_key], self._tokenizers[model_key]
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_name = self.config.get(model_key, "bert-base-uncased")
            
            logger.info(f"Loading BERT model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            self._models[model_key] = model
            self._tokenizers[model_key] = tokenizer
            logger.info(f"BERT model {model_name} loaded successfully")
            
            return model, tokenizer
            
        except ImportError:
            logger.error("transformers not installed")
            return None, None
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            return None, None
    
    def get_threshold(self, threshold_name: str) -> float:
        """Get a configured threshold value."""
        thresholds = self.config.get("thresholds", {})
        defaults = {
            "semantic_relevance": 0.45,
            "entity_confidence": 0.7,
            "signal_confidence": 0.5,
            "dependency_threshold": 0.3
        }
        return thresholds.get(threshold_name, defaults.get(threshold_name, 0.5))
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length for tokenization."""
        return self.config.get("max_sequence_length", 512)
    
    def get_batch_size(self) -> int:
        """Get batch size for inference."""
        return self.config.get("batch_size", 16)
    
    def is_available(self, model_type: str) -> bool:
        """
        Check if a model type is available.
        
        Args:
            model_type: One of 'ner', 'sentence_transformer', 'bert'
        
        Returns:
            True if model can be loaded
        """
        if model_type == "ner":
            try:
                from transformers import pipeline
                return True
            except ImportError:
                return False
        elif model_type == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
                return True
            except ImportError:
                return False
        elif model_type == "bert":
            try:
                from transformers import AutoModel
                return True
            except ImportError:
                return False
        return False


# Module-level singleton
_manager: Optional[ModelManager] = None


def get_model_manager(config_path: Optional[str] = None) -> ModelManager:
    """Get the model manager singleton."""
    global _manager
    if _manager is None:
        _manager = ModelManager(config_path)
    return _manager
