"""
Precedent Database

Maintains a structured record of event-type → outcome patterns for
computing Bayesian priors in historical precedent scoring.

Key features:
- Stores per-event-type and per-mechanism-type outcome counts
- Computes Bayesian priors with Laplace smoothing (avoids zero-probability)
- Provides record_outcome() for incremental learning
- Thread-safe file-based persistence (JSON)

Why Bayesian priors work well here:
- With limited data, Laplace smoothing prevents extreme probabilities
- As more observations accumulate, the prior converges to the true frequency
- The alpha parameter (default 1.0) represents a uniform "pseudo-count" prior
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PrecedentDatabase:
    """
    Stores and computes Bayesian precedent scores for event types.
    
    Each record tracks:
    - event_type: The type of source event (policy, legal, etc.)
    - mechanism_type: Optional mechanism classification
    - correct_count: How many times this pattern led to correct belief direction
    - total_count: Total observations of this pattern
    
    The Bayesian prior is:
        P(correct | event_type) = (correct + alpha) / (total + 2*alpha)
    
    Where alpha is the Laplace smoothing parameter (default 1.0).
    With alpha=1.0 and no data, the prior is 0.5 (maximum uncertainty).
    """
    
    def __init__(
        self,
        db_path: str,
        laplace_alpha: float = 1.0
    ):
        """
        Initialize precedent database.
        
        Args:
            db_path: Path to the JSON file storing precedent records
            laplace_alpha: Laplace smoothing parameter (higher = more conservative)
        """
        self.db_path = Path(db_path)
        self.alpha = laplace_alpha
        self._data: Dict = {}
        
        logger.info(f"PrecedentDatabase: initializing from {self.db_path}")
        self._load()
        logger.info(
            f"PrecedentDatabase: loaded {len(self._data.get('event_types', {}))} "
            f"event types, {self._total_observations()} total observations"
        )
    
    def _load(self) -> None:
        """Load precedent data from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                logger.debug(f"PrecedentDatabase: loaded from {self.db_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"PrecedentDatabase: failed to load {self.db_path}: {e}, "
                    f"starting with empty database"
                )
                self._data = self._empty_db()
        else:
            logger.info(
                f"PrecedentDatabase: no existing file at {self.db_path}, "
                f"creating new database"
            )
            self._data = self._empty_db()
            self._save()
    
    def _empty_db(self) -> Dict:
        """Create an empty database structure."""
        return {
            "version": "1.0",
            "created_at": _utc_now().isoformat(),
            "updated_at": _utc_now().isoformat(),
            "event_types": {},
            "mechanisms": {}
        }
    
    def _save(self) -> None:
        """Persist precedent data to disk."""
        self._data["updated_at"] = _utc_now().isoformat()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            logger.debug(f"PrecedentDatabase: saved to {self.db_path}")
        except IOError as e:
            logger.error(f"PrecedentDatabase: failed to save to {self.db_path}: {e}")
    
    def _total_observations(self) -> int:
        """Count total observations across all event types."""
        total = 0
        for type_data in self._data.get("event_types", {}).values():
            total += type_data.get("total", 0)
        return total
    
    def get_precedent(
        self,
        event_type: str,
        mechanism_type: Optional[str] = None
    ) -> float:
        """
        Get the Bayesian prior for an event type (and optionally mechanism).
        
        Formula: P = (correct + alpha) / (total + 2*alpha)
        
        This gives:
        - 0.5 when no data exists (maximum uncertainty)
        - Converges to true frequency as data grows
        - Never returns exactly 0.0 or 1.0 (Laplace smoothing)
        
        Args:
            event_type: Source event type (e.g., "policy", "legal")
            mechanism_type: Optional mechanism for finer-grained precedent
        
        Returns:
            Bayesian prior score [0.0-1.0]
        """
        # Check mechanism-specific precedent first (more specific)
        if mechanism_type:
            mech_key = f"{event_type}:{mechanism_type}"
            mech_data = self._data.get("mechanisms", {}).get(mech_key, {})
            mech_total = mech_data.get("total", 0)
            
            if mech_total >= 2:  # Need at least 2 observations for mechanism-level
                correct = mech_data.get("correct", 0)
                score = (correct + self.alpha) / (mech_total + 2 * self.alpha)
                logger.debug(
                    f"PrecedentDatabase: mechanism-level precedent for "
                    f"{mech_key}: {score:.4f} (correct={correct}, total={mech_total})"
                )
                return score
        
        # Fall back to event-type-level precedent
        type_data = self._data.get("event_types", {}).get(event_type, {})
        total = type_data.get("total", 0)
        correct = type_data.get("correct", 0)
        
        score = (correct + self.alpha) / (total + 2 * self.alpha)
        
        logger.debug(
            f"PrecedentDatabase: event-type precedent for "
            f"{event_type}: {score:.4f} (correct={correct}, total={total})"
        )
        
        return score
    
    def record_outcome(
        self,
        event_type: str,
        was_correct: bool,
        mechanism_type: Optional[str] = None,
        auto_save: bool = True
    ) -> None:
        """
        Record an outcome observation for learning.
        
        Updates both event-type-level and mechanism-level counts.
        
        Args:
            event_type: Source event type
            was_correct: Whether this event type predicted belief direction correctly
            mechanism_type: Optional mechanism type
            auto_save: Whether to save immediately after recording
        """
        logger.info(
            f"PrecedentDatabase: recording outcome for {event_type} "
            f"(mechanism={mechanism_type}): correct={was_correct}"
        )
        
        # Update event-type-level stats
        if "event_types" not in self._data:
            self._data["event_types"] = {}
        
        if event_type not in self._data["event_types"]:
            self._data["event_types"][event_type] = {
                "correct": 0,
                "total": 0,
                "first_observed": _utc_now().isoformat()
            }
        
        self._data["event_types"][event_type]["total"] += 1
        if was_correct:
            self._data["event_types"][event_type]["correct"] += 1
        self._data["event_types"][event_type]["last_updated"] = _utc_now().isoformat()
        
        # Update mechanism-level stats if mechanism provided
        if mechanism_type:
            if "mechanisms" not in self._data:
                self._data["mechanisms"] = {}
            
            mech_key = f"{event_type}:{mechanism_type}"
            if mech_key not in self._data["mechanisms"]:
                self._data["mechanisms"][mech_key] = {
                    "correct": 0,
                    "total": 0,
                    "first_observed": _utc_now().isoformat()
                }
            
            self._data["mechanisms"][mech_key]["total"] += 1
            if was_correct:
                self._data["mechanisms"][mech_key]["correct"] += 1
            self._data["mechanisms"][mech_key]["last_updated"] = _utc_now().isoformat()
        
        new_score = self.get_precedent(event_type, mechanism_type)
        logger.info(
            f"PrecedentDatabase: updated precedent for {event_type}: {new_score:.4f}"
        )
        
        if auto_save:
            self._save()
    
    def record_batch(
        self,
        outcomes: list,
        auto_save: bool = True
    ) -> None:
        """
        Record multiple outcomes at once.
        
        Args:
            outcomes: List of dicts with keys: event_type, was_correct, mechanism_type (optional)
            auto_save: Whether to save after the batch
        """
        logger.info(f"PrecedentDatabase: recording batch of {len(outcomes)} outcomes")
        
        for outcome in outcomes:
            self.record_outcome(
                event_type=outcome["event_type"],
                was_correct=outcome["was_correct"],
                mechanism_type=outcome.get("mechanism_type"),
                auto_save=False  # Save once after batch
            )
        
        if auto_save:
            self._save()
        
        logger.info(
            f"PrecedentDatabase: batch recording complete, "
            f"{self._total_observations()} total observations"
        )
    
    def get_reliability_stats(self) -> Dict:
        """
        Get reliability statistics for all event types.
        
        Returns:
            Dictionary mapping event_type to stats dict with:
            - correct: number of correct predictions
            - total: total observations
            - accuracy: raw accuracy (correct/total)
            - bayesian_score: Laplace-smoothed Bayesian prior
        """
        stats = {}
        
        for event_type, type_data in self._data.get("event_types", {}).items():
            total = type_data.get("total", 0)
            correct = type_data.get("correct", 0)
            
            stats[event_type] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else 0.0,
                "bayesian_score": (correct + self.alpha) / (total + 2 * self.alpha),
                "first_observed": type_data.get("first_observed"),
                "last_updated": type_data.get("last_updated")
            }
        
        return stats
    
    def seed_from_defaults(self) -> None:
        """
        Seed the database with domain-knowledge defaults.
        
        These represent expert priors about how reliable different event types
        are at predicting belief direction. They provide a reasonable starting
        point before real observations accumulate.
        """
        logger.info("PrecedentDatabase: seeding with domain-knowledge defaults")
        
        # Domain knowledge: (event_type, assumed_correct_rate, pseudo_observations)
        defaults = [
            ("legal", 0.75, 4),      # Legal rulings are binding → high reliability
            ("policy", 0.65, 4),     # Policy announcements often follow through
            ("economic", 0.55, 3),   # Economic data is mixed signal
            ("signal", 0.50, 3),     # Signals vary widely
            ("poll", 0.45, 3),       # Polls are noisy indicators
            ("narrative", 0.35, 3),  # Narratives often don't materialize
            ("market", 0.55, 3),     # Market signals moderate reliability
        ]
        
        for event_type, correct_rate, n_obs in defaults:
            if event_type not in self._data.get("event_types", {}):
                correct = int(n_obs * correct_rate)
                incorrect = n_obs - correct
                
                # Record the pseudo-observations
                for _ in range(correct):
                    self.record_outcome(event_type, True, auto_save=False)
                for _ in range(incorrect):
                    self.record_outcome(event_type, False, auto_save=False)
                
                logger.info(
                    f"  Seeded {event_type}: {correct}/{n_obs} correct "
                    f"→ prior={self.get_precedent(event_type):.4f}"
                )
        
        self._save()
        logger.info("PrecedentDatabase: seeding complete")
