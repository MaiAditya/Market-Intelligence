"""
Logging utilities for graph operations.

Provides structured JSON logging for all graph modifications.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def setup_graph_logger(
    module_name: str,
    output_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger for graph operations with JSON file output.
    
    Args:
        module_name: Name of the module (e.g., 'temporal_decay')
        output_dir: Directory for log files (default: data/graph_logs/<module_name>)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"causal_graph.{module_name}")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output_dir specified)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / f"{module_name}.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_edge_update(
    output_file: Path,
    edge_data: Dict[str, Any],
    append: bool = True
) -> None:
    """
    Log an edge update to a JSONL file.
    
    Args:
        output_file: Path to JSONL output file
        edge_data: Dictionary containing edge update information
        append: If True, append to file; if False, overwrite
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    edge_data['timestamp'] = datetime.utcnow().isoformat()
    
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(json.dumps(edge_data) + '\n')


def log_summary(
    output_file: Path,
    summary_data: Dict[str, Any]
) -> None:
    """
    Log a summary to a JSON file.
    
    Args:
        output_file: Path to JSON output file
        summary_data: Dictionary containing summary information
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    summary_data['timestamp'] = datetime.utcnow().isoformat()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)


def read_jsonl(file_path: Path) -> list:
    """
    Read a JSONL file and return list of records.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries
    """
    if not file_path.exists():
        return []
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    return records


def read_json(file_path: Path) -> Dict:
    """
    Read a JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary
    """
    if not file_path.exists():
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
