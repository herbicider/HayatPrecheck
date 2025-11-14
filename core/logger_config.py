import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

DEFAULT_LOG_FILE = "verification.log"
DEFAULT_MAX_BYTES = 10_000_000  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Messages that create excessive startup noise (missing Tesseract, Paddle warnings, etc.)
_SUPPRESSED_LOG_PATTERNS = [
    "tesseract not available",
    "auto-selected paddleocr",
    "module 'paddle' has no attribute 'disable_static_logging'",
    "using cpu. note: this module is much faster with a gpu.",
]


class _SuppressNoiseFilter(logging.Filter):
    """Filter out repetitive informational/warning logs from third-party libs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        return not any(pattern in message for pattern in _SUPPRESSED_LOG_PATTERNS)


def setup_logging(
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    level: int = logging.INFO,
    add_stream: bool = True,
) -> None:
    """Configure root logger with a rotating file handler.

    Safe to call multiple times; it avoids adding duplicate handlers.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Check if we already added a file handler for this log file
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None):
            if h.baseFilename.endswith(log_file):
                return  # Already configured

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    noise_filter = _SuppressNoiseFilter()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(noise_filter)
    logger.addHandler(file_handler)

    if add_stream:
        # Avoid duplicate stream handlers
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            stream_handler.addFilter(noise_filter)
            logger.addHandler(stream_handler)
    
    # Reduce verbosity of external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def log_rx_summary(rx_number: str, results: Dict[str, Any]) -> None:
    """Emit a clean Rx verification summary with OCR text and scores."""
    import datetime as _dt

    try:
        matches = sum(1 for r in results.values() if r["match"])
        total = len(results)
        match_percentage = (matches / total * 100) if total > 0 else 0

        timestamp = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rx_display = f"Rx#{rx_number}" if rx_number else "Rx#Unknown"

        logging.info(f"[{timestamp}] {rx_display} - {matches}/{total} fields matched ({match_percentage:.1f}%)")
        
        for field_name, result in results.items():
            status = 'YES' if result['match'] else 'NO'
            logging.info(f"=== {field_name.upper()} FIELD ===")
            
            # Check if this is VLM mode (no OCR text available)
            if result.get('method') == 'vlm':
                logging.info(f"  VLM Analysis: Visual comparison of prescription images")
                logging.info(f"  Score: {result['score']:.2f} | Threshold: {result['threshold']} | Match: {status}")
            else:
                # Log OCR text details if available
                if 'entered_raw' in result:
                    logging.info(f"  Raw entered text: '{result['entered_raw']}'")
                if 'source_raw' in result:
                    logging.info(f"  Raw source text: '{result['source_raw']}'")
                if 'entered_clean' in result:
                    logging.info(f"  Cleaned entered: '{result['entered_clean']}'")
                if 'source_clean' in result:
                    logging.info(f"  Cleaned source: '{result['source_clean']}'")
                if 'threshold' in result:
                    logging.info(f"  Score: {result['score']:.2f} | Threshold: {result['threshold']} | Match: {status}")
        
    except Exception as e:
        logging.error(f"Error logging Rx summary: {e}")


def log_field_details(field_name: str, entered_raw: str, source_raw: str, 
                     entered_clean: str, source_clean: str, score: float, 
                     threshold: float, is_match: bool) -> None:
    """Log detailed field comparison information for debugging.
    
    This function provides granular logging of each field's processing pipeline
    from raw OCR to final comparison result.
    """
    try:
        status = 'PASS' if is_match else 'FAIL'
        logging.debug(f"🔍 DETAILED ANALYSIS: {field_name}")
        logging.debug(f"   Raw OCR Entered: '{entered_raw}' (len: {len(entered_raw)})")
        logging.debug(f"   Raw OCR Source:  '{source_raw}' (len: {len(source_raw)})")
        logging.debug(f"   Clean Entered:   '{entered_clean}' (len: {len(entered_clean)})")
        logging.debug(f"   Clean Source:    '{source_clean}' (len: {len(source_clean)})")
        logging.debug(f"   Score: {score:.2f} | Threshold: {threshold} | Result: {status}")
        
        # Additional analysis for failed matches
        if not is_match and entered_clean and source_clean:
            import difflib
            diff = list(difflib.unified_diff(
                entered_clean.split(), source_clean.split(), 
                fromfile='entered', tofile='source', lineterm=''
            ))
            if diff:
                logging.debug(f"   📊 Word-level differences:")
                for line in diff[2:]:  # Skip header lines
                    if line.startswith('+') or line.startswith('-'):
                        logging.debug(f"      {line}")
                        
    except Exception as e:
        logging.error(f"Error logging field details for {field_name}: {e}")


def log_ocr_performance(field_name: str, region: tuple, ocr_time: float, 
                       text_length: int, confidence: Optional[float] = None) -> None:
    """Log OCR performance metrics for optimization purposes."""
    try:
        logging.debug(f"⚡ OCR PERFORMANCE: {field_name}")
        logging.debug(f"   Region: {region} | Time: {ocr_time:.3f}s | Text Length: {text_length}")
        if confidence is not None:
            logging.debug(f"   Confidence: {confidence:.2f}")
    except Exception as e:
        logging.error(f"Error logging OCR performance for {field_name}: {e}")

