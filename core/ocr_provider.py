
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from PIL.Image import Resampling
import numpy as np
import os
import sys
import time
import logging
import re
import warnings
from typing import Tuple, Dict, Any

# Suppress PyTorch DataLoader pin_memory warning when no GPU is available
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*", category=UserWarning)

from core.logger_config import log_ocr_performance

# Global cache for OCR providers to avoid reinitialization
_ocr_provider_cache = {}

# Static flag to prevent repeated availability logging
_availability_logged = False

# Static flag to prevent repeated fallback logging
_fallback_logged = {}

# Counter for cache usage logging
_cache_usage_count = 0

def check_gpu_availability():
    """Check if a supported GPU is available for deep learning."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logging.info(f"CUDA GPU available: {device_name} ({device_count} device(s))")
            return True
        else:
            logging.info("CUDA GPU not available")
            return False
    except ImportError:
        logging.info("PyTorch not available - cannot check GPU status")
        return False
    except Exception as e:
        logging.warning(f"Error checking GPU availability: {e}")
        return False

def auto_select_ocr_provider():
    """Automatically select the best OCR provider based on system capabilities."""
    available_providers = check_ocr_availability()
    gpu_available = check_gpu_availability()
    
    # Decision logic: EasyOCR with GPU > Tesseract > PaddleOCR > EasyOCR without GPU
    if gpu_available and "easyocr" in available_providers:
        logging.info("Auto-selected EasyOCR with GPU acceleration (best accuracy)")
        return "easyocr", True  # provider, use_gpu
    elif "tesseract" in available_providers:
        logging.info("Auto-selected Tesseract (good speed, widely compatible)")
        return "tesseract", False
    elif "paddleocr" in available_providers:
        logging.info("Auto-selected PaddleOCR (good accuracy, CPU optimized)")
        return "paddleocr", False
    elif "easyocr" in available_providers:
        logging.info("Auto-selected EasyOCR with CPU (GPU not available)")
        return "easyocr", False
    else:
        raise ImportError("No suitable OCR provider available")

def check_ocr_availability():
    """Check which OCR providers are available on the system."""
    global _availability_logged
    
    available_providers = []
    
    # Check EasyOCR
    try:
        import easyocr
        available_providers.append("easyocr")
        if not _availability_logged:
            logging.info("EasyOCR is available")
    except ImportError:
        if not _availability_logged:
            logging.warning("EasyOCR not available - install with: pip install easyocr")
    
    # Check Tesseract
    try:
        import pytesseract
        # Try to run tesseract to verify it's actually working
        pytesseract.get_tesseract_version()
        available_providers.append("tesseract")
        if not _availability_logged:
            logging.info("Tesseract is available")
    except Exception as e:
        if not _availability_logged:
            logging.warning(f"Tesseract not available: {e}")
    
    # Check PaddleOCR
    try:
        import paddleocr
        available_providers.append("paddleocr")
        if not _availability_logged:
            logging.info("PaddleOCR is available")
    except ImportError:
        if not _availability_logged:
            logging.warning("PaddleOCR not available - install with: pip install paddlepaddle paddleocr")
    
    # Mark that we've logged availability info (only log once)
    _availability_logged = True
    
    if not available_providers:
        logging.error("No OCR providers available! Please install at least one: EasyOCR, Tesseract, or PaddleOCR")
        raise ImportError("No OCR providers available")
    
    return available_providers

def get_cached_ocr_provider(provider_type: str, advanced_settings: Dict[str, Any]):
    """Get a cached OCR provider instance with smart fallback logic."""
    # Handle auto-selection
    if provider_type == "auto":
        provider_type, auto_gpu_setting = auto_select_ocr_provider()
        # Override GPU setting if auto-selected
        if provider_type == "easyocr":
            advanced_settings = advanced_settings.copy()
            if "easyocr" not in advanced_settings:
                advanced_settings["easyocr"] = {}
            advanced_settings["easyocr"]["use_gpu"] = auto_gpu_setting
        logging.info(f"Auto-selected provider: {provider_type} (GPU: {auto_gpu_setting if provider_type == 'easyocr' else 'N/A'})")
    
    # Check available providers
    available_providers = check_ocr_availability()
    
    # If requested provider is not available, use smart fallback
    if provider_type not in available_providers:
        global _fallback_logged
        fallback_key = f"{provider_type}_fallback"
        
        if "easyocr" in available_providers:
            # Only log this fallback warning once
            if fallback_key not in _fallback_logged:
                logging.warning(f"{provider_type} not available, falling back to EasyOCR")
                _fallback_logged[fallback_key] = True
            provider_type = "easyocr"
        elif "tesseract" in available_providers:
            # Only log this fallback warning once
            if fallback_key not in _fallback_logged:
                logging.warning(f"{provider_type} not available, falling back to Tesseract")
                _fallback_logged[fallback_key] = True
            provider_type = "tesseract"
        else:
            raise ImportError("No OCR providers available")
    
    # Create cache key based on provider type and relevant settings
    cache_key = f"{provider_type}_{hash(str(sorted(advanced_settings.items())))}"
    
    if cache_key not in _ocr_provider_cache:
        logging.debug(f"Creating new {provider_type} OCR provider instance (PID: {os.getpid()})...")
        try:
            if provider_type == "easyocr":
                _ocr_provider_cache[cache_key] = EasyOcrProvider(advanced_settings)
            elif provider_type == "paddleocr":
                _ocr_provider_cache[cache_key] = PaddleOcrProvider(advanced_settings)
            elif provider_type == "hybrid":
                _ocr_provider_cache[cache_key] = HybridOcrProvider(advanced_settings)
            else:  # tesseract
                _ocr_provider_cache[cache_key] = TesseractOcrProvider(advanced_settings)
            logging.info(f"Cached {provider_type} OCR provider created successfully")
        except Exception as e:
            logging.error(f"Failed to create {provider_type} OCR provider: {e}")
            # Final fallback - try the other available provider
            fallback_priority = ["tesseract", "easyocr", "paddleocr"]
            for fallback_provider in fallback_priority:
                if fallback_provider != provider_type and fallback_provider in available_providers:
                    logging.info(f"Trying fallback to {fallback_provider}...")
                    try:
                        if fallback_provider == "easyocr":
                            _ocr_provider_cache[cache_key] = EasyOcrProvider(advanced_settings)
                        elif fallback_provider == "paddleocr":
                            _ocr_provider_cache[cache_key] = PaddleOcrProvider(advanced_settings)
                        else:
                            _ocr_provider_cache[cache_key] = TesseractOcrProvider(advanced_settings)
                        logging.info(f"Successfully fell back to {fallback_provider}")
                        break
                    except Exception as fallback_error:
                        logging.error(f"Fallback to {fallback_provider} also failed: {fallback_error}")
                        continue
            else:
                raise ImportError("All OCR providers failed to initialize")
    else:
        global _cache_usage_count
        _cache_usage_count += 1
        
        # Only log cache usage every 50 times to reduce noise
        if _cache_usage_count % 50 == 1:  # Log on 1st, 51st, 101st usage, etc.
            logging.debug(f"Using cached {provider_type} OCR provider instance (usage count: {_cache_usage_count})")
    
    return _ocr_provider_cache[cache_key]

def preload_ocr_provider(advanced_settings: Dict[str, Any]):
    """Pre-warms the OCR provider to avoid delays on first use."""
    provider_type = advanced_settings.get("ocr_provider", "auto")
    logging.info(f"Pre-loading OCR provider: {provider_type}")
    try:
        get_cached_ocr_provider(provider_type, advanced_settings)
        logging.info("OCR provider pre-loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to pre-load OCR provider: {e}")

def _preprocess_image_with_cv2(image: Image.Image, ocr_config: Dict[str, Any]) -> np.ndarray:
    """
    Performs a standardized set of pre-processing steps on an image using OpenCV.
    Returns a processed image as a NumPy array.
    """
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Scaling
    scale_factor = ocr_config.get("resize_factor", 2)
    if scale_factor > 1:
        img_cv = cv2.resize(img_cv, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 3. Binarization (Adaptive Thresholding)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    return binary

class OcrProvider:
    """Abstract base class for OCR providers."""
    def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
        """
        Extracts text from a specified region of a screenshot.

        This method should be implemented by subclasses to provide specific OCR functionality.
        """
        raise NotImplementedError

class TesseractOcrProvider(OcrProvider):
    """OCR provider using Tesseract."""
    def __init__(self, advanced_settings: Dict[str, Any]):
        self.advanced_settings = advanced_settings
        self._setup_tesseract()

    def _setup_tesseract(self):
        """Sets the path to the Tesseract executable."""
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        tess_path = os.path.join(base_dir, "tesseract", "tesseract.exe")
        
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            logging.info(f"Tesseract path set to: {tess_path}")
        else:
            system_tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(system_tess_path):
                pytesseract.pytesseract.tesseract_cmd = system_tess_path
                logging.info(f"Tesseract path set to system installation: {system_tess_path}")
            else:
                logging.warning("Tesseract executable not found at expected path. Using system default.")

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Applies a series of preprocessing steps to an image to improve OCR accuracy."""
        ocr_config = self.advanced_settings.get("ocr", {})
        processed_array = _preprocess_image_with_cv2(image, ocr_config)
        # Tesseract works with PIL Images
        return Image.fromarray(processed_array)

    def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
        """Crops, preprocesses, and OCRs a region of an image with performance logging."""
        start_time = time.time()
        try:
            cropped = screenshot.crop(region)
            preprocessed = self._preprocess_image_for_ocr(cropped)
            
            # Get Tesseract config options from settings
            tesseract_config = self.advanced_settings.get("tesseract", {})
            config_options = tesseract_config.get("config_options", "--psm 7")
            
            text = pytesseract.image_to_string(preprocessed, config=config_options).strip()
            text = text.replace('|', '').replace('_', ' ')
            text = ' '.join(text.split())
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)
            
            # Simplified OCR corrections
            text = re.sub(r'(\w+)\s+(of|or|on)$', r'\1a', text)
            
            if len(text) < 3:
                # Use fallback PSM mode if primary config didn't work well
                alt_config = tesseract_config.get("fallback_config", "--psm 8")
                alt_text = pytesseract.image_to_string(preprocessed, config=alt_config).strip()
                alt_text = ' '.join(alt_text.replace('|', '').replace('_', ' ').split())
                if len(alt_text) > len(text):
                    text = alt_text
            
            # Log OCR performance
            ocr_time = time.time() - start_time
            if field_name:
                log_ocr_performance(field_name, region, ocr_time, len(text))
            
            return text
        except Exception as e:
            logging.error(f"Error in OCR for region {region}: {e}")
            return ""

class EasyOcrProvider(OcrProvider):
    """OCR provider using EasyOCR for better accuracy."""
    
    def __init__(self, advanced_settings: Dict[str, Any]):
        self.advanced_settings = advanced_settings
        self._setup_easyocr()

    def _setup_easyocr(self):
        """Initialize EasyOCR reader with smart GPU detection."""
        try:
            import easyocr
            import torch
            
            easyocr_config = self.advanced_settings.get("easyocr", {})
            use_gpu_setting = easyocr_config.get("use_gpu", False)  # Default to False
            
            # Smart GPU detection - only use GPU if actually available
            gpu_available = torch.cuda.is_available()
            use_gpu = use_gpu_setting and gpu_available
            
            if use_gpu_setting and not gpu_available:
                logging.info("GPU requested but not available, using CPU for EasyOCR")
            elif not use_gpu_setting:
                logging.info("Using CPU for EasyOCR (GPU disabled in config)")
            else:
                logging.info("Using GPU for EasyOCR")
            
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            logging.info(f"EasyOCR initialized (GPU: {use_gpu})")
            
        except ImportError:
            logging.error("EasyOCR not installed. Install with: pip install easyocr")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def _preprocess_image_for_ocr(self, image: Image.Image) -> np.ndarray:
        """Applies a series of preprocessing steps to an image to improve OCR accuracy for EasyOCR."""
        ocr_config = self.advanced_settings.get("ocr", {})
        # EasyOCR works with NumPy arrays
        return _preprocess_image_with_cv2(image, ocr_config)

    def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
        """Extract text using EasyOCR."""
        start_time = time.time()
        try:
            cropped = screenshot.crop(region)
            preprocessed_img_array = self._preprocess_image_for_ocr(cropped)
            
            # EasyOCR can now use the preprocessed image
            results = self.reader.readtext(preprocessed_img_array)
            
            # Extract text with confidence filtering
            easyocr_config = self.advanced_settings.get("easyocr", {})
            confidence_threshold = easyocr_config.get("confidence_threshold", 0.5)
            
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > confidence_threshold:
                    texts.append(text)
            
            text = ' '.join(texts)
            
            # Apply consistent text cleaning
            text = text.replace('|', '').replace('_', ' ')
            text = ' '.join(text.split())
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)
            text = re.sub(r'(\w+)\s+(of|or|on)$', r'\1a', text)
            
            # Log performance
            ocr_time = time.time() - start_time
            if field_name:
                log_ocr_performance(f"{field_name}_easyocr", region, ocr_time, len(text))
            
            return text
            
        except:
            cropped = screenshot.crop(region)
            img_array = np.array(cropped)
            
            # EasyOCR handles preprocessing internally
            results = self.reader.readtext(img_array)
            
            # Extract text with confidence filtering
            easyocr_config = self.advanced_settings.get("easyocr", {})
            confidence_threshold = easyocr_config.get("confidence_threshold", 0.5)
            
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > confidence_threshold:
                    texts.append(text)
            
            text = ' '.join(texts)
            
            # Apply consistent text cleaning
            text = text.replace('|', '').replace('_', ' ')
            text = ' '.join(text.split())
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)
            text = re.sub(r'(\w+)\s+(of|or|on)$', r'\1a', text)
            
            # Log performance
            ocr_time = time.time() - start_time
            if field_name:
                log_ocr_performance(f"{field_name}_easyocr", region, ocr_time, len(text))
            
            return text
            


class PaddleOcrProvider(OcrProvider):
    """OCR provider using PaddleOCR for comprehensive text recognition."""
    
    def __init__(self, advanced_settings: Dict[str, Any]):
        self.advanced_settings = advanced_settings
        self._setup_paddleocr()

    def _setup_paddleocr(self):
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
            import paddle
            
            paddleocr_config = self.advanced_settings.get("paddleocr", {})
            
            # Suppress PaddlePaddle logging by default
            paddle.disable_static_logging()
            
            config = {
                'use_angle_cls': paddleocr_config.get("use_angle_cls", True),
                'lang': paddleocr_config.get("lang", 'en'),
                'show_log': paddleocr_config.get("show_log", False),
                'use_gpu': paddleocr_config.get("use_gpu", False),  # Default to CPU for compatibility
            }
            
            self.engine = PaddleOCR(**config)
            logging.info("PaddleOCR initialized successfully")
            
        except ImportError:
            logging.error("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def _preprocess_image_for_ocr(self, image: Image.Image) -> np.ndarray:
        """Applies preprocessing steps to improve OCR accuracy for PaddleOCR."""
        ocr_config = self.advanced_settings.get("ocr", {})
        # PaddleOCR works with NumPy arrays
        return _preprocess_image_with_cv2(image, ocr_config)

    def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
        """Extract text using PaddleOCR."""
        start_time = time.time()
        try:
            cropped = screenshot.crop(region)
            preprocessed_img_array = self._preprocess_image_for_ocr(cropped)
            
            # PaddleOCR expects BGR format (OpenCV format)
            result = self.engine.ocr(preprocessed_img_array, cls=True)
            
            paddleocr_config = self.advanced_settings.get("paddleocr", {})
            confidence_threshold = paddleocr_config.get("confidence_threshold", 0.5)
            
            texts = []
            if result and len(result) > 0 and result[0] is not None:
                for line in result[0]:
                    if len(line) >= 2:
                        text = line[1][0]
                        # Ensure confidence is a float for comparison
                        try:
                            confidence = float(line[1][1]) if len(line[1]) > 1 else 1.0
                        except (ValueError, TypeError) as e:
                            logging.debug(f"PaddleOCR: Failed to convert confidence '{line[1][1] if len(line[1]) > 1 else 'N/A'}' to float: {e}")
                            confidence = 1.0  # Default to high confidence if conversion fails
                        
                        if confidence > confidence_threshold:
                            texts.append(text)
            
            text = ' '.join(texts)
            
            # Apply consistent text cleaning
            text = text.replace('|', '').replace('_', ' ')
            text = ' '.join(text.split())
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)
            text = re.sub(r'(\w+)\s+(of|or|on)$', r'\1a', text)
            
            # Log performance
            ocr_time = time.time() - start_time
            if field_name:
                log_ocr_performance(f"{field_name}_paddleocr", region, ocr_time, len(text))
            
            return text
            
        except Exception as e:
            logging.error(f"Error in PaddleOCR for region {region}: {e}")
            return ""


class HybridOcrProvider(OcrProvider):
    """Hybrid OCR provider that tries Tesseract first, then falls back to EasyOCR"""
    
    def __init__(self, advanced_settings: Dict[str, Any]):
        self.advanced_settings = advanced_settings
        self.tesseract_provider = None
        self.easyocr_provider = None
        
        # Initialize Tesseract if available
        try:
            self.tesseract_provider = TesseractOcrProvider(advanced_settings)
            logging.info("HybridOCR: Tesseract initialized successfully")
        except Exception as e:
            logging.warning(f"HybridOCR: Tesseract initialization failed: {e}")
        
        # Initialize EasyOCR if available
        try:
            self.easyocr_provider = EasyOcrProvider(advanced_settings)
            logging.info("HybridOCR: EasyOCR initialized successfully")
        except Exception as e:
            logging.warning(f"HybridOCR: EasyOCR initialization failed: {e}")
        
        if not self.tesseract_provider and not self.easyocr_provider:
            raise RuntimeError("HybridOCR: No OCR providers available")
    
    def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
        """Try Tesseract first, fallback to EasyOCR if Tesseract fails or returns poor results"""
        tesseract_result = ""
        easyocr_result = ""
        
        # Try Tesseract first (faster)
        if self.tesseract_provider:
            try:
                tesseract_result = self.tesseract_provider.get_text_from_region(screenshot, region, field_name)
                
                # Check if Tesseract result is acceptable
                if self._is_result_acceptable(tesseract_result):
                    logging.debug(f"HybridOCR: Using Tesseract result for {field_name}: '{tesseract_result[:50]}...'")
                    return tesseract_result
                else:
                    logging.debug(f"HybridOCR: Tesseract result poor for {field_name}, trying EasyOCR")
            except Exception as e:
                logging.warning(f"HybridOCR: Tesseract failed for {field_name}: {e}")
        
        # Fallback to EasyOCR
        if self.easyocr_provider:
            try:
                easyocr_result = self.easyocr_provider.get_text_from_region(screenshot, region, field_name)
                logging.debug(f"HybridOCR: Using EasyOCR result for {field_name}: '{easyocr_result[:50]}...'")
                return easyocr_result
            except Exception as e:
                logging.error(f"HybridOCR: EasyOCR also failed for {field_name}: {e}")
        
        # Return best available result
        if tesseract_result:
            logging.warning(f"HybridOCR: Returning poor Tesseract result for {field_name} (no EasyOCR available)")
            return tesseract_result
        
        logging.error(f"HybridOCR: All OCR methods failed for {field_name}")
        return ""
    
    def _is_result_acceptable(self, text: str) -> bool:
        """Determine if an OCR result is acceptable or needs fallback"""
        if not text or len(text.strip()) == 0:
            return False
        
        # Check for garbled text (high ratio of special characters)
        clean_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        ratio = clean_chars / total_chars
        
        # If less than 70% of characters are alphanumeric/space, consider it poor
        if ratio < 0.7:
            return False
        
        # Check for obvious OCR artifacts
        artifacts = ['|||', '___', '...', '^^^', '~~~']
        if any(artifact in text for artifact in artifacts):
            return False
        
        return True
