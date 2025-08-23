
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
from typing import Tuple, Dict, Any

from logger_config import log_ocr_performance

# Global cache for OCR providers to avoid reinitialization
_ocr_provider_cache = {}

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
    
    # Decision logic: EasyOCR with GPU > Tesseract > EasyOCR without GPU
    if gpu_available and "easyocr" in available_providers:
        logging.info("Auto-selected EasyOCR with GPU acceleration (best accuracy)")
        return "easyocr", True  # provider, use_gpu
    elif "tesseract" in available_providers:
        logging.info("Auto-selected Tesseract (best speed, no GPU required)")
        return "tesseract", False
    elif "easyocr" in available_providers:
        logging.info("Auto-selected EasyOCR with CPU (GPU not available)")
        return "easyocr", False
    else:
        raise ImportError("No suitable OCR provider available")

def check_ocr_availability():
    """Check which OCR providers are available on the system."""
    available_providers = []
    
    # Check EasyOCR
    try:
        import easyocr
        available_providers.append("easyocr")
        logging.info("EasyOCR is available")
    except ImportError:
        logging.warning("EasyOCR not available - install with: pip install easyocr")
    
    # Check Tesseract
    try:
        import pytesseract
        # Try to run tesseract to verify it's actually working
        pytesseract.get_tesseract_version()
        available_providers.append("tesseract")
        logging.info("Tesseract is available")
    except Exception as e:
        logging.warning(f"Tesseract not available: {e}")
    
    if not available_providers:
        logging.error("No OCR providers available! Please install either EasyOCR (pip install easyocr) or Tesseract")
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
        if "easyocr" in available_providers:
            logging.warning(f"{provider_type} not available, falling back to EasyOCR")
            provider_type = "easyocr"
        elif "tesseract" in available_providers:
            logging.warning(f"{provider_type} not available, falling back to Tesseract")
            provider_type = "tesseract"
        else:
            raise ImportError("No OCR providers available")
    
    # Create cache key based on provider type and relevant settings
    cache_key = f"{provider_type}_{hash(str(sorted(advanced_settings.items())))}"
    
    if cache_key not in _ocr_provider_cache:
        logging.info(f"Creating new {provider_type} OCR provider instance...")
        try:
            if provider_type == "easyocr":
                _ocr_provider_cache[cache_key] = EasyOcrProvider(advanced_settings)
            else:  # tesseract
                _ocr_provider_cache[cache_key] = TesseractOcrProvider(advanced_settings)
            logging.info(f"Cached {provider_type} OCR provider created successfully")
        except Exception as e:
            logging.error(f"Failed to create {provider_type} OCR provider: {e}")
            # Final fallback - try the other available provider
            for fallback_provider in available_providers:
                if fallback_provider != provider_type:
                    logging.info(f"Trying fallback to {fallback_provider}...")
                    try:
                        if fallback_provider == "easyocr":
                            _ocr_provider_cache[cache_key] = EasyOcrProvider(advanced_settings)
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
        logging.info(f"Using cached {provider_type} OCR provider instance")
    
    return _ocr_provider_cache[cache_key]

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
            
