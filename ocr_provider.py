
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
        
        width, height = image.size
        new_size = (width * ocr_config.get("resize_factor", 2), height * ocr_config.get("resize_factor", 2))
        resized = image.resize(new_size, Resampling.LANCZOS)
        
        gray = resized.convert('L')
        
        contrast = ImageEnhance.Contrast(gray).enhance(ocr_config.get("contrast_factor", 1.5))
        brightness = ImageEnhance.Brightness(contrast).enhance(ocr_config.get("brightness_factor", 1.2))
        
        try:
            img_array = np.array(brightness)
            threshold_value = np.mean(img_array) + ocr_config.get("threshold_mean_offset", -20)
            binary_array = np.where(img_array > threshold_value, 255, 0).astype(np.uint8)
            return Image.fromarray(binary_array, mode='L')
        except ImportError:
            fallback_threshold = ocr_config.get("fallback_threshold", 120)
            return brightness.point(lambda x: 255 if x > fallback_threshold else 0, mode='L')

    def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
        """Crops, preprocesses, and OCRs a region of an image with performance logging."""
        start_time = time.time()
        try:
            cropped = screenshot.crop(region)
            preprocessed = self._preprocess_image_for_ocr(cropped)
            
            text = pytesseract.image_to_string(preprocessed, config="--psm 7").strip()
            text = text.replace('|', '').replace('_', ' ')
            text = ' '.join(text.split())
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)
            
            # Simplified OCR corrections
            text = re.sub(r'(\w+)\s+(of|or|on)$', r'\1a', text)
            
            if len(text) < 3:
                alt_text = pytesseract.image_to_string(preprocessed, config="--psm 8").strip()
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

# Example of how to add a new provider
# class PaddleOcrProvider(OcrProvider):
#     def __init__(self, advanced_settings: Dict[str, Any]):
#         # Initialize PaddleOCR here
#         pass
#
#     def get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_name: str = "") -> str:
#         # Implement PaddleOCR logic here
#         pass
