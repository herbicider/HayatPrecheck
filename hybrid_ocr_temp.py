
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
