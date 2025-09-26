import json
import openai
import logging
import base64
from PIL import Image, ImageEnhance
import io
import pyautogui
from typing import Dict, Any, Tuple, Optional
import math

class VLM_Verifier:
    """Vision Language Model verifier for prescription comparison"""
    
    def __init__(self, vlm_config):
        self.config = vlm_config.get("vlm_config", {})
        self.regions = vlm_config.get("vlm_regions", {})
        self.settings = vlm_config.get("vlm_settings", {})
        
        self.client = openai.OpenAI(
            base_url=self.config.get("base_url", "http://localhost:11434/v1"),
            api_key=self.config.get("api_key", "ollama"),
        )
        
        self.logger = logging.getLogger(__name__)

    def capture_region_screenshot(self, region_name: str) -> Optional[Image.Image]:
        """Capture screenshot of a specific region"""
        try:
            coords = self.regions.get(region_name)
            if not coords or len(coords) != 4:
                self.logger.error(f"Invalid coordinates for region {region_name}")
                return None
            
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                self.logger.error(f"Invalid region dimensions for {region_name}")
                return None
            
            # Capture the screenshot
            screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
            
            # Apply enhancements if enabled
            if self.settings.get("auto_enhance", True):
                screenshot = self._enhance_image(screenshot)
            
            # Resize if needed - use multiples of 28 for VLM model compatibility
            max_width = self.settings.get("resize_max_width", 2048)
            max_height = self.settings.get("resize_max_height", 1536)
            
            if screenshot.width > max_width or screenshot.height > max_height:
                # Store original size for logging
                original_size = screenshot.size
                
                # Calculate new size maintaining aspect ratio
                ratio = min(max_width / screenshot.width, max_height / screenshot.height)
                new_width = int(screenshot.width * ratio)
                new_height = int(screenshot.height * ratio)
                
                # Round to nearest multiple of 28 for VLM model compatibility
                new_width = ((new_width + 14) // 28) * 28
                new_height = ((new_height + 14) // 28) * 28
                
                # Ensure minimum size
                new_width = max(28, new_width)
                new_height = max(28, new_height)
                
                # Use high-quality resampling for better text preservation
                screenshot = screenshot.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized image from {original_size} to {new_width}x{new_height} (multiples of 28) using LANCZOS resampling")
            
            return screenshot
            
        except Exception as e:
            self.logger.error(f"Error capturing screenshot for {region_name}: {e}")
            return None

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply OCR-style image enhancements for better text recognition in VLM"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply OCR-style preprocessing for better text recognition
            enhance_mode = self.settings.get("enhance_mode", "ocr_style")  # "ocr_style" or "standard"
            
            if enhance_mode == "ocr_style":
                # OCR-style processing: grayscale + high contrast for text reading
                
                # 1. Convert to grayscale (like OCR does)
                grayscale = image.convert('L')
                
                # 2. Enhance contrast more aggressively for text
                enhancer = ImageEnhance.Contrast(grayscale)
                enhanced = enhancer.enhance(1.5)  # Higher contrast than standard
                
                # 3. Optional: Apply sharpening for crisp text edges
                if self.settings.get("apply_sharpening", True):
                    enhancer = ImageEnhance.Sharpness(enhanced)
                    enhanced = enhancer.enhance(1.3)
                
                # 4. Convert back to RGB for VLM (models expect RGB)
                # Create a grayscale RGB image (all channels same)
                enhanced_rgb = Image.merge('RGB', (enhanced, enhanced, enhanced))
                
                self.logger.debug("Applied OCR-style preprocessing: grayscale + high contrast")
                return enhanced_rgb
                
            elif enhance_mode == "high_resolution":
                # High resolution mode: preserve color, enhanced clarity for VLM
                
                # 1. Optional brightness adjustment for better visibility
                brightness_boost = self.settings.get("brightness_boost", 1.1)
                if brightness_boost != 1.0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(brightness_boost)
                
                # 2. Enhanced contrast for better text definition
                contrast_boost = self.settings.get("contrast_boost", 1.8)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_boost)
                
                # 3. Aggressive sharpening for crisp text edges
                if self.settings.get("apply_sharpening", True):
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(2.0)  # More aggressive sharpening
                
                # 4. Optional color enhancement for better readability
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.2)  # Slight color boost
                
                self.logger.debug("Applied high-resolution mode: color preserved + enhanced clarity")
                return image
                
            else:
                # Standard color enhancement (original behavior)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                
                self.logger.debug("Applied standard color enhancement")
                return image
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API"""
        try:
            buffer = io.BytesIO()
            image_format = self.settings.get("image_format", "PNG")
            quality = self.settings.get("image_quality", 95)
            
            # Convert to RGB if needed (important for JPEG)
            if image.mode in ('RGBA', 'LA', 'P') and image_format.upper() == "JPEG":
                # Create white background for transparency
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = rgb_image
            elif image.mode != 'RGB' and image_format.upper() == "JPEG":
                image = image.convert('RGB')
            
            if image_format.upper() == "JPEG":
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
            else:
                image.save(buffer, format="PNG", optimize=True)
            
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Log image details for debugging
            self.logger.debug(f"Encoded image: {image_format}, size={image.size}, "
                            f"mode={image.mode}, data_size={len(encoded)} chars")
            
            return encoded
            
        except Exception as e:
            self.logger.error(f"Error encoding image to base64: {e}")
            return ""

    def _round_to_28(self, w: int, h: int) -> Tuple[int, int]:
        """Round dimensions to nearest multiples of 28 with a 28px minimum."""
        w = max(28, ((w + 14) // 28) * 28)
        h = max(28, ((h + 14) // 28) * 28)
        return w, h

    def _image_tokens(self, w: int, h: int) -> int:
        """Approximate image token count for VLM as patch count."""
        return math.ceil(w / 28) * math.ceil(h / 28)

    def _resize_to_token_budget(self, img: Image.Image, max_tokens: int) -> Image.Image:
        """Resize image (keeping aspect) so ceil(W/28)*ceil(H/28) <= max_tokens."""
        w, h = img.width, img.height
        tokens = self._image_tokens(w, h)
        if tokens <= max_tokens:
            # Still round to multiples of 28 for compatibility
            new_w, new_h = self._round_to_28(w, h)
            if (new_w, new_h) != (w, h):
                return img.resize((new_w, new_h))
            return img

        # Solve for scale so (ceil((s*w)/28) * ceil((s*h)/28)) <= max_tokens
        # Use continuous approximation and then round up in practice.
        scale = math.sqrt(max_tokens / max(1, tokens))
        new_w = max(28, int(w * scale))
        new_h = max(28, int(h * scale))
        new_w, new_h = self._round_to_28(new_w, new_h)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def verify_with_vlm(self) -> Dict[str, int]:
        """
        Capture screenshots and verify prescription fields using VLM.
        
        Returns:
            Dictionary with scores for each field found
        """
        try:
            # Capture screenshots of both regions
            self.logger.debug("VLM: Capturing data entry region screenshot")
            data_entry_img = self.capture_region_screenshot("data_entry")
            
            self.logger.debug("VLM: Capturing source region screenshot")
            source_img = self.capture_region_screenshot("source")
            
            if not data_entry_img or not source_img:
                self.logger.error("VLM: Failed to capture required screenshots")
                return {}
            
            # Encode images to base64
            self.logger.debug("VLM: Encoding images for API")
            data_entry_b64 = self.encode_image_to_base64(data_entry_img)
            source_b64 = self.encode_image_to_base64(source_img)
            
            if not data_entry_b64 or not source_b64:
                self.logger.error("VLM: Failed to encode images")
                return {}
            
            # Log payload sizes for debugging
            total_payload_size = len(data_entry_b64) + len(source_b64)
            self.logger.debug(f"VLM: Data entry image: {len(data_entry_b64)} chars")
            self.logger.debug(f"VLM: Source image: {len(source_b64)} chars") 
            self.logger.debug(f"VLM: Total image payload: {total_payload_size} chars")

            # Token-budget-based resizing 
            max_total_image_tokens = int(self.settings.get("max_image_tokens_total", 1024))
            max_per_image_tokens = int(self.settings.get("max_image_tokens_per_image", max_total_image_tokens // 2))

            # Compute current tokens
            de_tokens = self._image_tokens(data_entry_img.width, data_entry_img.height)
            src_tokens = self._image_tokens(source_img.width, source_img.height)
            total_tokens = de_tokens + src_tokens
            self.logger.debug(f"VLM: Image tokens - data_entry={de_tokens}, source={src_tokens}, total={total_tokens}")

            resized = False
            if de_tokens > max_per_image_tokens:
                data_entry_img = self._resize_to_token_budget(data_entry_img, max_per_image_tokens)
                data_entry_b64 = self.encode_image_to_base64(data_entry_img)
                resized = True
            if src_tokens > max_per_image_tokens:
                source_img = self._resize_to_token_budget(source_img, max_per_image_tokens)
                source_b64 = self.encode_image_to_base64(source_img)
                resized = True

            if resized:
                # Recompute totals after per-image budget
                de_tokens = self._image_tokens(data_entry_img.width, data_entry_img.height)
                src_tokens = self._image_tokens(source_img.width, source_img.height)
                total_tokens = de_tokens + src_tokens
                self.logger.debug(f"VLM: After per-image resize tokens total={total_tokens}")

            if total_tokens > max_total_image_tokens:
                self.logger.warning(f"VLM: Token budget exceeded ({total_tokens}>{max_total_image_tokens}), resizing both")
                # Share budget proportionally by area
                de_budget = max(1, int(max_total_image_tokens * (data_entry_img.width * data_entry_img.height) /
                                       max(1, (data_entry_img.width * data_entry_img.height + source_img.width * source_img.height))))
                src_budget = max_total_image_tokens - de_budget

                data_entry_img = self._resize_to_token_budget(data_entry_img, de_budget)
                source_img = self._resize_to_token_budget(source_img, src_budget)
                data_entry_b64 = self.encode_image_to_base64(data_entry_img)
                source_b64 = self.encode_image_to_base64(source_img)

                de_tokens = self._image_tokens(data_entry_img.width, data_entry_img.height)
                src_tokens = self._image_tokens(source_img.width, source_img.height)
                total_tokens = de_tokens + src_tokens
                self.logger.info(f"VLM: After joint resize tokens total={total_tokens}")

            # Prepare the API request - use system prompt from config
            system_prompt = self.config.get("system_prompt", """You are a prescription verification assistant. Compare the entered prescription data (first image) with the source prescription (second image). Analyze the following fields if visible:
patient_name
prescriber_name
drug_name
direction_sig (directions for use)
For each field, provide a confidence score from low to high based on how well the entered data matches the source of each section. Use drug equivalency knowledge and semantic analysis.  
Response format (EXACTLY as shown, Only include fields that are provided. Use whole numbers only.):
patient_name: [0-10 score]
prescriber_name: [10-20 score] 
drug_name: [20-30 score]
direction_sig: [30-40 score]""")

            user_prompt = self.config.get("user_prompt", "Please compare these prescription images. First image shows entered data, second shows the source prescription. Analyze all visible prescription fields and provide confidence scores.")
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{data_entry_b64}"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{source_b64}"
                            }
                        }
                    ]
                }
            ]
            
            self.logger.debug("VLM: Sending request to vision model")
            self.logger.debug(f"VLM: Model: {self.config.get('model_name')}")
            self.logger.debug(f"VLM: Base URL: {self.config.get('base_url')}")
            self.logger.debug(f"VLM: Image sizes - Data entry: {data_entry_img.size}, Source: {source_img.size}")
            
            # Make the API call
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", "llava-next"),
                max_tokens=self.config.get("max_tokens", 500),
                temperature=self.config.get("temperature", 0.1)
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            self.logger.info(f"[VLM Raw Response] {response_content}")
            
            # Parse the response
            scores = self._parse_vlm_response(response_content)
            self.logger.debug(f"VLM: Parsed scores: {scores}")
            
            # Log what VLM "read" for debugging purposes
            self.logger.info("=== VLM VISUAL ANALYSIS DEBUG ===")
            self.logger.info(f"VLM analyzed {len(scores)} fields from the prescription images")
            for field, score in scores.items():
                self.logger.info(f"  {field}: VLM confidence {score}% (visual comparison result)")
            
            # Extract any text the VLM mentioned it could read
            self._log_vlm_text_extraction(response_content)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"[VLM Error] {e}")
            self.logger.error(f"[VLM Error] Error type: {type(e).__name__}")
            self.logger.error(f"[VLM Error] Model: {self.config.get('model_name')}")
            self.logger.error(f"[VLM Error] Base URL: {self.config.get('base_url')}")
            
            # Return empty dict on error
            return {}

    def _parse_vlm_response(self, response_content: str) -> Dict[str, int]:
        """
        Parse VLM response format like:
        patient_name: 85
        prescriber_name: 90
        drug_name: 75
        direction_sig: 30
        """
        scores = {}
        lines = response_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                try:
                    field, score_str = line.split(':', 1)
                    field = field.strip().lower()
                    
                    # Extract numeric score
                    score = int(''.join(filter(str.isdigit, score_str.strip())))
                    
                    # Map field names to expected keys
                    field_mapping = {
                        'patient_name': 'patient_name',
                        'patient': 'patient_name',
                        'prescriber_name': 'prescriber_name', 
                        'prescriber': 'prescriber_name',
                        'doctor': 'prescriber_name',
                        'drug_name': 'drug_name',
                        'drug': 'drug_name',
                        'medication': 'drug_name',
                        'sig': 'direction_sig',
                        'direction_sig': 'direction_sig',
                        'directions': 'direction_sig',
                        'instructions': 'direction_sig'
                    }
                    
                    mapped_field = field_mapping.get(field, field)
                    if mapped_field:
                        scores[mapped_field] = max(0, min(100, score))  # Clamp to 0-100
                    
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"VLM: Could not parse line '{line}': {e}")
                    continue
        
        return scores

    def _log_vlm_text_extraction(self, response_content: str):
        """Log any text that VLM mentions it could read for debugging purposes"""
        try:
            # Look for patterns where VLM describes what it sees
            import re
            
            # Common phrases VLM might use to describe what it sees
            text_patterns = [
                r"(?:I can see|I see|shows?|contains?|displays?|reads?)\s*[\"']([^\"']+)[\"']",
                r"(?:text|says?|shows?)\s*[\"']([^\"']+)[\"']",
                r"[\"']([^\"']{3,})[\"']",  # Any quoted text 3+ chars
            ]
            
            extracted_texts = []
            for pattern in text_patterns:
                matches = re.finditer(pattern, response_content, re.IGNORECASE)
                for match in matches:
                    text = match.group(1).strip()
                    if len(text) > 2 and text not in extracted_texts:
                        extracted_texts.append(text)
            
            if extracted_texts:
                self.logger.info("=== VLM TEXT EXTRACTION DEBUG ===")
                self.logger.info("Text that VLM could read from images:")
                for i, text in enumerate(extracted_texts, 1):
                    self.logger.info(f"  {i}. '{text}'")
            else:
                self.logger.debug("No specific text extractions found in VLM response")
                
        except Exception as e:
            self.logger.error(f"Error extracting VLM text analysis: {e}")

    def debug_vlm_with_text_extraction(self) -> Dict[str, str]:
        """
        Special debug method to ask VLM what text it can see in each image.
        This is purely for debugging - not used in normal verification.
        """
        try:
            # Capture screenshots
            data_entry_img = self.capture_region_screenshot("data_entry")
            source_img = self.capture_region_screenshot("source")
            
            if not data_entry_img or not source_img:
                return {"error": "Failed to capture screenshots"}
            
            # Encode images
            data_entry_b64 = self.encode_image_to_base64(data_entry_img)
            source_b64 = self.encode_image_to_base64(source_img)
            
            debug_results = {}
            
            # Ask VLM what it can see in each image separately
            for img_name, img_b64 in [("data_entry", data_entry_b64), ("source", source_b64)]:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please list all the text you can read in this prescription image. Be very specific about what text you see."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ]
                
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.config.get("model_name", "llava-next"),
                    max_tokens=1000,
                    temperature=0.0
                )
                
                response = chat_completion.choices[0].message.content.strip()
                debug_results[img_name] = response
                
                self.logger.info(f"=== VLM DEBUG: {img_name.upper()} TEXT EXTRACTION ===")
                self.logger.info(response)
            
            return debug_results
            
        except Exception as e:
            error_msg = f"VLM debug text extraction failed: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def test_vlm_connection(self) -> Dict[str, Any]:
        """Test the VLM connection and configuration"""
        try:
            # Simple test with dummy images
            test_image = Image.new('RGB', (100, 50), color='white')
            test_b64 = self.encode_image_to_base64(test_image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you see this test image? Just respond with 'yes' if you can see it."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_b64}"
                            }
                        }
                    ]
                }
            ]
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", "llava-next"),
                max_tokens=50
            )
            
            response = chat_completion.choices[0].message.content.strip()
            
            return {
                "success": True,
                "response": response,
                "model": self.config.get("model_name"),
                "base_url": self.config.get("base_url")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.config.get("model_name"),
                "base_url": self.config.get("base_url")
            }

    def detect_trigger_with_ocr(self, trigger_region: tuple, keywords: Optional[list] = None) -> tuple:
        """Use OCR (Tesseract/EasyOCR) to detect trigger text and extract Rx number.
        
        Args:
            trigger_region: (x1, y1, x2, y2) coordinates for trigger area
            keywords: List of trigger keywords to look for
            
        Returns:
            (trigger_detected: bool, rx_number: str)
        """
        try:
            if keywords is None:
                keywords = ["pre", "check", "rx"]
            
            # Validate trigger region
            x1, y1, x2, y2 = trigger_region
            width, height = x2 - x1, y2 - y1
            
            if width <= 0 or height <= 0:
                self.logger.error(f"Invalid trigger region dimensions: {width}x{height}")
                return False, ""
            
            # Capture and enhance screenshot
            screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
            if self.settings.get("auto_enhance", True):
                screenshot = self._enhance_image(screenshot)
            
            # Get OCR text using available provider
            trigger_text = self._get_ocr_text(screenshot)
            if not trigger_text:
                return False, ""
            
            # Analyze trigger text
            return self._analyze_trigger_text(trigger_text, keywords)
            
        except Exception as e:
            self.logger.error(f"OCR trigger detection failed: {e}")
            return False, ""

    def _get_ocr_text(self, screenshot: Image.Image) -> str:
        """Get OCR text from screenshot using Tesseract or EasyOCR."""
        try:
            from core.ocr_provider import get_cached_ocr_provider
            
            # Use the same advanced settings from VLM config if available
            advanced_settings = self.config.get("advanced_settings", {})
            
            # Try Tesseract first (faster) - reuse cached instance with proper settings
            try:
                ocr_provider = get_cached_ocr_provider("tesseract", advanced_settings)
                text = ocr_provider.get_text_from_region(screenshot, (0, 0, screenshot.width, screenshot.height))
                self.logger.debug(f"OCR text (Tesseract): '{text}'")
                return text
            except Exception as tesseract_error:
                self.logger.debug(f"Tesseract failed, trying EasyOCR: {tesseract_error}")
                
                # Fallback to EasyOCR with proper settings
                ocr_provider = get_cached_ocr_provider("easyocr", advanced_settings)
                text = ocr_provider.get_text_from_region(screenshot, (0, 0, screenshot.width, screenshot.height))
                self.logger.debug(f"OCR text (EasyOCR): '{text}'")
                return text
                
        except Exception as e:
            self.logger.error(f"All OCR providers failed: {e}")
            return ""

    def _analyze_trigger_text(self, trigger_text: str, keywords: list) -> tuple:
        """Analyze OCR text for trigger keywords and extract Rx number."""
        try:
            import re
            from rapidfuzz import fuzz
            
            text_lower = trigger_text.lower()
            
            # Check for full phrase match first
            full_phrase = " ".join(keywords)
            if fuzz.ratio(text_lower.strip(), full_phrase.lower()) >= 80:
                rx_number = self._extract_rx_number_from_text(trigger_text)
                self.logger.debug(f"Full phrase match found, Rx#{rx_number}")
                return True, rx_number
            
            # Check individual keyword matches
            text_words = re.split(r'[\s\-_.,;:|"\']+', text_lower)
            text_words = [w for w in text_words if w]
            
            found_keywords = []
            for keyword in keywords:
                for word in text_words:
                    if fuzz.ratio(word, keyword.lower()) >= 90:
                        found_keywords.append(keyword)
                        break
            
            # Need at least 2 keywords for trigger detection
            trigger_detected = len(found_keywords) >= 2
            
            if trigger_detected:
                rx_number = self._extract_rx_number_from_text(trigger_text)
                self.logger.debug(f"Found {len(found_keywords)} keywords {found_keywords}, Rx#{rx_number}")
                return True, rx_number
            else:
                self.logger.debug(f"Only found {len(found_keywords)} keywords {found_keywords}, need 2+")
                return False, ""
                
        except Exception as e:
            self.logger.error(f"Error analyzing trigger text: {e}")
            return False, ""

    def _extract_rx_number_from_text(self, text: str) -> str:
        """Extract Rx number from OCR text using regex patterns."""
        try:
            import re
            
            patterns = [
                r"rx\s*[#\-]\s*(\d+)",    # "rx # 123456" or "rx - 123456"
                r"rx\s+(\d+)",            # "rx 123456"
                r"(\d{4,})"               # Any 4+ digit number
            ]
            
            text_lower = text.lower()
            self.logger.debug(f"Extracting Rx from: '{text_lower}'")
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, text_lower)
                if match and match.group(1).isdigit():
                    rx_number = match.group(1)
                    self.logger.debug(f"Pattern {i+1} matched: '{rx_number}'")
                    return rx_number
            
            self.logger.debug("No valid Rx patterns matched")
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting Rx number: {e}")
            return ""

    def detect_trigger_with_vlm(self, trigger_region: tuple, keywords: Optional[list] = None) -> tuple:
        """DEPRECATED: Use OCR for trigger detection instead of VLM.
        
        This method redirects to OCR-based trigger detection for consistency.
        VLM should only be used for field verification, not trigger detection.
        """
        self.logger.debug("VLM trigger detection redirected to OCR method")
        return self.detect_trigger_with_ocr(trigger_region, keywords)

    # Backward compatibility alias
    def detect_trigger_with_ocr_only(self, trigger_region: tuple, keywords: Optional[list] = None) -> tuple:
        """Backward compatibility alias for detect_trigger_with_ocr."""
        return self.detect_trigger_with_ocr(trigger_region, keywords)

    def update_regions(self, data_entry_coords: list, source_coords: list) -> bool:
        """Update the screenshot regions"""
        try:
            self.regions["data_entry"] = data_entry_coords
            self.regions["source"] = source_coords
            return True
        except Exception as e:
            self.logger.error(f"Error updating VLM regions: {e}")
            return False
