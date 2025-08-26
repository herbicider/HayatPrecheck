import json
import openai
import logging
import base64
from PIL import Image, ImageEnhance
import io
import pyautogui
from typing import Dict, Any, Tuple, Optional

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
            
            # Resize if needed
            max_width = self.settings.get("resize_max_width", 1024)
            max_height = self.settings.get("resize_max_height", 768)
            
            if screenshot.width > max_width or screenshot.height > max_height:
                screenshot.thumbnail((max_width, max_height), Image.Lanczos)
            
            return screenshot
            
        except Exception as e:
            self.logger.error(f"Error capturing screenshot for {region_name}: {e}")
            return None

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements for better OCR/VLM performance"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
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
            
            if image_format.upper() == "JPEG":
                image.save(buffer, format="JPEG", quality=quality)
            else:
                image.save(buffer, format="PNG")
                
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error encoding image to base64: {e}")
            return ""

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
            
            # Prepare the API request
            system_prompt = self.config.get("system_prompt", "")
            user_prompt = self.config.get("user_prompt", "Compare these prescription images.")
            
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
            
            # Make the API call
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", "llava-next"),
                max_tokens=self.config.get("max_tokens", 500),
                temperature=self.config.get("temperature", 0.1)
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            self.logger.info(f"[VLM Response] {response_content}")
            
            # Parse the response
            scores = self._parse_vlm_response(response_content)
            self.logger.debug(f"VLM: Parsed scores: {scores}")
            
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

    def detect_trigger_with_vlm(self, trigger_region: tuple, keywords: Optional[list] = None) -> tuple:
        """Use VLM to detect trigger text and extract Rx number
        
        Args:
            trigger_region: (x1, y1, x2, y2) coordinates for trigger area
            keywords: List of trigger keywords to look for
            
        Returns:
            (trigger_detected: bool, rx_number: str)
        """
        try:
            if keywords is None:
                keywords = ["pre", "check", "rx"]
            
            # Capture trigger region screenshot
            x1, y1, x2, y2 = trigger_region
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                self.logger.error(f"Invalid trigger region dimensions")
                return False, ""
            
            # Capture the screenshot
            screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
            
            # Enhance image for better VLM processing
            if self.settings.get("auto_enhance", True):
                screenshot = self._enhance_image(screenshot)
            
            # Convert to base64
            image_b64 = self.encode_image_to_base64(screenshot)
            
            # Create trigger detection prompt
            keywords_text = "', '".join(keywords)
            trigger_prompt = f"""
Analyze this image and determine if it contains pharmacy verification trigger text.

Look for text patterns that indicate a prescription verification process, such as:
- Words like: '{keywords_text}'
- Pharmacy verification interface elements
- Prescription checking prompts
- Rx numbers or prescription identifiers

Respond with EXACTLY this format:
TRIGGER: YES/NO
RX_NUMBER: [number if found, or NONE]

Examples:
- If you see "pre-check rx 123456": TRIGGER: YES, RX_NUMBER: 123456
- If you see "check prescription": TRIGGER: YES, RX_NUMBER: NONE
- If you see unrelated text: TRIGGER: NO, RX_NUMBER: NONE
"""

            # Call VLM
            response = self.client.chat.completions.create(
                model=self.config.get("model_name", ""),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a pharmacy verification assistant. Analyze images to detect prescription verification triggers and extract Rx numbers."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": trigger_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            self.logger.info(f"VLM trigger response: {content}")
            
            # Extract trigger status and Rx number
            trigger_detected = False
            rx_number = ""
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('TRIGGER:'):
                    trigger_detected = 'YES' in line.upper()
                elif line.startswith('RX_NUMBER:'):
                    rx_part = line.split(':', 1)[1].strip()
                    if rx_part != 'NONE':
                        rx_number = rx_part
            
            self.logger.info(f"VLM trigger detection: trigger={trigger_detected}, rx={rx_number}")
            return trigger_detected, rx_number
            
        except Exception as e:
            self.logger.error(f"VLM trigger detection failed: {e}")
            return False, ""

    def update_regions(self, data_entry_coords: list, source_coords: list) -> bool:
        """Update the screenshot regions"""
        try:
            self.regions["data_entry"] = data_entry_coords
            self.regions["source"] = source_coords
            return True
        except Exception as e:
            self.logger.error(f"Error updating VLM regions: {e}")
            return False
