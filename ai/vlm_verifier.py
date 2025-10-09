import json
import openai
import logging
import base64
from PIL import Image, ImageEnhance
import io
import pyautogui
from typing import Dict, Any, Tuple, Optional
import math
import numpy as np

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
                # Save original for debugging if enabled
                if self.settings.get("debug_save_images", False):
                    self._save_debug_image(screenshot, f"{region_name}_original")
                
                screenshot = self._enhance_image(screenshot)
                
                # Save processed version for debugging
                if self.settings.get("debug_save_images", False):
                    self._save_debug_image(screenshot, f"{region_name}_processed")
            
            # Resize if needed - use multiples of 28 for VLM model compatibility
            max_width = self.settings.get("resize_max_width", 1920)
            max_height = self.settings.get("resize_max_height", 1440)
            
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

    def _save_debug_image(self, image: Image.Image, prefix: str):
        """Save debug images to help troubleshoot preprocessing"""
        try:
            import os
            from datetime import datetime
            
            debug_dir = "vlm_debug_images"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(debug_dir, filename)
            
            image.save(filepath, "PNG")
            self.logger.debug(f"Saved debug image: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save debug image: {e}")

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply pharmacy-specific image enhancements for better VLM text recognition"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply pharmacy-specific preprocessing
            enhance_mode = self.settings.get("enhance_mode", "pharmacy_optimized")
            
            if enhance_mode == "pharmacy_optimized":
                # Pharmacy-specific processing: handles highlighted text and prescription formats
                
                # 1. Normalize highlighting (yellow/blue backgrounds to white)
                image_array = np.array(image)
                
                # Detect yellow highlighting (common in pharmacy software)
                # Yellow pixels: high R+G, low B
                yellow_mask = (
                    (image_array[:, :, 0] > 200) &  # High red
                    (image_array[:, :, 1] > 200) &  # High green
                    (image_array[:, :, 2] < 150)    # Lower blue
                )
                
                # Detect blue highlighting - improved detection for pharmacy software
                # Multiple blue detection strategies for better coverage
                
                # Strategy 1: Traditional blue (low R+G, high B)
                blue_mask_1 = (
                    (image_array[:, :, 0] < 160) &  # Increased red threshold
                    (image_array[:, :, 1] < 160) &  # Increased green threshold  
                    (image_array[:, :, 2] > 150)    # Lowered blue threshold for lighter blues
                )
                
                # Strategy 2: Light blue highlighting (common in Windows apps)
                # Look for pixels where blue channel significantly exceeds red+green
                blue_dominance = image_array[:, :, 2] > (image_array[:, :, 0] + image_array[:, :, 1]) * 0.8
                blue_mask_2 = blue_dominance & (image_array[:, :, 2] > 120)
                
                # Strategy 3: HSV-based blue detection for better color space handling
                # Convert small regions to HSV for more accurate blue detection
                try:
                    from colorsys import rgb_to_hsv
                    # Vectorized HSV conversion for blue hue detection (180-260 degrees)
                    normalized_rgb = image_array.astype(float) / 255.0
                    # Simple blue hue check: blue dominant and not grayscale
                    is_colorful = (np.max(image_array, axis=2) - np.min(image_array, axis=2)) > 30
                    blue_hue_mask = (image_array[:, :, 2] > image_array[:, :, 0]) & \
                                   (image_array[:, :, 2] > image_array[:, :, 1]) & \
                                   is_colorful & (image_array[:, :, 2] > 100)
                except:
                    blue_hue_mask = np.zeros_like(blue_mask_1)
                
                # Combine all blue detection strategies
                blue_mask = blue_mask_1 | blue_mask_2 | blue_hue_mask
                
                # Smart highlighting removal with text preservation
                # Instead of pure white, use a light background that preserves text contrast
                
                if np.any(yellow_mask):
                    # For yellow highlighting, use very light yellow-white to preserve text readability
                    image_array[yellow_mask] = [252, 252, 240]  # Very light cream
                    yellow_count = np.sum(yellow_mask)
                    self.logger.debug(f"Converted {yellow_count} yellow highlighted pixels to light background")
                    
                if np.any(blue_mask):
                    # For blue highlighting, detect if text is likely dark or light
                    blue_pixels = image_array[blue_mask]
                    
                    # Check if there are dark pixels within blue regions (likely text)
                    dark_text_in_blue = np.any((blue_pixels[:, 0] < 100) & 
                                              (blue_pixels[:, 1] < 100) & 
                                              (blue_pixels[:, 2] < 150))
                    
                    if dark_text_in_blue:
                        # Keep some blue tint but make it very light for contrast
                        image_array[blue_mask] = [240, 248, 255]  # Alice blue (very light)
                        self.logger.debug("Converted blue highlighting to light blue background (dark text detected)")
                    else:
                        # No dark text detected, safe to use white
                        image_array[blue_mask] = [255, 255, 255]  # White
                        self.logger.debug("Converted blue highlighting to white background (no dark text)")
                    
                    blue_count = np.sum(blue_mask)
                    self.logger.debug(f"Processed {blue_count} blue highlighted pixels")
                
                image = Image.fromarray(image_array)
                
                # 2. Moderate text contrast enhancement for readability
                contrast_boost = self.settings.get("contrast_boost", 1.8)  # More conservative
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_boost)
                
                # 3. Gentle sharpening that preserves text quality
                if self.settings.get("apply_sharpening", True):
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(1.5)  # Much gentler sharpening
                
                # 4. Minimal brightness adjustment
                brightness_boost = self.settings.get("brightness_boost", 1.1)
                if brightness_boost != 1.0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(brightness_boost)
                
                self.logger.debug("Applied pharmacy-optimized preprocessing: highlighting normalized + text enhanced")
                return image
                
            elif enhance_mode == "ocr_style":
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
                
            elif enhance_mode == "rtx_optimized":
                # RTX 5080 optimized mode: maximum quality processing for high-end GPU
                
                # 1. Apply highlighting removal first (same as pharmacy_optimized)
                image_array = np.array(image)
                
                # Enhanced blue detection for RTX processing
                blue_mask_1 = (
                    (image_array[:, :, 0] < 160) & 
                    (image_array[:, :, 1] < 160) &  
                    (image_array[:, :, 2] > 150)
                )
                
                blue_dominance = image_array[:, :, 2] > (image_array[:, :, 0] + image_array[:, :, 1]) * 0.8
                blue_mask_2 = blue_dominance & (image_array[:, :, 2] > 120)
                
                is_colorful = (np.max(image_array, axis=2) - np.min(image_array, axis=2)) > 30
                blue_hue_mask = (image_array[:, :, 2] > image_array[:, :, 0]) & \
                               (image_array[:, :, 2] > image_array[:, :, 1]) & \
                               is_colorful & (image_array[:, :, 2] > 100)
                
                blue_mask = blue_mask_1 | blue_mask_2 | blue_hue_mask
                
                # Yellow highlighting
                yellow_mask = (
                    (image_array[:, :, 0] > 200) & 
                    (image_array[:, :, 1] > 200) & 
                    (image_array[:, :, 2] < 150)
                )
                
                # Smart highlighting removal
                if np.any(yellow_mask):
                    image_array[yellow_mask] = [252, 252, 240]
                if np.any(blue_mask):
                    blue_pixels = image_array[blue_mask]
                    dark_text_in_blue = np.any((blue_pixels[:, 0] < 100) & 
                                              (blue_pixels[:, 1] < 100) & 
                                              (blue_pixels[:, 2] < 150))
                    if dark_text_in_blue:
                        image_array[blue_mask] = [240, 248, 255]
                    else:
                        image_array[blue_mask] = [255, 255, 255]
                
                image = Image.fromarray(image_array)
                
                # 2. RTX-level contrast enhancement (higher than standard)
                contrast_boost = self.settings.get("contrast_boost", 2.5)  # Higher for RTX
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_boost)
                
                # 3. RTX-level sharpening (very aggressive for fine text)
                if self.settings.get("apply_sharpening", True):
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(3.0)  # Very aggressive for RTX processing
                
                # 4. Fine brightness control for RTX displays
                brightness_boost = self.settings.get("brightness_boost", 1.2)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness_boost)
                
                # 5. Color saturation for better VLM field recognition
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.3)
                
                self.logger.debug("Applied RTX-optimized mode: maximum quality + highlighting removal")
                return image
                
            elif enhance_mode == "minimal":
                # Minimal processing - just basic highlighting removal
                
                image_array = np.array(image)
                
                # Only basic blue highlighting detection (most conservative)
                blue_mask = (
                    (image_array[:, :, 0] < 140) &  # Very restrictive
                    (image_array[:, :, 1] < 140) &  
                    (image_array[:, :, 2] > 170)    # Only strong blues
                )
                
                # Convert blue highlights to very light background
                if np.any(blue_mask):
                    image_array[blue_mask] = [245, 250, 255]  # Very light blue-white
                    self.logger.debug(f"Minimal: Converted {np.sum(blue_mask)} blue pixels")
                
                image = Image.fromarray(image_array)
                
                # Very gentle contrast - barely noticeable
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)  # Minimal contrast boost
                
                self.logger.debug("Applied minimal enhancement - preserves original quality")
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
        FIVE-STEP VLM verification with DOB support:
        1. Send LEFT image  
        2. Extract from LEFT (patient name, DOB, prescriber, drug, directions)
        3. Send RIGHT image
        4. Extract ALL info from RIGHT 
        5. Search LEFT info in RIGHT and score matches
        
        Returns:
            Dictionary with scores for each field found
        """
        try:
            # Step 1: Capture LEFT image (data entry)
            self.logger.debug("VLM: Step 1 - Capturing LEFT image (data entry)")
            data_entry_img = self.capture_region_screenshot("data_entry")
            if not data_entry_img:
                self.logger.error("VLM: Failed to capture LEFT image")
                return {}
            
            # Step 2: Extract from LEFT image only 
            self.logger.debug("VLM: Step 2 - Extracting from LEFT image")
            data_entry_img = self._prepare_single_image(data_entry_img)
            left_data = self._extract_from_left_image(data_entry_img)
            if not left_data:
                self.logger.error("VLM: Failed to extract from LEFT image")
                return {}
            
            # Step 3: Capture RIGHT image (source prescription)
            self.logger.debug("VLM: Step 3 - Capturing RIGHT image (source)")
            source_img = self.capture_region_screenshot("source")
            if not source_img:
                self.logger.error("VLM: Failed to capture RIGHT image")
                return {}
            
            # Step 4: Extract ALL info from RIGHT image
            self.logger.debug("VLM: Step 4 - Extracting ALL info from RIGHT image")
            source_img = self._prepare_single_image(source_img)
            right_data = self._extract_from_right_image(source_img)
            if not right_data:
                self.logger.error("VLM: Failed to extract from RIGHT image")
                return {}
            
            # Step 5: Search LEFT info in RIGHT and score matches
            self.logger.debug("VLM: Step 5 - Searching and scoring matches")
            scores = self._search_and_score_matches(left_data, right_data)
            
            # Log final results
            self.logger.info("=== VLM VERIFICATION RESULTS ===")
            for field, score in scores.items():
                if score == 0:
                    status = "FAILED"
                elif score == 50:
                    status = "REVIEW"
                elif score == 95:
                    status = "ACCEPTABLE"
                elif score == 100:
                    status = "PERFECT"
                else:
                    status = "UNKNOWN"
                
                self.logger.info(f"  {field}: {score}% ({status})")
            
            return scores
            
        except Exception as e:
            self.logger.error(f"[VLM Error] {e}")
            self.logger.error(f"[VLM Error] Error type: {type(e).__name__}")
            self.logger.error(f"[VLM Error] Model: {self.config.get('model_name')}")
            self.logger.error(f"[VLM Error] Base URL: {self.config.get('base_url')}")
            
                        # Return empty dict on error
            return {}
    
    def _prepare_single_image(self, image: Image.Image) -> str:
        """Prepare and encode a single image for API"""
        try:
            # Apply token budget (resize if needed)
            max_width = self.settings.get("resize_max_width", 1920)
            max_height = self.settings.get("resize_max_height", 1440)
            
            # Resize if image is too large
            if image.width > max_width or image.height > max_height:
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized image to {image.width}x{image.height}")
            
            # Apply enhancements
            image = self._enhance_image(image)
            
            # Save debug image if enabled
            if self.settings.get("debug_save_images", False):
                self._save_debug_image(image, "processed")
            
            # Encode to base64
            return self.encode_image_to_base64(image)
            
        except Exception as e:
            self.logger.error(f"Failed to prepare image: {e}")
            return ""
    
    def _extract_from_left_image(self, data_entry_b64: str) -> Dict[str, str]:
        """Extract fields from LEFT image using Step 1 prompts"""
        try:
            import json
            
            # Use customizable prompts from config
            vlm_model_config = self.config.get("vlm_config", {})
            system_prompt = vlm_model_config.get("step1_system_prompt", "Extract from LEFT image only.")
            user_prompt = vlm_model_config.get("step1_user_prompt", "Extract prescription fields from the LEFT image.")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data_entry_b64}"}}
                    ]
                }
            ]
            
            # Call OpenAI API
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", ""),
                max_tokens=self.config.get("max_tokens", 500),
                temperature=self.config.get("temperature", 0.0)
            )
            
            response_content = chat_completion.choices[0].message.content or ""
            self.logger.debug(f"LEFT extraction raw: {response_content}")
            
            # Validate response before parsing
            if not self._validate_vlm_response(response_content, "Step1"):
                self.logger.error("Step1: Response validation failed")
                return {}
            
            # Use robust JSON parsing
            extracted = self._robust_json_parse(response_content, "Step1")
            if not extracted:
                self.logger.error("Step1: Failed to extract valid JSON, using empty dict")
                return {}
            
            self.logger.debug(f"LEFT extracted: {extracted}")
            return extracted
            
        except Exception as e:
            self.logger.error(f"Failed to extract from LEFT image: {e}")
            return {}
    
    def _extract_from_right_image(self, source_b64: str) -> Dict[str, str]:
        """Extract ALL info from RIGHT image using Step 2 prompts"""
        try:
            import json
            
            # Use customizable prompts from config
            vlm_model_config = self.config.get("vlm_config", {})
            system_prompt = vlm_model_config.get("step2_system_prompt", "Extract from RIGHT image.")
            user_prompt = vlm_model_config.get("step2_user_prompt", "Extract all prescription information from the RIGHT image.")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{source_b64}"}}
                    ]
                }
            ]
            
            # Call OpenAI API
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", ""),
                max_tokens=self.config.get("max_tokens", 500),
                temperature=self.config.get("temperature", 0.0)
            )
            
            response_content = chat_completion.choices[0].message.content or ""
            self.logger.debug(f"RIGHT extraction raw: {response_content}")
            
            # Validate response before parsing
            if not self._validate_vlm_response(response_content, "Step2"):
                self.logger.error("Step2: Response validation failed")
                return {}
            
            # Use robust JSON parsing
            extracted = self._robust_json_parse(response_content, "Step2")
            if not extracted:
                self.logger.error("Step2: Failed to extract valid JSON, using empty dict")
                return {}
            
            self.logger.debug(f"RIGHT extracted: {extracted}")
            return extracted
            
        except Exception as e:
            self.logger.error(f"Failed to extract from RIGHT image: {e}")
            return {}
    
    def _validate_vlm_response(self, response_content: str, context: str = "VLM response") -> bool:
        """
        Pre-validate VLM response before attempting JSON parsing.
        
        Args:
            response_content: Raw response from VLM
            context: Context for logging
            
        Returns:
            True if response looks valid for JSON parsing, False otherwise
        """
        try:
            if not response_content or not response_content.strip():
                self.logger.error(f"{context}: Empty response received")
                return False
                
            content = response_content.strip()
            
            # Check for minimum requirements
            if len(content) < 10:
                self.logger.error(f"{context}: Response too short ({len(content)} chars)")
                return False
            
            # Check for JSON structure indicators
            has_braces = "{" in content and "}" in content
            has_quotes = '"' in content
            
            if not has_braces:
                self.logger.error(f"{context}: No JSON braces found in response")
                return False
                
            if not has_quotes:
                self.logger.warning(f"{context}: No quotes found - may not be valid JSON")
            
            # Check for common VLM failure patterns
            failure_patterns = [
                "i can't",
                "i cannot", 
                "unable to",
                "not possible",
                "cannot see",
                "can't see",
                "no text",
                "not visible",
                "unclear"
            ]
            
            content_lower = content.lower()
            for pattern in failure_patterns:
                if pattern in content_lower:
                    self.logger.error(f"{context}: VLM failure pattern detected: '{pattern}'")
                    return False
            
            # Check for excessive explanatory text (should be mostly JSON)
            # Count non-JSON characters vs total
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_portion = content[json_start:json_end]
                non_json_chars = len(content) - len(json_portion)
                
                # If more than 50% is non-JSON content, it's probably problematic
                if non_json_chars > len(json_portion):
                    self.logger.warning(f"{context}: Response has excessive non-JSON content ({non_json_chars} vs {len(json_portion)})")
            
            self.logger.debug(f"{context}: Response validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"{context}: Response validation error: {e}")
            return False

    def _robust_json_parse(self, response_content: str, context: str = "VLM response") -> Dict[str, Any]:
        """
        Robust JSON parser with multiple fallback strategies for handling malformed VLM responses.
        
        Args:
            response_content: Raw response from VLM
            context: Context for logging (e.g., "step1", "step2", "step3")
            
        Returns:
            Parsed JSON dictionary or empty dict on failure
        """
        import json
        import re
        
        try:
            if not response_content or not response_content.strip():
                self.logger.error(f"{context}: Empty response received")
                return {}
            
            original_content = response_content
            self.logger.debug(f"{context}: Original response: {response_content[:200]}...")
            
            # Strategy 1: Try direct JSON parsing first
            try:
                cleaned = response_content.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract from markdown code blocks
            if "```json" in response_content:
                try:
                    json_str = response_content.split("```json")[1].split("```")[0].strip()
                    self.logger.debug(f"{context}: Extracted from markdown: {json_str}")
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    pass
            
            # Strategy 3: Find JSON object boundaries
            try:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_content[start:end]
                    self.logger.debug(f"{context}: Extracted by boundaries: {json_str}")
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Strategy 4: Clean common formatting issues and retry
            try:
                # Remove common markdown artifacts
                cleaned = response_content.replace("```json", "").replace("```", "").strip()
                
                # Remove text before first { and after last }
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = cleaned[start:end]
                    
                    # Fix common JSON issues
                    json_str = json_str.replace("'", '"')  # Single to double quotes
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                    
                    self.logger.debug(f"{context}: Cleaned JSON: {json_str}")
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Strategy 5: Try to extract key-value pairs with regex
            try:
                self.logger.warning(f"{context}: Attempting regex extraction as last resort")
                pairs = {}
                
                # Look for "field": value patterns
                patterns = [
                    r'"([^"]+)":\s*(\d+)',  # "field": 123
                    r'([a-zA-Z_]+):\s*(\d+)',  # field: 123
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response_content, re.IGNORECASE)
                    for field, value in matches:
                        field = field.strip().lower().replace(" ", "_")
                        if field in ['patient_name', 'patient_dob', 'prescriber_name', 'drug_name', 'direction_sig']:
                            try:
                                pairs[field] = int(value)
                            except ValueError:
                                pass
                
                if pairs:
                    self.logger.info(f"{context}: Regex extraction found: {pairs}")
                    return pairs
                    
            except Exception as regex_error:
                self.logger.error(f"{context}: Regex extraction failed: {regex_error}")
            
            # All strategies failed
            self.logger.error(f"{context}: All JSON parsing strategies failed")
            self.logger.error(f"{context}: Original response: {original_content}")
            return {}
            
        except json.JSONDecodeError as e:
            self.logger.error(f"{context}: JSON decode error: {e}")
            self.logger.debug(f"{context}: Problematic content: {response_content}")
            return {}
        except Exception as e:
            self.logger.error(f"{context}: JSON parsing error: {e}")
            self.logger.debug(f"{context}: Problematic content: {response_content}")
            return {}

    def _search_and_score_matches(self, left_data: Dict[str, str], right_data: Dict[str, str]) -> Dict[str, int]:
        """Search LEFT info in RIGHT data and score matches using Step 3 prompts"""
        response_content = ""
        try:
            import json
            import re
            
            # Use customizable prompts from config
            vlm_model_config = self.config.get("vlm_config", {})
            system_prompt = vlm_model_config.get("step3_system_prompt", "Search and score matches.")
            user_prompt = vlm_model_config.get("step3_user_prompt", "Search LEFT data in RIGHT data and score.")
            
            # Format the prompt with actual data
            formatted_prompt = user_prompt.format(
                left_data=json.dumps(left_data, indent=2),
                right_data=json.dumps(right_data, indent=2)
            )
            
            # Enhanced system prompt to enforce JSON format
            enhanced_system_prompt = f"""{system_prompt}

CRITICAL: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no extra text.
Your response must be exactly in this format:
{{
  "patient_name": 100,
  "patient_dob": 100,
  "prescriber_name": 95,
  "drug_name": 100,
  "direction_sig": 100
}}

Use only these exact scores: 0, 50, 95, or 100. Nothing else."""
            
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Call OpenAI API
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", ""),
                max_tokens=self.config.get("max_tokens", 200),
                temperature=self.config.get("temperature", 0.0)
            )
            
            response_content = chat_completion.choices[0].message.content or ""
            self.logger.debug(f"Step3 raw response: {response_content}")
            
            # Validate response before parsing
            if not self._validate_vlm_response(response_content, "Step3"):
                self.logger.error("Step3: Response validation failed")
                return {"patient_name": 0, "patient_dob": 0, "prescriber_name": 0, "drug_name": 0, "direction_sig": 0}
            
            # Use robust JSON parsing
            scores_data = self._robust_json_parse(response_content, "Step3")
            
            if not scores_data:
                self.logger.error("Step3: No valid JSON data extracted, using fallback")
                return {"patient_name": 0, "patient_dob": 0, "prescriber_name": 0, "drug_name": 0, "direction_sig": 0}
            
            # Validate and normalize scores with strict missing data checks
            validated_scores = {}
            expected_fields = ["patient_name", "patient_dob", "prescriber_name", "drug_name", "direction_sig"]
            
            for field in expected_fields:
                score = scores_data.get(field, 0)
                
                # Ensure score is numeric
                try:
                    score = int(float(score))
                except (ValueError, TypeError):
                    self.logger.warning(f"Step3: Invalid score '{score}' for {field}, defaulting to 0")
                    score = 0
                
                # STRICT MISSING DATA VALIDATION
                # Check if we're trying to score a field that's missing in the right data
                right_field_value = right_data.get(field)
                left_field_value = left_data.get(field)
                
                if score > 0:  # Only validate if we're giving a positive score
                    # Check for explicitly null or missing right data
                    if (right_field_value is None or 
                        right_field_value == "null" or 
                        right_field_value == "" or
                        str(right_field_value).lower().strip() in ["null", "none", "n/a", "not available", "missing"]):
                        
                        # But left side has data
                        if (left_field_value and 
                            left_field_value != "null" and 
                            left_field_value != "" and
                            str(left_field_value).lower().strip() not in ["null", "none", "n/a"]):
                            
                            self.logger.warning(f"Step3: {field} - LEFT has data '{left_field_value}' but RIGHT is missing/null '{right_field_value}'. Forcing score to 0")
                            score = 0
                
                # Validate score range and round to nearest valid value
                valid_scores = [0, 50, 95, 100]
                if score not in valid_scores:
                    original_score = score
                    if score < 25:
                        score = 0
                    elif score < 72:
                        score = 50
                    elif score < 97:
                        score = 95
                    else:
                        score = 100
                    self.logger.info(f"Step3: Rounded score {original_score} -> {score} for {field}")
                
                validated_scores[field] = score
            
            self.logger.debug(f"Step3: Final validated scores: {validated_scores}")
            return validated_scores
            
        except Exception as e:
            self.logger.error(f"Failed to search and score matches: {e}")
            self.logger.error(f"Step3: Raw response was: {response_content}")
            return {"patient_name": 0, "patient_dob": 0, "prescriber_name": 0, "drug_name": 0, "direction_sig": 0}

    def _parse_vlm_response(self, response_content: str) -> Dict[str, int]:
        """
        Parse VLM JSON response format:
        {
          "left": {
            "patient_name": "text from left",
            "prescriber_name": "text from left",
            "drug_name": "text from left",
            "direction_sig": "text from left"
          },
          "right": {
            "patient_name": "text from right",
            "prescriber_name": "text from right", 
            "drug_name": "text from right",
            "direction_sig": "text from right"
          },
          "scores": {
            "patient_name": 100,
            "prescriber_name": 95,
            "drug_name": 100,
            "direction_sig": 100
          }
        }
        """
        scores = {}
        
        try:
            import json
            
            # Clean the response - remove any markdown code blocks if present
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            data = json.loads(cleaned_response)
            
            # Log extraction details for debugging
            self.logger.debug("=== VLM JSON EXTRACTION ===")
            if "left" in data:
                for field, value in data["left"].items():
                    self.logger.debug(f"LEFT {field}: {value}")
            
            if "right" in data:
                for field, value in data["right"].items():
                    self.logger.debug(f"RIGHT {field}: {value}")
            
            # Extract and validate scores
            if "scores" in data:
                for field, score in data["scores"].items():
                    # Validate score
                    valid_scores = [0, 50, 95, 100]
                    if score not in valid_scores:
                        self.logger.warning(f"VLM: Invalid score {score} for field '{field}'. Valid scores: {valid_scores}")
                        # Round to nearest valid score
                        if score < 25:
                            score = 0
                        elif score < 72:
                            score = 50
                        elif score < 97:
                            score = 95
                        else:
                            score = 100
                        self.logger.info(f"VLM: Rounded score to {score} for field '{field}'")
                    
                    # Map field names to expected keys
                    field_mapping = {
                        'patient_name': 'patient_name',
                        'prescriber_name': 'prescriber_name',
                        'drug_name': 'drug_name',
                        'direction_sig': 'direction_sig'
                    }
                    
                    mapped_field = field_mapping.get(field, field)
                    if mapped_field in ['patient_name', 'prescriber_name', 'drug_name', 'direction_sig']:
                        scores[mapped_field] = score
                        self.logger.debug(f"VLM: {mapped_field}: {score}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"VLM: Failed to parse JSON response: {e}")
            self.logger.debug(f"VLM: Raw response was: {response_content[:200]}...")
            return {}
        except Exception as e:
            self.logger.error(f"VLM: Error processing JSON response: {e}")
            return {}
        
        # Log final results (minimal for production)
        self.logger.info("=== VLM VERIFICATION RESULTS ===")
        for field, score in scores.items():
            if score == 0:
                status = "FAILED"
            elif score == 50:
                status = "REVIEW"
            elif score == 95:
                status = "ACCEPTABLE"
            elif score == 100:
                status = "PERFECT"
            else:
                status = "UNKNOWN"
            
            self.logger.info(f"  {field}: {score}% ({status})")
        
        return scores

    def _prepare_images_for_api(self, data_entry_img: Image.Image, source_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply token budget and resizing to both images"""
        # Token-budget-based resizing
        max_total_image_tokens = int(self.settings.get("max_image_tokens_total", 3072))
        max_per_image_tokens = int(self.settings.get("max_image_tokens_per_image", max_total_image_tokens // 2))

        # Compute current tokens
        de_tokens = self._image_tokens(data_entry_img.width, data_entry_img.height)
        src_tokens = self._image_tokens(source_img.width, source_img.height)
        total_tokens = de_tokens + src_tokens
        self.logger.debug(f"VLM: Image tokens - data_entry={de_tokens}, source={src_tokens}, total={total_tokens}")

        resized = False
        if de_tokens > max_per_image_tokens:
            data_entry_img = self._resize_to_token_budget(data_entry_img, max_per_image_tokens)
            resized = True
        if src_tokens > max_per_image_tokens:
            source_img = self._resize_to_token_budget(source_img, max_per_image_tokens)
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

            de_tokens = self._image_tokens(data_entry_img.width, data_entry_img.height)
            src_tokens = self._image_tokens(source_img.width, source_img.height)
            total_tokens = de_tokens + src_tokens
            self.logger.info(f"VLM: After joint resize tokens total={total_tokens}")

        return data_entry_img, source_img

    def _extract_from_left_image(self, data_entry_b64: str) -> Dict[str, str]:
        """Step 1: Extract fields from LEFT image only"""
        try:
            # Create focused prompt for LEFT extraction only
            extraction_prompt = """Extract prescription fields from this pharmacy data entry image.

This image shows what pharmacy staff ENTERED into their system. The fields are color-coded:
- GREEN background = patient name
- YELLOW background = prescriber name  
- BLUE background = drug name and directions

Find the actual text in each color area and respond with ONLY this JSON:
{
  "patient_name": "actual patient name from image",
  "prescriber_name": "actual prescriber name from image", 
  "drug_name": "actual drug name from image",
  "direction_sig": "actual directions from image"
}"""

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": extraction_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{data_entry_b64}"
                            }
                        }
                    ]
                }
            ]

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", "llava-next"),
                max_tokens=500,
                temperature=0.0
            )

            response_content = chat_completion.choices[0].message.content.strip()
            self.logger.debug(f"LEFT extraction raw: {response_content}")

            # Parse JSON response
            import json
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            left_data = json.loads(cleaned_response)
            self.logger.debug(f"LEFT extracted: {left_data}")
            return left_data

        except Exception as e:
            self.logger.error(f"Failed to extract from left image: {e}")
            return {}

    def _extract_from_right_image(self, source_b64: str) -> Dict[str, str]:
        """Step 2: Extract fields from RIGHT image only"""
        try:
            # Create focused prompt for RIGHT extraction only
            extraction_prompt = """Extract prescription fields from this original prescription document.

This image shows the ORIGINAL prescription from the doctor/source. Find these fields:
- Patient name (look for labels like "Patient:", "Name:", "Patient Name:")
- Prescriber name (look for labels like "Prescriber:", "Dr.:", "Issued by:", "Written by:")
- Drug name (look for labels like "Drug:", "Medication:", "Rx:", medication names)
- Directions (look for dosing instructions, sig, "Take", "Use", etc.)

Find the actual text for each field and respond with ONLY this JSON:
{
  "patient_name": "actual patient name from prescription",
  "prescriber_name": "actual prescriber name from prescription",
  "drug_name": "actual drug name from prescription", 
  "direction_sig": "actual directions from prescription"
}"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": extraction_prompt
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

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", "llava-next"),
                max_tokens=500,
                temperature=0.0
            )

            response_content = chat_completion.choices[0].message.content.strip()
            self.logger.debug(f"RIGHT extraction raw: {response_content}")

            # Parse JSON response
            import json
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            right_data = json.loads(cleaned_response)
            self.logger.debug(f"RIGHT extracted: {right_data}")
            return right_data

        except Exception as e:
            self.logger.error(f"Failed to extract from right image: {e}")
            return {}

    def _compare_and_score(self, left_extractions: Dict[str, str], right_extractions: Dict[str, str]) -> Dict[str, int]:
        """Step 3: Compare extractions and apply scoring with name normalization"""
        try:
            import json
            
            comparison_prompt = f"""Compare these prescription field extractions and score them.

LEFT (entered data): {json.dumps(left_extractions, indent=2)}

RIGHT (original prescription): {json.dumps(right_extractions, indent=2)}

Apply name normalization rules:
- Patient names: "Last, First" = "First Last" (100 score if same person)
- Prescriber names: Ignore titles (Dr., MD, NP, DO) (100 score if same person)

Score each field:
- 0 = extraction failed or no match
- 50 = different person/drug (needs review)
- 95 = similar/equivalent match
- 100 = exact match or normalized match

Respond with ONLY this JSON:
{{
  "patient_name": 100,
  "prescriber_name": 95,
  "drug_name": 100,
  "direction_sig": 100
}}"""

            messages = [
                {
                    "role": "user",
                    "content": comparison_prompt
                }
            ]

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.get("model_name", "llava-next"),
                max_tokens=200,
                temperature=0.0
            )

            response_content = chat_completion.choices[0].message.content.strip()
            self.logger.debug(f"Comparison raw: {response_content}")

            # Parse JSON response
            import json
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            scores_data = json.loads(cleaned_response)
            
            # Validate scores
            validated_scores = {}
            for field, score in scores_data.items():
                valid_scores = [0, 50, 95, 100]
                if score not in valid_scores:
                    self.logger.warning(f"Invalid score {score} for {field}, rounding to nearest valid")
                    if score < 25:
                        score = 0
                    elif score < 72:
                        score = 50
                    elif score < 97:
                        score = 95
                    else:
                        score = 100
                
                if field in ['patient_name', 'prescriber_name', 'drug_name', 'direction_sig']:
                    validated_scores[field] = score

            self.logger.debug(f"Final scores: {validated_scores}")
            return validated_scores

        except Exception as e:
            self.logger.error(f"Failed to compare and score: {e}")
            return {}

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
