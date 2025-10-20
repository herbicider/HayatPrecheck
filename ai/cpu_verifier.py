
import json
import openai
import logging

class AI_Verifier:
    def __init__(self, config):
        # Support both old format (ai_config key) and new format (llm_config key)
        if isinstance(config, dict) and "llm_config" in config:
            self.config = config.get("llm_config", {})
        elif isinstance(config, dict) and "ai_config" in config:
            self.config = config.get("ai_config", {})
        else:
            # Direct config object or fallback
            self.config = config if isinstance(config, dict) else {}
        self.client = openai.OpenAI(
            base_url=self.config.get("base_url"),
            api_key=self.config.get("api_key"),
        )

    def _robust_json_parse(self, response_content, context="LLM response"):
        """
        Robust JSON parser with multiple fallback strategies for handling malformed LLM responses.
        
        Args:
            response_content: Raw response from LLM
            context: Context for logging
            
        Returns:
            Parsed JSON dictionary or empty dict on failure
        """
        import re
        
        try:
            if not response_content or not response_content.strip():
                logging.error(f"{context}: Empty response received")
                return {}
            
            original_content = response_content
            logging.debug(f"{context}: Original response: {response_content[:200]}...")
            
            # Strategy 1: Try direct JSON parsing first
            try:
                cleaned = response_content.strip()
                parsed_json = json.loads(cleaned)
                # Normalize field names if they have _score suffix
                return self._normalize_field_names(parsed_json)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract from markdown code blocks
            if "```json" in response_content:
                try:
                    json_str = response_content.split("```json")[1].split("```")[0].strip()
                    logging.debug(f"{context}: Extracted from markdown: {json_str}")
                    parsed_json = json.loads(json_str)
                    return self._normalize_field_names(parsed_json)
                except (json.JSONDecodeError, IndexError):
                    pass
            
            # Strategy 3: Find JSON object boundaries
            try:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_content[start:end]
                    logging.debug(f"{context}: Extracted by boundaries: {json_str}")
                    parsed_json = json.loads(json_str)
                    return self._normalize_field_names(parsed_json)
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
                    
                    logging.debug(f"{context}: Cleaned JSON: {json_str}")
                    parsed_json = json.loads(json_str)
                    return self._normalize_field_names(parsed_json)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Strategy 5: Try to extract key-value pairs with regex (fallback to legacy format)
            try:
                logging.warning(f"{context}: Attempting legacy format parsing as last resort")
                pairs = {}
                
                # Look for field: value patterns (legacy format)
                patterns = [
                    r'([a-zA-Z_]+):\s*(\d+)',  # field: 123
                    r'"([^"]+)":\s*(\d+)',     # "field": 123
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response_content, re.IGNORECASE)
                    for field, value in matches:
                        field = field.strip().lower().replace(" ", "_")
                        
                        # Remove _score suffix if present
                        if field.endswith("_score"):
                            field = field[:-6]
                        
                        # Map common field variations
                        field_mapping = {
                            'patient_name': 'patient_name',
                            'prescriber_name': 'prescriber_name',
                            'drug_name': 'drug_name',
                            'sig': 'direction_sig',
                            'direction_sig': 'direction_sig',
                            'directions': 'direction_sig'
                        }
                        mapped_field = field_mapping.get(field, field)
                        if mapped_field in ['patient_name', 'prescriber_name', 'drug_name', 'direction_sig']:
                            try:
                                pairs[mapped_field] = int(value)
                            except ValueError:
                                pass
                
                if pairs:
                    logging.info(f"{context}: Legacy format extraction found: {pairs}")
                    return pairs
                    
            except Exception as regex_error:
                logging.error(f"{context}: Legacy format extraction failed: {regex_error}")
            
            # All strategies failed
            logging.error(f"{context}: All JSON parsing strategies failed")
            logging.error(f"{context}: Original response: {original_content}")
            return {}
            
        except json.JSONDecodeError as e:
            logging.error(f"{context}: JSON decode error: {e}")
            logging.debug(f"{context}: Problematic content: {response_content}")
            return {}
        except Exception as e:
            logging.error(f"{context}: JSON parsing error: {e}")
            logging.debug(f"{context}: Problematic content: {response_content}")
            return {}

    def _normalize_field_names(self, parsed_json):
        """
        Normalize field names by removing _score suffix and mapping to expected field names.
        
        This handles LLM responses that use field names like "patient_name_score" instead of "patient_name".
        
        Args:
            parsed_json: Dictionary with field names that may have _score suffix
            
        Returns:
            Dictionary with normalized field names
        """
        if not isinstance(parsed_json, dict):
            return parsed_json
        
        normalized = {}
        
        # Field name mapping to handle various LLM response formats
        field_mapping = {
            "patient_name_score": "patient_name",
            "prescriber_name_score": "prescriber_name", 
            "drug_name_score": "drug_name",
            "directions_score": "direction_sig",
            "direction_sig_score": "direction_sig",
            "sig_score": "direction_sig",
            # Also handle without _score suffix (direct mapping)
            "patient_name": "patient_name",
            "prescriber_name": "prescriber_name",
            "drug_name": "drug_name",
            "direction_sig": "direction_sig",
            "directions": "direction_sig",
            "sig": "direction_sig",
            # Handle some other variations that might appear
            "address_score": "address",
            "dob_score": "dob", 
            "quantity_score": "quantity",
            "refills_score": "refills"
        }
        
        for key, value in parsed_json.items():
            # Try direct mapping first
            normalized_key = field_mapping.get(key)
            
            if normalized_key:
                normalized[normalized_key] = value
                logging.debug(f"LLM: Mapped field '{key}' -> '{normalized_key}' with score {value}")
            else:
                # Try removing _score suffix for unmapped fields
                if key.endswith("_score"):
                    base_key = key[:-6]  # Remove "_score"
                    normalized[base_key] = value
                    logging.debug(f"LLM: Normalized field '{key}' -> '{base_key}' with score {value}")
                else:
                    # Keep as-is for unknown fields
                    normalized[key] = value
                    logging.debug(f"LLM: Kept field '{key}' as-is with score {value}")
        
        logging.info(f"LLM: Field normalization - Input fields: {list(parsed_json.keys())} -> Output fields: {list(normalized.keys())}")
        return normalized

    def verify_with_ai(self, field_to_verify, ocr_results):
        """
        Verify prescription fields using AI and return structured scores.
        
        Args:
            field_to_verify: The specific field being verified (for backwards compatibility)
            ocr_results: Dictionary of field data with (entered, source) tuples
            
        Returns:
            Dictionary with scores for each field, or single score if legacy format
        """
        system_prompt = self.config.get("system_prompt")
        user_prompt_template = self.config.get("user_prompt")
        
        # Prepare placeholders for template
        placeholders = {"field_to_verify": field_to_verify}
        for field, (entered, source) in ocr_results.items():
            placeholders[f"{field}_entered"] = entered
            placeholders[f"{field}_source"] = source
            
        user_prompt = user_prompt_template.format(**placeholders)

        # Log the batch request with detailed field info
        available_fields = list(ocr_results.keys())
        logging.debug(f"AI batch request for available fields: {available_fields}")
        
        # Debug: Log the actual data being sent to AI
        for field, (entered, source) in ocr_results.items():
            logging.debug(f"AI input - {field}: entered='{entered}' | source='{source}'")
        
        # Check for problematic empty fields
        empty_fields = [field for field, (entered, source) in ocr_results.items() if not entered.strip() or not source.strip()]
        if empty_fields:
            logging.warning(f"AI verification called with empty fields: {empty_fields}")
            for field in empty_fields:
                entered, source = ocr_results[field]
                logging.warning(f"Empty field {field}: entered='{entered}' (len={len(entered)}) | source='{source}' (len={len(source)})")
        
        logging.debug(f"AI user prompt: {user_prompt}")

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.config.get("model_name"),
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            
            # Log only the LLM response content for debugging
            logging.info(f"[LLM Response] {response_content}")
            
            # Use robust JSON parsing (handles both JSON and legacy formats)
            scores = self._robust_json_parse(response_content, "LLM")
            
            if scores:
                logging.info(f"LLM: Parsed scores from response: {scores}")
                # Validate scores against empty fields with strict missing data checks
                validated_scores = self._validate_ai_scores_strict(scores, ocr_results)
                logging.info(f"LLM: Validated scores: {validated_scores}")
                return validated_scores
            
            # Complete fallback - return 0 for all fields
            logging.error("LLM: All parsing strategies failed, returning zeros")
            error_scores = {}
            for field in ocr_results.keys():
                error_scores[field] = 0
            return error_scores
        
        except Exception as e:
            logging.error(f"[LLM Error] An error occurred during AI verification: {e}")
            logging.error(f"[LLM Error] Error type: {type(e).__name__}")
            logging.error(f"[LLM Error] Model: {self.config.get('model_name')}")
            logging.error(f"[LLM Error] Base URL: {self.config.get('base_url')}")
            print(f"An error occurred during AI verification: {e}")
            # Return empty dict for all fields that were requested
            error_scores = {}
            for field in ocr_results.keys():
                error_scores[field] = 0
            logging.info(f"[LLM Error] Returning zero scores for all fields: {error_scores}")
            return error_scores



    def _validate_ai_scores_strict(self, scores, ocr_results):
        """Enhanced validation with strict missing data checks like VLM verifier."""
        validated_scores = {}
        
        for field, score in scores.items():
            if field in ocr_results:
                entered, source = ocr_results[field]
                
                # STRICT MISSING DATA VALIDATION (same logic as VLM)
                if score > 0:  # Only validate if we're giving a positive score
                    # Check for missing entered data
                    if not entered.strip():
                        logging.warning(f"LLM: {field} - Empty entered data but got score {score}. Forcing to 0")
                        score = 0
                    # Check for missing source data  
                    elif not source.strip():
                        logging.warning(f"LLM: {field} - Empty source data but got score {score}. Forcing to 0")
                        score = 0
                    # Check for explicit null/missing indicators
                    elif (str(source).lower().strip() in ["null", "none", "n/a", "not available", "missing"] or
                          str(entered).lower().strip() in ["null", "none", "n/a", "not available", "missing"]):
                        logging.warning(f"LLM: {field} - Null/missing indicator detected. Entered: '{entered}', Source: '{source}'. Forcing to 0")
                        score = 0
                
                # Validate score range
                score = max(0, min(100, int(score)))
                validated_scores[field] = score
            else:
                # Field not in input data, but still validate range
                score = max(0, min(100, int(score)))
                validated_scores[field] = score
                
        return validated_scores

    def _validate_ai_scores(self, scores, ocr_results):
        """Legacy validation method - kept for backward compatibility."""
        validated_scores = {}
        
        for field, score in scores.items():
            if field in ocr_results:
                entered, source = ocr_results[field]
                
                # Check for empty entered field - should get very low score
                if not entered.strip():
                    if score > 20:  # AI gave high score to empty field
                        logging.warning(f"AI gave high score ({score}) to empty entered field '{field}'. Reducing to 0.")
                        validated_scores[field] = 0
                    else:
                        validated_scores[field] = score
                # Check for empty source field - should get very low score  
                elif not source.strip():
                    if score > 20:
                        logging.warning(f"AI gave high score ({score}) to empty source field '{field}'. Reducing to 0.")
                        validated_scores[field] = 0
                    else:
                        validated_scores[field] = score
                else:
                    # Both fields have content, trust AI score
                    validated_scores[field] = score
            else:
                # Field not in input data
                validated_scores[field] = score
                
        return validated_scores


