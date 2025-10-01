
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
            
            # Try to parse structured response (new format)
            scores = self._parse_structured_response(response_content)
            if scores:
                # Validate scores against empty fields
                validated_scores = self._validate_ai_scores(scores, ocr_results)
                return validated_scores
            
            # Fallback to legacy single score extraction
            legacy_score = self._extract_legacy_score(response_content)
            validated_legacy = self._validate_single_score(legacy_score, field_to_verify, ocr_results)
            return {field_to_verify: validated_legacy}
        
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

    def _parse_structured_response(self, response_content):
        """
        Parse structured AI response format like:
        patient_name: 85
        prescriber_name: 90
        drug_name: 75
        sig: 30
        """
        scores = {}
        lines = response_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                try:
                    field, score_str = line.split(':', 1)
                    field = field.strip().lower()
                    score = int(''.join(filter(str.isdigit, score_str.strip())))
                    
                    # Map field names to expected keys
                    field_mapping = {
                        'patient_name': 'patient_name',
                        'prescriber_name': 'prescriber_name', 
                        'drug_name': 'drug_name',
                        'sig': 'direction_sig',
                        'direction_sig': 'direction_sig',
                        'directions': 'direction_sig'
                    }
                    
                    mapped_field = field_mapping.get(field, field)
                    scores[mapped_field] = score
                    
                except (ValueError, IndexError):
                    continue
        
        return scores if scores else None

    def _extract_legacy_score(self, response_content):
        """Extract single score from legacy response format."""
        try:
            return int(''.join(filter(str.isdigit, response_content)))
        except ValueError:
            return 0

    def _validate_ai_scores(self, scores, ocr_results):
        """Validate AI scores against input data to prevent unrealistic scores for empty fields."""
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

    def _validate_single_score(self, score, field_name, ocr_results):
        """Validate single score for legacy format."""
        if field_name in ocr_results:
            entered, source = ocr_results[field_name]
            
            # Check for empty fields
            if not entered.strip() or not source.strip():
                if score > 20:
                    logging.warning(f"AI gave high score ({score}) to empty field '{field_name}'. Reducing to 0.")
                    return 0
                    
        return score
