from typing import Any
import numpy as np
import requests
import json
import os
from openai import OpenAI

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP


class GrokTranslation(BaseLLMTranslation):
    """Translation engine using Grok AI from X."""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.api_key = None
        self.api_base_url = "https://api.x.ai/v1"
        self.temperature = 0.3
        self.max_tokens = 5000
        self.supports_images = False
        self.client = None
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Grok translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: Grok model name
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Grok AI'))
        self.api_key = credentials.get('api_key', '')
        self.model = MODEL_MAP.get(self.model_name, 'grok-beta')
        
        # Khởi tạo OpenAI client với API key và base URL của X.AI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url
        )
        
        # Các mô hình có khả năng xử lý hình ảnh
        vision_models = ["grok-2-vision-1212", "grok-vision-beta"]
        self.supports_images = self.model in vision_models
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using X.AI's OpenAI-compatible API.
        
        Args:
            user_prompt: Text prompt from user
            system_prompt: System instructions
            image: Image as numpy array
            
        Returns:
            Translated text
        """
        try:
            messages = []
            
            # Thêm system prompt
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
            
            # Thêm user message với hình ảnh nếu có
            if self.supports_images and self.img_as_llm_input and image is not None:
                encoded_image, mime_type = self.encode_image(image)
                image_url = f"data:{mime_type};base64,{encoded_image}"
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": user_prompt
                })
            
            # Gọi API với thư viện OpenAI
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            error_msg = f"X.AI API request failed: {str(e)}"
            raise RuntimeError(error_msg) 