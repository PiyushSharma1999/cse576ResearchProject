import os
import requests
from typing import Optional, Dict

class DeepseekReasoner:
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Model parameters
        self.max_tokens = 512
        self.temperature = 0.5
        self.total_tokens = 0

    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Generate response to a prompt with optional context
        Args:
            prompt: Input question/request
            context: Optional context dictionary
        """
        try:
            # Build message list
            messages = []
            if context:
                messages.append(context)
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            self.total_tokens += data.get('usage', {}).get('total_tokens', 0)
            
            return data['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}") from e

    def get_usage(self) -> Dict:
        """Get current token usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_tokens * 0.000002  # Update with actual pricing
        }