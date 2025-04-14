import random
from deep_reasoner import DeepseekReasoner

class AutomatedPromptGenerator:
    def __init__(self):
        self.reasoner = DeepseekReasoner()
        self.domains = [
            "fictional stories", 
            "general knowledge", 
            "programming"
        ]
        
    def _generate_clean_prompt(self, domain):
        prompt = f"""
        Generate a challenging question about {domain} that requires specific knowledge to answer.
        The question should be answerable in 1-2 sentences but potentially confusing without domain expertise.
        Format: What/Why/How question without context clues in the question itself.
        """
        return self.reasoner.generate(prompt).strip('"').strip()

    def _generate_irrelevant_context(self, clean_prompt, domain):
        prompt = f"""
        For this {domain} question: "{clean_prompt}"
        Generate 6-7 factual sentences that are:
        1. Related to the question's general domain
        2. Do NOT answer the question
        3. Contain no false information
        4. Appear relevant but lack critical information
        Use technical terms where appropriate.
        """
        return self.reasoner.generate(prompt)

    def generate_dataset(self, num_samples = 10):
        dataset = []
        for _ in range(num_samples):
            domain = random.choice(self.domains)
            clean_prompt = self._generate_clean_prompt(domain)
            expected_answer = self.reasoner.generate(
                f"Concise answer to: {clean_prompt}"
            )
            irrelevant_context = self._generate_irrelevant_context(clean_prompt, domain)
            
            dataset.append({
                "domain": domain,
                "clean_prompt": clean_prompt,
                "irrelevant_context": irrelevant_context,
                "expected_answer": expected_answer
            })
        return dataset