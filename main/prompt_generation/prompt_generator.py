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

    def generate_clean_prompt(self, domain=None):
        """Generate just the clean prompt without any additional text"""
        if domain is None:
            domain = random.choice(self.domains)
            
        prompt = f"""
        Generate a challenging question about {domain} that requires specific knowledge to answer.
        The question should be answerable in 1-2 sentences but potentially confusing without domain expertise.
        Format: What/Why/How question without context clues in the question itself.
        
        IMPORTANT: Provide ONLY the question text with no additional commentary or explanations.
        """
        response = self.reasoner.generate(prompt)
        # Clean up the response - remove quotes, extra spaces, explanations
        clean_response = response.strip().strip('"').strip()
        
        # If there are multiple lines, just take the first line with a question mark
        if "\n" in clean_response:
            lines = clean_response.split("\n")
            for line in lines:
                if "?" in line:
                    clean_response = line.strip()
                    break
        
        return clean_response, domain

    def generate_irrelevant_context(self, domain):
        """Generate irrelevant context about the domain without knowing the specific prompt"""
        prompt = f"""
        Generate 6-7 factual sentences about {domain} that:
        1. Are related to the general domain
        2. Contain no false information
        3. Use technical terms where appropriate
        4. Form a cohesive paragraph
        
        IMPORTANT: Provide ONLY the context paragraph with no additional commentary, explanations or formatting.
        """
        response = self.reasoner.generate(prompt)
        # Clean up the response - remove quotes, extra spaces, explanations
        clean_response = response.strip().strip('"').strip()
        
        # If there are headings or other formatting markers, try to remove them
        lines = clean_response.split("\n")
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like headings or instructions
            if not line.strip() or line.strip().lower().startswith(("here", "context:", "paragraph:")):
                continue
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)

    def generate_expected_answer(self, clean_prompt):
        """Generate expected answer for the clean prompt"""
        prompt = f"""
        Provide a concise answer to this question: {clean_prompt}
        
        IMPORTANT: Provide ONLY the answer with no additional commentary or explanations.
        """
        response = self.reasoner.generate(prompt)
        return response.strip().strip('"').strip()

    def generate_dataset(self, num_samples=10):
        dataset = []
        for _ in range(num_samples):
            try:
                # First generate clean prompt
                clean_prompt, domain = self.generate_clean_prompt()
                
                # Then generate irrelevant context based only on domain
                irrelevant_context = self.generate_irrelevant_context(domain)
                
                # Get expected answer
                expected_answer = self.generate_expected_answer(clean_prompt)
                
                # Create combined prompt (context + clean prompt)
                combined_prompt = f"{irrelevant_context}\n\n{clean_prompt}"

                irr_context_identification_prompt = f"{combined_prompt} \n\n What part of the context is irrelevant or unnecessary to answer the above reasoning question?"
                
                dataset.append({
                    "domain": domain,
                    "clean_prompt": clean_prompt,
                    "irrelevant_context": irrelevant_context,
                    "combined_prompt": combined_prompt,
                    "expected_answer": expected_answer,
                    "irrelevant_context_identification_prompt": irr_context_identification_prompt
                })
                print(f"Generated sample {len(dataset)}/{num_samples}")
            except Exception as e:
                print(f"Error generating sample: {str(e)}")
                
        return dataset
