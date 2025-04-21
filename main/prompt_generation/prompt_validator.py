import re
from deep_reasoner import DeepseekReasoner

class PromptValidator:
    def __init__(self):
        self.reasoner = DeepseekReasoner()
        self.rules = {
            "context_length": self._check_context_length,
            "domain_alignment": self._check_domain_alignment,
            "answer_exclusion": self._check_answer_exclusion,
            "factual_consistency": self._check_factual_consistency
        }

    def validate_entry(self, entry):
        """Validate a single CSV entry against all rules"""
        results = {}
        try:
            for rule_name, rule_func in self.rules.items():
                results[rule_name] = rule_func(entry)
                
            return {
                "is_valid": all(results.values()),
                "details": results,
                "entry": entry
            }
        except KeyError as e:
            return {"error": f"Missing field: {str(e)}"}

    def _check_context_length(self, entry):
        """Verify 6-7 sentences in irrelevant context"""
        sentences = re.split(r'(?<=[.!?]) +', entry["irrelevant_context"])
        return 6 <= len(sentences) <= 7

    def _check_domain_alignment(self, entry):
        """Check context matches domain keywords"""
        domain_keywords = {
            "fictional stories": ["character", "plot", "chapter", "setting", "story", "novel"],
            "general knowledge": ["science", "history", "fact", "data", "world", "discovery"],
            "programming": ["code", "algorithm", "language", "system", "software", "function"]
        }
        
        keywords = domain_keywords.get(entry["domain"], [])
        context = entry["irrelevant_context"].lower()
        return any(kw in context for kw in keywords)

    def _check_answer_exclusion(self, entry):
        """Ensure context doesn't contain answer keywords"""
        answer_keywords = set(entry["expected_answer"].lower().split()[:5])
        context_words = set(entry["irrelevant_context"].lower().split())
        return len(answer_keywords & context_words) < 2

    def _check_factual_consistency(self, entry):
        """Basic check for obvious contradictions"""
        verification_prompt = f"""
        Verify this statement contains no factual errors:
        Context: {entry["irrelevant_context"]}
        Question: {entry["clean_prompt"]}
        Answer: {entry["expected_answer"]}
        Respond ONLY with 'True' or 'False'
        """
        
        response = self.reasoner.generate(verification_prompt, use_history=False)
        return response.lower().strip().startswith("true") or response.lower().strip() == "true"