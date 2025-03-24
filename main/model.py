# model.py

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import config
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GraniteModel:
    def __init__(self):
        """ Load the Granite model and tokenizer """
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map=config.device,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_response(self, prompt):
        """ Generate response from the model """
        conv = [{"role": "user", "content": prompt}]
        
        input_ids = self.tokenizer.apply_chat_template(
            conv, return_tensors="pt", thinking=False, return_dict=True, add_generation_prompt=True
        ).to(config.device)

        set_seed(42)
        output = self.model.generate(**input_ids, max_new_tokens=8192)
        
        response = self.tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def compute_similarity(self, response1, response2):
        """ Compute cosine similarity between two responses """
        embedding1 = self.sbert_model.encode(response1)
        embedding2 = self.sbert_model.encode(response2)
        
        return cosine_similarity([embedding1], [embedding2])[0][0]
