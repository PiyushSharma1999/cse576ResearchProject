import csv

class PromptCSVWriter:
    @staticmethod
    def write_to_csv(data, filename="prompts.csv"):
        """Write dataset to CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "domain",
                "clean_prompt",
                "irrelevant_context",
                "combined_prompt",
                "expected_answer"
            ])
            
            writer.writeheader()
            for item in data:
                writer.writerow(item)
        
        print(f"Dataset written to {filename}")