import csv

class PromptCSVWriter:
    @staticmethod
    def write_to_csv(data, filename = "prompts.csv"):
        """Write dataset to CSV with combined prompts column"""
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
                combined = f"{item['irrelevant_context']}\n\n{item['clean_prompt']}"
                writer.writerow({**item, "combined_prompt": combined})