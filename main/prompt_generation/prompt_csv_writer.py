import csv

class PromptCSVWriter:
    @staticmethod
    def write_to_csv(data, fieldnames, filename="prompts.csv"):
        """Write dataset to CSV"""
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in data:
                writer.writerow(item)
        
        print(f"Dataset written to {filename}")