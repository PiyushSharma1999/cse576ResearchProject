import os
import json
import config
from prompt_generator import AutomatedPromptGenerator
from prompt_validator import PromptValidator
from prompt_csv_writer import PromptCSVWriter

def main():
    # Ensure API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Please set the DEEPSEEK_API_KEY environment variable")
        return
    
    # Parameters
    num_samples = 5  # Start with a small number for testing
    output_csv = "prompts.csv"
    validation_report = "validation_report.json"
    
    print(f"Starting process to generate {num_samples} prompt samples...")
    
    # Generate dataset
    generator = AutomatedPromptGenerator()
    dataset = generator.generate_dataset(num_samples)
    
    print(f"Generated {len(dataset)} samples. Validating...")
    
    # Validate all entries
    validator = PromptValidator()
    validation_results = []
    valid_dataset = []
    
    for entry in dataset:
        print(entry)
        result = validator.validate_entry(entry)
        validation_results.append(result)
        
        if result.get("is_valid", False):
            valid_dataset.append(entry)
    
    # Write validation report
    with open(validation_report, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"Validation complete. Found {len(valid_dataset)}/{len(dataset)} valid entries.")
    print(f"Validation report written to {validation_report}")
    
    # Write valid entries to CSV
    if valid_dataset:
        PromptCSVWriter.write_to_csv(valid_dataset, config.auto_gen_fieldnames, output_csv)
    else:
        print("No valid entries found. CSV not created.")

if __name__ == "__main__":
    main()