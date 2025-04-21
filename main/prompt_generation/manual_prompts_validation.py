import csv
import os
import json
import pandas as pd
from prompt_validator import PromptValidator

def main():
    # Check for environment variable
    if not os.getenv("MANUAL_PROMPTS_FILE_PATH"):
        print("Error: MANUAL_PROMPTS_FILE_PATH environment variable not found.")
        return

    # Get file path from environment variable
    manual_prompts_file = os.getenv("MANUAL_PROMPTS_FILE_PATH")
    
    # Verify file exists
    if not os.path.exists(manual_prompts_file):
        print(f"Error: File not found at {manual_prompts_file}")
        return
    
    try:
        # Load the dataset from CSV
        dataset = pd.read_csv(manual_prompts_file)
        print(f"Loaded {len(dataset)} rows from {manual_prompts_file}")
        
        # Initialize validator
        validator = PromptValidator()
        
        # Create lists to store validation results and valid entries
        validation_results = []
        valid_dataset = []
        
        # Process each row in the dataset
        for index, row in dataset.iterrows():
            # Convert row to dictionary
            entry = row.to_dict()
            
            print(f"Validating entry {index+1}/{len(dataset)}")
            
            # Validate the entry
            result = validator.validate_entry(entry)
            validation_results.append(result)
            
            # Add to valid dataset if valid
            if result.get("is_valid", False):
                valid_dataset.append(entry)
        
        # Write validation report
        validation_report = "manual_validation_report.json"
        with open(validation_report, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"Validation complete. Found {len(valid_dataset)}/{len(dataset)} valid entries.")
        print(f"Validation report written to {validation_report}")
        
        # Write valid entries to new CSV
        if valid_dataset:
            output_csv = "valid_manual_prompts.csv"
            df_valid = pd.DataFrame(valid_dataset)
            df_valid.to_csv(output_csv, index=False)
            print(f"Valid entries written to {output_csv}")
        else:
            print("No valid entries found. CSV not created.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()