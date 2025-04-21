# preprocess.py

import pandas as pd
import os
import config

class DeepSeekPreprocess:
    def __init__(self, file_name=config.file_path):
        
        self.file_name = file_name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, self.file_name)
        
        # Read the CSV data
        self.data = pd.read_csv(self.file_path)

        # Column name mapping
        self.column_mapping = {
            "domain": "Prompt without irrelevant context",
            "clean_prompt": "Prompt with irrelevant context",
            "irrelevant_context": "Irrelevant context",
            "combined_prompt": "Prompt with irrelevant context and asking for irrelevant context",
            "expected_answer": "Correct answer",
            "irrelevant_context_identification_prompt": "Irrelevant context identification prompt"
        }

    def display_data(self):
        # Display first row of the data
        print("First row of the data:")
        print(self.data.iloc[0])
    
    def get_data(self):
        return self.data
    
    def get_columns(self):
        # Return the specific columns
        renamed_data = self.data.rename(columns=self.column_mapping)
        
        # Return the renamed columns
        return renamed_data[[
            "Prompt without irrelevant context", 
            "Prompt with irrelevant context", 
            "Prompt with irrelevant context and asking for irrelevant context", 
            "Irrelevant context", 
            "Correct answer"
        ]]