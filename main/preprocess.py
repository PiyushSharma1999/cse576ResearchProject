# preprocess.py

import pandas as pd
import os
import config

class Preprocess:
    def __init__(self, file_name=config.file_path, 
                rel_col_name=config.rel_col_name,
                irr_col_name=config.irr_col_name, 
                irr_ask_col_name=config.irr_ask_col_name,
                irr_context_col=config.irr_context_col,
                answer_col_name=config.answer_col_name):
        
        self.file_name = file_name
        self.rel_col_name = rel_col_name
        self.irr_col_name = irr_col_name
        self.irr_ask_col_name = irr_ask_col_name
        self.irr_context_col = irr_context_col
        self.answer_col_name = answer_col_name
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, self.file_name)
        
        self.data = pd.read_csv(self.file_path)

    def display_data(self):
        # display first row
        print("First row of the data:")
        print(self.data.iloc[0])
    
    def get_data(self):
        return self.data
    
    def get_columns(self):
        return self.data[[self.rel_col_name, self.irr_col_name, self.irr_ask_col_name, self.irr_context_col, self.answer_col_name]]
