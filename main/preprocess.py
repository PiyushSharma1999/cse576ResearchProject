# preprocess.py

import pandas as pd
import os
import config

class Preprocess:
    def __init__(self, file_name=config.file_path, irr_col_name=config.irr_col_name, rel_col_name=config.rel_col_name):
        self.file_name = file_name
        self.irr_col_name = irr_col_name
        self.rel_col_name = rel_col_name
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, self.file_name)
        
        self.data = pd.read_csv(self.file_path)

    def display_data(self):
        print(self.data.head())
    
    def get_data(self):
        return self.data
    
    def get_columns(self):
        return self.data[[self.irr_col_name, self.rel_col_name]]
