# app.py

import time
from preprocess import Preprocess
from model import GraniteModel
import config

def main():
    """ Main execution logic """
    total_start_time = time.time()

    # Load and display data
    preprocessor = Preprocess()
    preprocessor.display_data()
    data = preprocessor.get_columns()

    model = GraniteModel()

    for index, row in data.iterrows():
        row_start_time = time.time()  # Start timer for this row
        
        irr_prompt = row[config.irr_col_name]
        rel_prompt = row[config.rel_col_name]
        
        print(f"\nProcessing Row {index + 1}...\n")
        
        # Generate model responses
        response_irr = model.generate_response(irr_prompt)
        response_rel = model.generate_response(rel_prompt)
        
        print(f"Irrelevant Context Response:\n{response_irr}\n")
        print(f"Relevant Context Response:\n{response_rel}\n")
        
        # Compute similarity
        similarity_score = model.compute_similarity(response_irr, response_rel)
        print(f"Cosine Similarity: {similarity_score}\n")

        # Calculate and print time taken for this row
        row_end_time = time.time()
        row_time = row_end_time - row_start_time
        print(f"Time taken for Row {index + 1}: {row_time:.2f} seconds\n")

    # Calculate and print total execution time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
