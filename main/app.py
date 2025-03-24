# app.py

import os
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

    data_to_save = []
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
        similarity_score_rel = model.compute_similarity(response_rel, response_rel)
        print(f"Cosine Similarity for Relevant Response: {similarity_score_rel}\n")
        similarity_score_irr = model.compute_similarity(response_irr, response_rel)
        print(f"Cosine Similarity for Relevant Response: {similarity_score_irr}\n")


        # Calculate and print time taken for this row
        row_end_time = time.time()
        row_time = row_end_time - row_start_time
        print(f"Time taken for Row {index + 1}: {row_time:.2f} seconds\n")

        # Save data
        data_to_save.append(
            {
                "Model": config.model_path,
                "Relevant Prompt": rel_prompt,
                "Irrelevant Prompt": irr_prompt,
                "Relevant Response": response_rel,
                "Irrelevant Response": response_irr,
                "Cosine Similarity (Relevant)": similarity_score_rel,
                "Cosine Similarity (Irrelevant)": similarity_score_irr,
                "Execution Time": row_time,
            }
        )

    file_path = "output.csv"
    file_exists = os.path.exists(file_path)

    # Save to csv
    with open("output.csv", "w") as f:
        # if header does not exit, add one
        if not file_exists or os.stat(file_path).st_size == 0:
            f.write("Model, Relevant Prompt, Irrelevant Prompt, Relevant Response, Irrelevant Response, Cosine Similarity (Relevant), Cosine Similarity (Irrelevant), Execution Time\n")
        for item in data_to_save:
            f.write(f'{item["Model"]}, {item["Relevant Prompt"]}, {item["Irrelevant Prompt"]}, {item["Relevant Response"]}, {item["Irrelevant Response"]}, {item["Cosine Similarity (Relevant)"]}, {item["Cosine Similarity (Irrelevant)"]}, {item["Execution Time"]}\n')

    # Calculate and print total execution time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
