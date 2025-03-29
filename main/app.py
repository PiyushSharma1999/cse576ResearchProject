# app.py

import os
import time
from preprocess import Preprocess
from model import GraniteModel
import config
import csv

def main():
    """ Main execution logic """
    total_start_time = time.time()

    # Load and display data
    preprocessor = Preprocess()
    preprocessor.display_data()
    data = preprocessor.get_columns()

    #Mistral AI and meta-llama need the Hugging face cl login to work
    models = [
        # "mistralai/Mistral-7B-Instruct-v0.3",
        "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "ibm-granite/granite-3.2-8b-instruct",
        "tiiuae/falcon-7b-instruct",
    ]

    data_to_save = []
    for model_name in models:

        model = GraniteModel(model_name) # todo - rename
        print("loaded model: " + model_name)
        for index, row in data.iterrows():
            row_start_time = time.time()  # Start timer for this row
            
            rel_prompt = row[config.rel_col_name]
            irr_prompt = row[config.irr_col_name]
            irr_ask_prompt = row[config.irr_ask_col_name]
            irr_context = row[config.irr_context_col]
            answer = row[config.answer_col_name]
            
            print(f"\nProcessing Row {index + 1}...\n")
            
            # Generate model responses
            response_irr = model.generate_response(irr_prompt)
            response_rel = model.generate_response(rel_prompt)
            response_irr_context = model.generate_response(irr_ask_prompt)
            
            print(f"Irrelevant Context Response:\n{response_irr}\n")
            print(f"Relevant Context Response:\n{response_rel}\n")
            
            # Compute similarity
            similarity_score_rel = model.compute_similarity(response_rel, answer)
            print(f"Cosine Similarity for Relevant Response: {similarity_score_rel}\n")
            similarity_score_irr = model.compute_similarity(response_irr, answer)
            print(f"Cosine Similarity for Irrelevant Response: {similarity_score_irr}\n")
            similarity_score_irr_context = model.compute_similarity(response_irr_context, irr_context)
            print(f"Cosine Similarity for Identify Irrelevant Context Response: {similarity_score_irr_context}\n")

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
                    "Identify Irrelevant Context Prompt": irr_ask_prompt,
                    "Irrelevant Context": irr_context,
                    "Correct answer": answer,
                    "Relevant Response": response_rel,
                    "Irrelevant Response": response_irr,
                    "Identify Irrelevant Context Response": response_irr_context,
                    "Cosine Similarity (Relevant)": similarity_score_rel,
                    "Cosine Similarity (Irrelevant)": similarity_score_irr,
                    "Cosine Similarity (Identify Irrelevant Context)": similarity_score_irr_context,
                    "Execution Time": row_time,
                }
            )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "main/data")
    file_path = os.path.join(data_folder, "output_file.csv")

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)

    file_exists = os.path.exists(file_path)

    # Save to csv
    with open(file_path, "a") as f:
        # if header does not exit, add one
        writer = csv.DictWriter(f, fieldnames=[
                "Model", 
                "Relevant Prompt", 
                "Irrelevant Prompt", 
                "Identify Irrelevant Context Prompt",
                "Irrelevant Context",
                "Correct answer",
                "Relevant Response", 
                "Irrelevant Response", 
                "Identify Irrelevant Context Response",
                "Cosine Similarity (Relevant)", 
                "Cosine Similarity (Irrelevant)", 
                "Cosine Similarity (Identify Irrelevant Context)",
                "Execution Time"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_to_save)
    # Calculate and print total execution time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
