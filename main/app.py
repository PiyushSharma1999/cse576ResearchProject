# app.py

import os
import time
from preprocess import Preprocess
from model import GraniteModel
import config
import csv
from io import StringIO
import torch
import pandas as pd
import requests

def generate_with_huggingface(model_name, prompt):
    API_TOKEN = "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"
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
    }
    
    payload = {
        "inputs": prompt,
        "options": {"use_cache": False},
    }
    
    # Make the API call
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()[0]['generated_text']  # Only return the generated text
    else:
        return f"Error: {response.status_code}, {response.text}"



def generate_prompts():
    llamaPrompt = '''create 30 prompts in either the general knowledge domain, programming, logical reasoning, or fictional story domain and ask a question about the prompt. Include 1-2 lines of irrelevant context, which should be relevant to the information in the prompt but have no impact on the solution to the question asked. The irrelevant context, when included, should be able to distract a small LLM trained on around 7 billion parameter. Create the prompt in the following CSV format, ensure no commas are in the response or add the correct punctuation for it to be acceptable in the CSV format. Respond with only the CSV prompt.
    Prompt without irrelevant context,Prompt with irrelevant context,Prompt with irrelevant context and asking for irrelevant context,Irrelevant context,Correct answer

    Some examples below
    "Four people (A=1min, B=2min, C=5min, D=10min) must cross a bridge at night. The bridge holds max 2 people. They have one flashlight. How do all cross in 17 minutes?","Four individuals (A=1min [vegan], B=2min [left-handed], C=5min [allergic to nuts], D=10min [former Olympian]) must cross a suspension bridge (built 1937, max load 300lbs) at 02:00hrs during a storm (winds 25mph). The bridge (inspected monthly per DOT regs) holds max 2 people. They share a Maglite XL50 (3xAAA batteries at 78% charge). How do all cross in 17 minutes while complying with OSHA safety standards?","Identify the irrelevant context in the prompt: ""Four individuals (A=1min [vegan], B=2min [left-handed], C=5min [allergic to nuts], D=10min [former Olympian]) must cross a suspension bridge (built 1937, max load 300lbs) at 02:00hrs during a storm (winds 25mph). The bridge (inspected monthly per DOT regs) holds max 2 people. They share a Maglite XL50 (3xAAA batteries at 78% charge). How do all cross in 17 minutes while complying with OSHA safety standards?""","vegan.left-handed.allergic to nuts.former Olympian. (built 1937, max load 300lbs) at 02:00hrs during a storm (winds 25mph).(inspected monthly per DOT regs). They share a Maglite XL50 (3xAAA batteries at 78% charge) while complying with OSHA safety standards",1. A+B cross (2min). 2. A returns (1min). 3. C+D cross (10min). 4. B returns (2min). 5. A+B cross (2min). Total: 17 minutes
    "In the story, Sarah discovers a hidden door in her attic that leads to a magical forest. What does Sarah find in the attic?","In the story, Sarah discovers a hidden door in her attic that leads to a magical forest. Sarah's grandmother had told her stories about enchanted realms when she was younger. The magical forest is known for its talking animals and ever-changing seasons. What does Sarah find in the attic?","Identify the irrelevant context in the prompt: ""In the story, Sarah discovers a hidden door in her attic that leads to a magical forest. Sarah's grandmother had told her stories about enchanted realms when she was younger. The magical forest is known for its talking animals and ever-changing seasons. What does Sarah find in the attic?""",Sarah's grandmother’s stories about enchanted realms.,Sarah finds a hidden door in the attic that leads to a magical forest.'''

    print("generating prompts")

    # will generate the input data
    response_csv = generate_with_huggingface(llamaPrompt)
    
    # Wrap it in StringIO so csv.DictReader can parse it like a file
    csvfile = StringIO(response_csv)

    print("Finished generating responses")
    return csvfile


def save_input_prompts_to_csv(csvfile):
    """Save the generated prompts to a CSV file."""
    fieldnames = [
        "Prompt without irrelevant context", 
        "Prompt with irrelevant context", 
        "Prompt with irrelevant context and asking for irrelevant context", 
        "Irrelevant context",
        "Correct answer"
    ]

    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    data_to_save = list(reader)

    # Define file paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "main/data")
    file_path = os.path.join(data_folder, "generated_input_file.csv")

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)

    # Save the input data to a CSV file
    file_exists = os.path.exists(file_path)
    with open(file_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_to_save)


def process_models(data):
    #Mistral AI and meta-llama need the Hugging face cl login to work
    models = [
        "Qwen/Qwen2.5-7B-Instruct", 
        "tiiuae/falcon-7b-instruct",
        "ibm-granite/granite-3.2-8b-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        #"HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
        "CohereForAI/aya-expanse-8b"
        #"google/gemma-7b-it",
        #"Qwen/Qwen2.5-7B-Instruct",      
        # "mistralai/Mistral-7B-Instruct-v0.3", # requires requested access
        # "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct", # done
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct", # done
        # "ibm-granite/granite-3.2-8b-instruct", # done
        # "tiiuae/falcon-7b-instruct", # done
        # "open-r1/OlympicCoder-7B",
        # "HuggingFaceH4/zephyr-7b-beta" # done
        # "deepseek-ai/deepseek-llm-7b-chat", # done
        # "Qwen/Qwen2-7B-Chat",
        # "mosaicml/mpt-7b-instruct" # needs its own model loader
    ]

    data_to_save = []
    for model_name in models:
        model = GraniteModel(model_name) # todo - rename
        print("loaded model: " + model_name)

        for index, row in data.iterrows():
            row_start_time = time.time()  # Start timer for this row

            print(f"\nProcessing Model: {model_name}, Prompt {index + 1}...")
            
            rel_prompt = row[config.rel_col_name]
            irr_prompt = row[config.irr_col_name]
            irr_ask_prompt = row[config.irr_ask_col_name]
            irr_context = row[config.irr_context_col]
            answer = row[config.answer_col_name]
                        
            # Generate model responses
            response_irr = model.generate_response(irr_prompt)
            response_rel = model.generate_response(rel_prompt)
            response_irr_context = model.generate_response(irr_ask_prompt)

            # Calculate and print time taken for this row
            row_end_time = time.time()
            row_time = row_end_time - row_start_time
            print(f"Time taken for Row {index + 1}: {row_time:.2f} seconds\n")

            # Save data
            data_to_save.append(
                {
                    "Model": model_name,
                    "Relevant Prompt": rel_prompt,
                    "Irrelevant Prompt": irr_prompt,
                    "Identify Irrelevant Context Prompt": irr_ask_prompt,
                    "Irrelevant Context": irr_context,
                    "Correct answer": answer,
                    "Relevant Response": response_rel,
                    "Irrelevant Response": response_irr,
                    "Identify Irrelevant Context Response": response_irr_context,
                }
            )

        #delete model and free up cache
        del model
        torch.cuda.empty_cache()

    return data_to_save


def evaluate_responses(data_to_save, ):
    data_passed = []
    data_not_passed = []
    data_identified_correctly = []
    data_identified_incorrectly = []

    # this will have two additional propertites, tricked and identified correctly
    data_combined = []

    for row in data_to_save:
        to_add = row.copy()

        # Retrieve the correct answer and irrelevant response from the row
        prompt = row["Relevant Prompt"]
        irr_prompt = row["Irrelevant Prompt"]
        irr_ask_prompt = row["Identify Irrelevant Context Prompt"]
        irr_context = row["Irrelevant Context"]
        correct_answer = row["Correct answer"]
        response_rel = row["Relevant Response"]
        response_irr = row["Irrelevant Response"]
        response_irr_context = row["Identify Irrelevant Context Response"]

        # check if llm got relevant response correct
        rel_comparison = compare_relevant_responses(prompt, correct_answer, response_rel)

        if  rel_comparison == "yes":
            # check if llm got irrelevant response correct
            irr_comparison = compare_irrelevant_responses(irr_prompt, correct_answer, response_irr, irr_context)
            # if it got tricked, consider it passed
            if irr_comparison == "yes":
                data_passed.append(row)
                to_add["Tricked"] = "yes"
            else:
                data_not_passed.append(row)
                to_add["Tricked"] = "no"

        #will add to the not passed data if not
        else:
            data_not_passed.append(row)
            to_add["Tricked"] = "no"

        
        # check if llm identified the irrelevant context correctly
        identified_comparison = compare_identified_responses(irr_ask_prompt, irr_context, response_irr_context)
        # if it identified correctly add to the identified data
        if identified_comparison == "yes":
            data_identified_correctly.append(row)
            to_add["Identified"] = "yes"
        else:
            data_identified_incorrectly.append(row)
            to_add["Identified"] = "no"

        # ad to grand total
        print("Irrelevant prompt: ", irr_prompt, "Identified: ", to_add["Identified"], "Tricked: ", to_add["Tricked"])
        data_combined.append(to_add)

    print("Finished evaluating responses", "", len(data_passed), "passed and", len(data_not_passed), "not passed", "", len(data_identified_correctly), "identified correctly", "", len(data_identified_incorrectly), "identified incorrectly")
    
    return data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined


def compare_relevant_responses(prompt, correct_answer, response):
    # Prompt to compare correct answer with the LLM's relevant response
    comparisonPrompt = (
        f"Is the following response correct in relation to the prompt: {prompt} and the correct answer to the prompt {correct_answer}. Respond with only YES or NO. "
        f"Response: {response}"
    )

    # Generate response then strip and lower()
    result = generate_with_huggingface(comparisonPrompt)
    result = result.strip().lower().rstrip('.')

    return result

    
def compare_irrelevant_responses(prompt, correct_answer, response, irr_context):
    # Prompt to compare correct answer with the LLM's relevant response
    comparisonPrompt = (
        f"For the prompt: {prompt}, this is the correct answer {correct_answer}. This is the irrelevant context that was included in the prompt: {irr_context}. If the following response answer correctly, respond only with NO. If the following response got the answer wrong, did it get it wrong because of the irrelevant context included? Respond with only YES or NO. "
        f"Response: {response}"
    )

    # Generate response then strip and lower()
    result = generate_with_huggingface(comparisonPrompt)
    result = result.strip().lower().rstrip('.')

    return result
    

def compare_identified_responses(irr_ask_prompt, irr_context, response_irr_context):
    # Prompt to compare correct answer with the LLM's relevant response
    comparisonPrompt = (
        f"For the prompt: {irr_ask_prompt}, this is the irrelevant context that was included in the prompt: {irr_context}. If the following response identifies the irrelevant context correctly, respond only with YES. If the following response does not identify the irrelevant context correctly, respond with NO. "
        f"Response: {response_irr_context}"
    )

    # Generate response then strip and lower()
    result = generate_with_huggingface(comparisonPrompt)
    result = result.strip().lower().rstrip('.')

    return result


def save_results(data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined):
    #find the ouput csv file route
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "main/data")
    file_path_passed = os.path.join(data_folder, "output_passed_file.csv")
    file_path_not_passed = os.path.join(data_folder, "output_not_passed_file.csv")
    file_path_identified_correctly = os.path.join(data_folder, "output_identified_correctly_file.csv")
    file_path_identified_incorrectly = os.path.join(data_folder, "output_identified_incorrectly_file.csv")

    combined_path = os.path.join(data_folder, "combined_output_file.csv")

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)

    file_exists = os.path.exists(file_path_passed)

    # Save the output that passed to a csv
    with open(file_path_passed, "a", newline='') as f:
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
                "Tricked",
                "Identified"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_passed)
    
    file_exists = os.path.exists(file_path_not_passed)

    # Save the output to a csv
    with open(file_path_not_passed, "a", newline='') as f:
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
                "Tricked",
                "Identified"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_not_passed)

    file_exists = os.path.exists(file_path_identified_correctly)
    # Save the output to a csv
    with open(file_path_identified_correctly, "a", newline='') as f:
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
                "Tricked",
                "Identified"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_identified_correctly)

    file_exists = os.path.exists(file_path_identified_incorrectly)
    # Save the output to a csv
    with open(file_path_identified_incorrectly, "a", newline='') as f:
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
                "Tricked",
                "Identified"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_identified_incorrectly)

    file_exists = os.path.exists(combined_path)
    # Save the output to a csv
    with open(combined_path, "a", newline='') as f:
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
                "Tricked",
                "Identified"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_combined)

    print("Finished saving results to CSV files")



def main():
    """ Main execution logic """
    total_start_time = time.time()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "main/data")
    input_file_path = os.path.join(data_folder, "input_file.csv")
    output_file_path = os.path.join(data_folder, "output_file.csv")


    if os.path.exists(input_file_path) and os.path.getsize(input_file_path) > 0:
        print(f"{input_file_path} found and is not empty, loading data...")

        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            print(f"{output_file_path} found and is not empty, loading data...")

            # output exists, just evaluate and save
            data_to_save = pd.read_csv(output_file_path).to_dict(orient="records")
            data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined = evaluate_responses(data_to_save)
            save_results(data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined)

            torch.cuda.empty_cache()

            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            print(f"Total Execution Time: {total_time:.2f} seconds")

            return
        else:
            preprocessor = Preprocess(file_name=input_file_path)
            data = preprocessor.get_columns()
    else:
        print(f"{input_file_path} not found or is empty, generating data...")
        
        # Generate input prompts 
        input_prompts_csv = generate_prompts()

        # Save the generated prompts to a CSV file
        save_input_prompts_to_csv(input_prompts_csv)

        # Load and display data
        preprocessor = Preprocess(file_name="generated_input_file.csv")
        preprocessor.display_data()
        data = preprocessor.get_columns()

    

    
    # Process models and generate responses
    data_to_save = process_models(data)


    # Evaluate responses
    data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined = evaluate_responses(data_to_save)
        

    # Save results to CSV files
    save_results(data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined)


    # Calculate and print total execution time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
