# app.py

import os
import time
from deepseek_preprocessor import DeepSeekPreprocess
from preprocess import Preprocess
from model import GraniteModel
import config
import csv
from io import StringIO
import torch
import pandas as pd
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

def generate_prompts(model_name, llamaModel):
    num_prompts = 30

    llamaPrompt = f'''Create {num_prompts} reasoning problems from domains such as a leetcode problem or fictional storytelling. For each, include:

        1. A version of the prompt with 3 ore more sentences of context to help answer the question.
        2. A version of the same prompt *with* 3 or more lines of irrelevant context added. This context should be related to the topic but not helpful in solving the question. Place this irrelevant context **before** the question.
        3. A version where you ask the model to **identify the irrelevant context**. This version should begin with the full context (irrelevant + relevant), followed by a clear reasoning question, then end with: *"What part of the context is irrelevant or unnecessary to answer the above reasoning question?"*
        4. The actual irrelevant context itself, copied exactly.
        5. The correct answer to the reasoning question.

        **Format the response as a single CSV table** using the following header row:

        Prompt without irrelevant context,Prompt with irrelevant context,Prompt with irrelevant context and asking for irrelevant context,Irrelevant Context,Correct answer

        Use quotation marks to wrap each full field. Do not include any commas unless they are properly escaped or replaced with punctuation such as semicolons. Respond ONLY with the CSV. Do not add any explanations or extra content.

        Here is an example of the format:

        "A train travels from Station A to Station B, a distance of 200 miles. The train travels at a constant speed of 50 miles per hour. It does not stop at any other stations during the journey. If the train leaves Station A at 9:00 AM, what time will it arrive at Station B?","A train travels from Station A to Station B, a distance of 200 miles. The train travels at a constant speed of 50 miles per hour. There is a 30-minute gas stop at Station C, which is located 100 miles between Station A and Station B. The train does not need to stop at Station C for gas on this particular trip, but it would normally refuel there on long journeys. The train is also equipped with a high-speed Wi-Fi network for passengers to use during the trip. The train’s air conditioning system is set to a comfortable 72°F for optimal passenger comfort. If the train leaves Station A at 9:00 AM, what time will it arrive at Station B?","Context: The train does not need to stop at Station C for gas on this particular trip, but it would normally refuel there on long journeys. The train is also equipped with a high-speed Wi-Fi network for passengers to use during the trip. The train’s air conditioning system is set to a comfortable 72°F for optimal passenger comfort. A train travels from Station A to Station B, a distance of 200 miles. The train travels at a constant speed of 50 miles per hour. Reasoning Question: If the train leaves Station A at 9:00 AM, what time will it arrive at Station B? What part of the context is irrelevant or unnecessary to answer the above reasoning question?","There is a 30-minute gas stop at Station C, which is located 100 miles between Station A and Station B. The train does not need to stop at Station C for gas on this particular trip, but it would normally refuel there on long journeys. The train is also equipped with a high-speed Wi-Fi network for passengers to use during the trip. The train’s air conditioning system is set to a comfortable 72°F for optimal passenger comfort.","The train travels a total of 200 miles at a speed of 50 miles per hour. The total travel time is 200 ÷ 50 = 4 hours. The train leaves at 9:00 AM, so it will arrive at 1:00 PM."

        Now generate 30 such rows in this format.
    '''

    print(f"Generating {num_prompts} prompts...")

    # will generate the input data
    response_csv = llamaModel.generate_response(llamaPrompt)
    
    # Wrap it in StringIO so csv.DictReader can parse it like a file
    csvfile = StringIO(response_csv)

    print("Finished generating {num_prompts} prompts...")
    return csvfile


def save_input_prompts_to_csv(csvfile):
    print("Saving generated prompts to CSV file...")

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
    input_data_folder = os.path.join(project_root, "main/data/input")
    file_path = os.path.join(input_data_folder, "generated_input_file.csv")

    # Ensure the data folder exists
    os.makedirs(input_data_folder, exist_ok=True)

    # Save the input data to a CSV file
    file_exists = os.path.exists(file_path)
    with open(file_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_to_save)

    print(f"Generated prompts saved to {file_path}.")


def process_models(data):
    print("Processing models...")

    if isinstance(data, list):
        data = pd.DataFrame(data)

    #Mistral AI and meta-llama need the Hugging face cl login to work
    models = [
        "Zyphra/Zamba2-7B-Instruct", # DONE 
        "Qwen/Qwen2.5-7B-Instruct",  # DONE
        "tiiuae/falcon-7b-instruct", # DONE
        # "ibm-granite/granite-3.2-8b-instruct",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "meta-llama/Llama-3.1-8B-Instruct",
        #"HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
        # "CohereForAI/aya-expanse-8b"
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
        print("Loaded model: " + model_name)

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

            obj = {
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

            # Save data
            data_to_save.append(obj)

        #delete model and free up cache
        del model
        torch.cuda.empty_cache()

    
    print(f"Finished processing {len(model)} models")

    return data_to_save


def evaluate_responses(data_to_save, llamaModelName):
    print("Evaluating responses...")

    llama70b = GraniteModel(llamaModelName)

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
        rel_comparison = compare_relevant_responses(prompt, correct_answer, response_rel, llama70b)

        if  rel_comparison == "yes":
            # check if llm got irrelevant response correct
            irr_comparison = compare_irrelevant_responses(irr_prompt, correct_answer, response_irr, irr_context, llama70b)
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
        identified_comparison = compare_identified_responses(irr_ask_prompt, irr_context, response_irr_context, llama70b)
        # if it identified correctly add to the identified data
        if identified_comparison == "yes":
            data_identified_correctly.append(row)
            to_add["Identified"] = "yes"
        else:
            data_identified_incorrectly.append(row)
            to_add["Identified"] = "no"

        # ad to grand total
        print("Model: ", row["Model"], " |  Identified: ", to_add["Identified"], " |  Tricked: ", to_add["Tricked"])
        data_combined.append(to_add)

    print("Finished evaluating responses", "", len(data_passed), "passed and", len(data_not_passed), "not passed", "", len(data_identified_correctly), "identified correctly", "", len(data_identified_incorrectly), "identified incorrectly")
    
    return data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined


def compare_relevant_responses(prompt, correct_answer, response, llama70b):
    # Prompt to compare correct answer with the LLM's relevant response
    comparisonPrompt = (
        f"Is the following response correct in relation to the prompt: {prompt} and the correct answer to the prompt {correct_answer}. Respond with only YES or NO. "
        f"Response: {response}"
    )

    # Generate response then strip and lower()
    result = llama70b.generate_response(comparisonPrompt)
    result = result.strip().lower().rstrip('.')

    return result

    
def compare_irrelevant_responses(prompt, correct_answer, response, irr_context, llama70b):
    # Prompt to compare correct answer with the LLM's relevant response
    comparisonPrompt = (
        f"For the prompt: {prompt}, this is the correct answer {correct_answer}. This is the irrelevant context that was included in the prompt: {irr_context}. If the following response answer correctly, respond only with NO. If the following response got the answer wrong, did it get it wrong because of the irrelevant context included? Respond with only YES or NO. "
        f"Response: {response}"
    )

    # Generate response then strip and lower()
    result = llama70b.generate_response(comparisonPrompt)
    result = result.strip().lower().rstrip('.')

    return result
    

def compare_identified_responses(irr_ask_prompt, irr_context, response_irr_context, llama70b):
    # Prompt to compare correct answer with the LLM's relevant response
    comparisonPrompt = (
        f"For the prompt: {irr_ask_prompt}, this is the irrelevant context that was included in the prompt: {irr_context}. If the following response identifies the irrelevant context correctly, respond only with YES. If the following response does not identify the irrelevant context correctly, respond with NO. "
        f"Response: {response_irr_context}"
    )

    # Generate response then strip and lower()
    result = llama70b.generate_response(comparisonPrompt)
    result = result.strip().lower().rstrip('.')

    return result


def save_responses_only_to_csv(output_path, filename, data):
    processed_outputs = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)
    file_exists = os.path.exists(processed_outputs)

    # Save the output that passed to a CSV
    with open(processed_outputs, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Model", 
            "Relevant Prompt", 
            "Irrelevant Prompt", 
            "Identify Irrelevant Context Prompt",
            "Irrelevant Context",
            "Correct answer",
            "Relevant Response", 
            "Irrelevant Response", 
            "Identify Irrelevant Context Response"
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(data)


def save_responses_and_evaluation_to_csv(output_path, filename, data):
    processed_outputs = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)
    file_exists = os.path.exists(processed_outputs)

    with open(processed_outputs, "a", newline='') as f:
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
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(data)


def main():
    """ Main execution logic """
    total_start_time = time.time()
    llamaModelName = "meta-llama/Llama-3.3-70B-Instruct" 

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manual_data_folder = os.path.join(project_root, "main/data/manual")
    output_data_folder = os.path.join(project_root, "main/data/output")
    prompt_generation_folder = os.path.join(project_root, "main/prompt_generation")

    deepSeek_prompts_file_path = os.path.join(prompt_generation_folder, "prompts.csv")
    manual_input_file_path = os.path.join(manual_data_folder, "input_file.csv")
    manual_output_file_path = os.path.join(manual_data_folder, "output_file.csv")
    preran_output_file_path = os.path.join(output_data_folder, "processed_output.csv")

    need_to_process = False

    # If Deep Seek generated prompts file exists and is not empty, load it
    if os.path.exists(deepSeek_prompts_file_path) and os.path.getsize(deepSeek_prompts_file_path) > 0:
        print(f"Deep seek generated prompts csv found and is not empty, loading data...")

        # Load and display data
        preprocessor = DeepSeekPreprocess(file_name=deepSeek_prompts_file_path)
        preprocessor.display_data()
        data = preprocessor.get_columns()

        need_to_process = True

    # If manual input file exists and is not empty, load it
    elif os.path.exists(manual_input_file_path) and os.path.getsize(manual_input_file_path) > 0:
        print(f"Manual prompts csv found and is not empty, loading data...")

        # If manual output responses file exists and is not empty, load it and evaluate it
        if os.path.exists(manual_output_file_path) and os.path.getsize(manual_output_file_path) > 0:
            print(f"{manual_output_file_path} found and is not empty, loading data...")
            data_to_save = pd.read_csv(manual_output_file_path).to_dict(orient="records")

        # If the output file is not found, check if the preran_output_file_path exists <- this csv holds responses that were generated but not evaluated yet
        elif os.path.exists(preran_output_file_path) and os.path.getsize(preran_output_file_path) > 0:
            print(f"{preran_output_file_path} found and is not empty, loading data...")
            data_to_save = pd.read_csv(preran_output_file_path).to_dict(orient="records")

        else:
            # Load and display data
            preprocessor = Preprocess(file_name=manual_input_file_path)
            preprocessor.display_data()
            data = preprocessor.get_columns()

            need_to_process = True

    # If neither file exists or is empty, generate new data
    else:
        print(f"Manual input and Deep Seek prompts csv not found or is empty, generating data...")

        llamaModel = GraniteModel(llamaModelName)
        torch.cuda.empty_cache()
        
        # Generate input prompts 
        input_prompts_csv = generate_prompts(llamaModelName, llamaModel)
        save_input_prompts_to_csv(input_prompts_csv)

        #will delete the model and free up the cache
        del llamaModel
        torch.cuda.empty_cache()

        # Load and display data
        preprocessor = Preprocess(file_name="generated_input_file.csv")
        preprocessor.display_data()
        data = preprocessor.get_columns()

        need_to_process = True


    if need_to_process:
        # Load the data from the CSV file
        data_to_save = process_models(data)
        save_responses_only_to_csv(output_data_folder, "processed_output.csv", data_to_save)

    # Evaluate responses
    data_passed, data_not_passed, data_identified_correctly, data_identified_incorrectly, data_combined = evaluate_responses(data_to_save, llamaModelName)    

    # Save results to CSV files
    print("Saving evaluation results to CSV files...")
    save_responses_and_evaluation_to_csv(output_data_folder, "output_passed_file.csv", data_passed)
    save_responses_and_evaluation_to_csv(output_data_folder, "output_not_passed_file.csv", data_not_passed)
    save_responses_and_evaluation_to_csv(output_data_folder, "combined_output_file.csv", data_combined)    
    
    # Clear the model to free memory
    torch.cuda.empty_cache()

    # Calculate and print total execution time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Finished! Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
