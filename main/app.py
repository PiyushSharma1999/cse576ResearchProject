# app.py

import os
import time
from preprocess import Preprocess
from model import GraniteModel
import config
import csv
from io import StringIO
import torch

def main():
    """ Main execution logic """
    total_start_time = time.time()
    llamaModelName = "meta-llama/Llama-3.3-70B-Instruct" 

    #will use the deepseep v2 model to create an initial dataset to test
    torch.cuda.empty_cache()
    #llamaModelName = "meta-llama/Llama-3.3-70B-Instruct" 
    llamaPrompt = '''create 30 prompts in either the general knowledge domain or fictional story domain and ask a question about the prompt. Include 1-2 lines of irrelevant context, which should be relevant to the information in the prompt but have no impact on the solution to the question asked. The irrelevant context, when included, should be able to distract a small LLM trained on around 7 billion parameter. Create the prompt in the following CSV format, ensure no commas are in the response or add the correct punctuation for it to be acceptable in the CSV format. Respond with only the CSV prompt.
Prompt without irrelevant context,Prompt with irrelevant context,Prompt with irrelevant context and asking for irrelevant context,Irrelevant context,Correct answer

Some examples below
"Four people (A=1min, B=2min, C=5min, D=10min) must cross a bridge at night. The bridge holds max 2 people. They have one flashlight. How do all cross in 17 minutes?","Four individuals (A=1min [vegan], B=2min [left-handed], C=5min [allergic to nuts], D=10min [former Olympian]) must cross a suspension bridge (built 1937, max load 300lbs) at 02:00hrs during a storm (winds 25mph). The bridge (inspected monthly per DOT regs) holds max 2 people. They share a Maglite XL50 (3xAAA batteries at 78% charge). How do all cross in 17 minutes while complying with OSHA safety standards?","Identify the irrelevant context in the prompt: ""Four individuals (A=1min [vegan], B=2min [left-handed], C=5min [allergic to nuts], D=10min [former Olympian]) must cross a suspension bridge (built 1937, max load 300lbs) at 02:00hrs during a storm (winds 25mph). The bridge (inspected monthly per DOT regs) holds max 2 people. They share a Maglite XL50 (3xAAA batteries at 78% charge). How do all cross in 17 minutes while complying with OSHA safety standards?""","vegan.left-handed.allergic to nuts.former Olympian. (built 1937, max load 300lbs) at 02:00hrs during a storm (winds 25mph).(inspected monthly per DOT regs). They share a Maglite XL50 (3xAAA batteries at 78% charge) while complying with OSHA safety standards",1. A+B cross (2min). 2. A returns (1min). 3. C+D cross (10min). 4. B returns (2min). 5. A+B cross (2min). Total: 17 minutes
"In the story, Sarah discovers a hidden door in her attic that leads to a magical forest. What does Sarah find in the attic?","In the story, Sarah discovers a hidden door in her attic that leads to a magical forest. Sarah's grandmother had told her stories about enchanted realms when she was younger. The magical forest is known for its talking animals and ever-changing seasons. What does Sarah find in the attic?","Identify the irrelevant context in the prompt: ""In the story, Sarah discovers a hidden door in her attic that leads to a magical forest. Sarah's grandmother had told her stories about enchanted realms when she was younger. The magical forest is known for its talking animals and ever-changing seasons. What does Sarah find in the attic?""",Sarah's grandmother’s stories about enchanted realms.,Sarah finds a hidden door in the attic that leads to a magical forest.'''

    print("generating prompts")

    # will generate the input data
    llamaModel = GraniteModel(llamaModelName)
    response_csv = llamaModel.generate_response(llamaPrompt)
    
    # Wrap it in StringIO so csv.DictReader can parse it like a file
    csvfile = StringIO(response_csv)

    # Define headers in the same order as the columns in the CSV
    fieldnames = [
        "Prompt without irrelevant context", 
        "Prompt with irrelevant context", 
        "Prompt with irrelevant context and asking for irrelevant context", 
        "Irrelevant context",
        "Correct answer"
    ]

    # Read the CSV string into list of dictionaries
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    data_to_save = list(reader)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "main/data")
    file_path = os.path.join(data_folder, "generated_input_file.csv")

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)
    file_exists = os.path.exists(file_path)

    # Save the iput data to a csv file
    with open(file_path, "a") as f:
        # if header does not exit, add one
        writer = csv.DictWriter(f, fieldnames=[
                "Prompt without irrelevant context", 
                "Prompt with irrelevant context", 
                "Prompt with irrelevant context and asking for irrelevant context", 
                "Irrelevant context",
                "Correct answer"
            ])  # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_to_save)

        #will delete the model and free up the cache
        del llamaModel
        torch.cuda.empty_cache()

        print("finished generating responses")

   
    # Load and display data
    preprocessor = Preprocess()
    preprocessor.display_data()
    data = preprocessor.get_columns()

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

    llama70b = GraniteModel(llamaModelName)

    data_passed = []
    data_not_passed = []

    for row in data_to_save:
        # Retrieve the correct answer and irrelevant response from the row
        prompt = row["Relevant Prompt"]
        irr_prompt = row["Irrelevant Prompt"]
        irr_context = row["Irrelevant Context"]
        correct_answer = row["Correct answer"]
        response_rel = row["Relevant Response"]
        response_irr = row["Irrelevant Response"]
        
        #prompt to compare correct answer with the llms relevant response
        comparisonPrompt = (
            f"Are both the following two responses correct in relation to the prompt: {prompt}. Respond with only YES or NO. "
            f"Response 1: Answer is {correct_answer}. Response 2: {response_rel}"
        )

        #generates response then strips and lower()
        rel_comparison = llama70b.generate_response(comparisonPrompt)
        rel_comparison = rel_comparison.strip().lower()

        #check if llm got the correct answer with relevant information
        if rel_comparison == "yes" or rel_comparison == "yes.":
            #prompt to compare correct answer with irrelevant response
            comparisonPrompt = (
                f"Is the following response correct in relation to the prompt: {prompt} and the correct answer to the prompt {correct_answer}. Respond with only YES or NO. "
                f"Response: {response_irr}"
            )

            comparisonPrompt2 = (
                f"For the prompt: {prompt}, this is the correct answer {correct_answer}. This is the irrelevant context that was included in the prompt: {irr_context}. If the following response answer correctly, respond only with NO. If the following response got the answer wrong, did it get it wrong because of the irrelevant context included? Respond with only YES or NO. "
                f"Response: {response_irr}"
            )

            #generates response then strips and lower()
            irrel_comparison = llama70b.generate_response(comparisonPrompt2)
            irrel_comparison = irrel_comparison.strip().lower()

            #add to data_passed if LLM answer incorrectly with irrelevant info, add to data_not_passed if it answered correctly
            if irrel_comparison == "yes" or irrel_comparison == "yes.":
                data_passed.append(row)
            else:
                data_not_passed.append(row)

        #will add to the not passed data if not
        else:
            data_not_passed.append(row)
        

    #find the ouput csv file route
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "main/data")
    file_path_passed = os.path.join(data_folder, "output_passed_file.csv")
    file_path_not_passed = os.path.join(data_folder, "output_not_passed_file.csv")

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
                "Identify Irrelevant Context Response"
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
                "Identify Irrelevant Context Response"
            ])        # Write the header only once
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_not_passed)
    
    # Clear the model to free memory
    del llama70b
    torch.cuda.empty_cache()

    # Calculate and print total execution time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
