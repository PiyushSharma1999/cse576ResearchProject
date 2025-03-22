# not actually using this file, just a temporary file to test out some code


import csv

def load_dataset(csv_file_path):
    entries = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            entry = {
                "domain": row['Domain'],
                "model_prompt": row['Model prompt is generated from'],
                "prompt_without_context": row['Prompt without irrelevant context'],
                "prompt_with_context": row['Prompt with irrelevant context'],
                "small_llm_used": row['Small LLM used'],
                "small_llm_parameters": row['Small LLM parameters'],
                "response_without_context": row['Response from small LLM without irrelevant context'],
                "response_with_context": row['Response from small LLM with irrelevant context'],
                "correct_answer": row['Correct answer'],
                "llm_correct_without_context": row['LLM answer correctly without irrelevant context?'],
                "llm_correct_with_context": row['LLM answer correctly with irrelevant context?'],
                "context_effect": row['How did the irrelevant context affect the response?'],
                "use_in_final_dataset": row['Use example in final dataset? (yes or no)']
            }
            entries.append(entry)
    return entries

# 900mil Parameter models
model_900mil = [
    "gpt2-medium",      # 345M parameters
    "gpt2-large",       # 762M parameters
    "gpt2-xl",          # 1.5B parameters
    "EleutherAI/gpt-neo-1.3B",  # 1.3B (closer to 1B, could be in the 900M range)
]

# 1bil parameter models
model_1bil = [
    "gpt-neo-1.3B",     # 1.3B parameters
    "EleutherAI/gpt-neo-1.3B", # 1.3B parameters
    "EleutherAI/gpt-neo-2.7B", # 2.7B parameters
    "babbage",           # OpenAI's Babbage model, closer to 1B range
    "curie",             # OpenAI's Curie model, a bit larger than 1B
    "davinci",           # OpenAI's Davinci, but typically much larger
]

# 7bil parameter models
model_7bil = [
    "EleutherAI/gpt-neo-2.7B", # 2.7B parameters (closer to 7B range)
    "gpt3-ada-13B",          # 13B parameters
    "EleutherAI/gpt-j-6B",    # 6B parameters
    "Cushman-7B",            # Approximate 7B parameter model
    "davinci-002",           # OpenAI's Davinci models, typically in the 10-175B range, but Davinci-002 could be relevant
]

def process(entry):
    # only needs prompt with and prompt without context and correct answer
    relevant_prompt = entry["prompt_without_context"]
    irrelevant_prompt = entry["prompt_with_context"]
    correct_answer = entry["correct_answer"]


csv_file_path = r"custom_dataset.csv"
dataset = load_dataset(csv_file_path)
print(len(dataset))

for entry in dataset:
    process(entry)