import openai
import re
from prettytable import PrettyTable
from tqdm import tqdm
import itertools
import wandb
import json
from tenacity import retry, stop_after_attempt, wait_exponential

openai.api_key = "sk-xpA4UDDjmVQU8oRp0cA2EaC9111f401483A2E9E957C02dBb"
openai.base_url = "http://127.0.0.1:33000/v1/"

use_wandb = False # set to True if you want to use wandb to log your config and results

# --------------------------------------------------------------------------------

# K is a constant factor that determines how much ratings change
K = 32

CANDIDATE_MODEL = 'claude-3-opus-20240229'
# CANDIDATE_MODEL = 'claude-3-haiku-20240307'
# CANDIDATE_MODEL = 'glm-4'
CANDIDATE_MODEL_TEMPERATURE = 0.9

# GENERATION_MODEL = 'claude-3-opus-20240229'
GENERATION_MODEL = 'glm-4'
GENERATION_MODEL_TEMPERATURE = 0.8
GENERATION_MODEL_MAX_TOKENS = 800

# TEST_CASE_MODEL = 'claude-3-opus-20240229'
# TEST_CASE_MODEL = 'glm-4'
# TEST_CASE_MODEL_TEMPERATURE = 0.8

# NUMBER_OF_TEST_CASES = 10 # this determines how many test cases to generate... the higher, the more expensive, but the better the results will be
NUMBER_OF_TEST_CASES = 3 # this determines how many test cases to generate... the higher, the more expensive, but the better the results will be

N_RETRIES = 3  # number of times to retry a call to the ranking model if it fails
# RANKING_MODEL = 'claude-3-opus-20240229'
RANKING_MODEL = 'claude-3-haiku-20240307'
RANKING_MODEL_TEMPERATURE = 0.5

NUMBER_OF_PROMPTS = 3 # this determines how many candidate prompts to generate... the higher, the more expensive, but the better the results will be

# WANDB_PROJECT_NAME = "claude-prompt-eng" # used if use_wandb is True, Weights &| Biases project name
WANDB_PROJECT_NAME = "one-api-prompt-eng" # used if use_wandb is True, Weights &| Biases project name
WANDB_RUN_NAME = None # used if use_wandb is True, optionally set the Weights & Biases run name to identify this run

# --------------------------------------------------------------------------------

def start_wandb_run():
  # start a new wandb run and log the config
  wandb.init(
    project=WANDB_PROJECT_NAME,
    name=WANDB_RUN_NAME,
    config={
      "K": K,
      "candiate_model": CANDIDATE_MODEL,
      "candidate_model_temperature": CANDIDATE_MODEL_TEMPERATURE,
      "generation_model": GENERATION_MODEL,
      "generation_model_temperature": GENERATION_MODEL_TEMPERATURE,
      "generation_model_max_tokens": GENERATION_MODEL_MAX_TOKENS,
      "n_retries": N_RETRIES,
      "ranking_model": RANKING_MODEL,
      "ranking_model_temperature": RANKING_MODEL_TEMPERATURE,
      "number_of_prompts": NUMBER_OF_PROMPTS
      })

  return

# Optional logging to Weights & Biases to reocrd the configs, prompts and results
if use_wandb:
  start_wandb_run()

# --------------------------------------------------------------------------------

def remove_first_line(test_string):
    if test_string.startswith("Here") and test_string.split("\n")[0].strip().endswith(":"):
        print("test_string startswith Here", test_string)
        return re.sub(r'^.*\n', '', test_string, count=1)
    return test_string

def generate_candidate_prompts(description, input_variables, test_cases, number_of_prompts):
    variable_descriptions = "\n".join(f"{var['variable']}: {var['description']}" for var in input_variables)

    messages = [{
        "role": "system",
        "content": f"""Your job is to generate system prompts for Claude 3, given a description of the use-case, some test cases/input variable examples that will help you understand what the prompt will need to be good at.
The prompts you will be generating will be for freeform tasks, such as generating a landing page headline, an intro paragraph, solving a math problem, etc.
In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output.
<most_important>Make sure to incorporate the provided input variable placeholders into the prompt, using placeholders like {{{{VARIABLE_NAME}}}} for each variable. Ensure you place placeholders inside four squiggly lines like {{{{VARIABLE_NAME}}}}. At inference time/test time, we will slot the variables into the prompt, like a template.</most_important>
Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.
You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.
Here are the input variables and their descriptions:
{variable_descriptions}
Most importantly, output NOTHING but the prompt (with the variables contained in it like {{{{VARIABLE_NAME}}}}). Do not include anything else in your message.""",
    }, {
        "role": "user",
        "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your flexible system prompt, and nothing else. Be creative, and remember, the goal is not to complete the task, but write a prompt that will complete the task."
    }]

    prompts = []

    for i in range(number_of_prompts):
        response = openai.chat.completions.create(
            model=CANDIDATE_MODEL,
            max_tokens=1500,
            temperature=CANDIDATE_MODEL_TEMPERATURE,
            messages=messages,
        )
        response_text = response.choices[0].message.content

        prompts.append(remove_first_line(response_text))

    return prompts

def expected_score(r1, r2):
    return 1 / (1 + 10**((r2 - r1) / 400))

def update_elo(r1, r2, score1):
    e1 = expected_score(r1, r2)
    e2 = expected_score(r2, r1)
    return r1 + K * (score1 - e1), r2 + K * ((1 - score1) - e2)

# Get Score - retry up to N_RETRIES times, waiting exponentially between retries.
@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def get_score(description, test_case, pos1, pos2, input_variables, ranking_model_name, ranking_model_temperature):
    variable_values = "\n".join(f"{var['variable']}: {test_case.get(var['variable'], '')}" for var in input_variables)

    messages = [{
        "role": "system",
        "content": f"""Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.
You will be provided with the task description, input variable values, and two generations - one for each system prompt.
Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.
Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.
Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.
Respond with your ranking ('A' or 'B'), and nothing else. Be fair and unbiased in your judgement."""
    }, {
        "role": "user",
        "content": f"""Task: {description.strip()}
Variables: {test_case['variables']}
Generation A: {remove_first_line(pos1)}
Generation B: {remove_first_line(pos2)}"""
    }]

    response = openai.chat.completions.create(
        model=RANKING_MODEL,
        max_tokens=1,
        temperature=ranking_model_temperature,
        messages=messages,
    )

    score = response.choices[0].message.content

    return score

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def get_generation(prompt, test_case, input_variables):
    # Replace variable placeholders in the prompt with their actual values from the test case
    for var_dict in test_case['variables']:
        for variable_name, variable_value in var_dict.items():
            prompt = prompt.replace(f"{{{{{variable_name}}}}}", variable_value)

    messages = [{
        "role": "system",
        "content": 'Complete the task perfectly.'
    }, {
        "role": "user",
        "content": prompt
    }]

    response = openai.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=GENERATION_MODEL_MAX_TOKENS,
        temperature=GENERATION_MODEL_TEMPERATURE,
        messages=messages,
    )
    generation = response.choices[0].message.content

    return generation

def test_candidate_prompts(test_cases, description, input_variables, prompts):
    # Initialize each prompt with an ELO rating of 1200
    prompt_ratings = {prompt: 1200 for prompt in prompts}

    # Calculate total rounds for progress bar
    total_rounds = len(test_cases) * len(prompts) * (len(prompts) - 1) // 2

    # Initialize progress bar
    pbar = tqdm(total=total_rounds, ncols=70)

    # For each pair of prompts
    for prompt1, prompt2 in itertools.combinations(prompts, 2):
        # For each test case
        for test_case in test_cases:
            # Update progress bar
            pbar.update()

            # Generate outputs for each prompt
            generation1 = get_generation(prompt1, test_case, input_variables)
            generation2 = get_generation(prompt2, test_case, input_variables)

            # Rank the outputs
            score1 = get_score(description, test_case, generation1, generation2, input_variables, RANKING_MODEL, RANKING_MODEL_TEMPERATURE)
            score2 = get_score(description, test_case, generation2, generation1, input_variables, RANKING_MODEL, RANKING_MODEL_TEMPERATURE)

            # Convert scores to numeric values
            score1 = 1 if score1 == 'A' else 0 if score1 == 'B' else 0.5
            score2 = 1 if score2 == 'B' else 0 if score2 == 'A' else 0.5

            # Average the scores
            score = (score1 + score2) / 2

            # Update ELO ratings
            r1, r2 = prompt_ratings[prompt1], prompt_ratings[prompt2]
            r1, r2 = update_elo(r1, r2, score)
            prompt_ratings[prompt1], prompt_ratings[prompt2] = r1, r2

            # Print the winner of this round
            if score > 0.5:
                print(f"Winner: {prompt1}")
            elif score < 0.5:
                print(f"Winner: {prompt2}")
            else:
                print("Draw")

    # Close progress bar
    pbar.close()

    return prompt_ratings

def generate_optimal_prompt(description, input_variables, num_test_cases=10, number_of_prompts=10, use_wandb=False):
    if use_wandb:
        wandb_table = wandb.Table(columns=["Prompt", "Ranking"] + [var["variable"] for var in input_variables])
        if wandb.run is None:
            start_wandb_run()

    test_cases = generate_test_cases(description, input_variables, num_test_cases)
    prompts = generate_candidate_prompts(description, input_variables, test_cases, number_of_prompts)
    print('Here are the possible prompts:', prompts)
    prompt_ratings = test_candidate_prompts(test_cases, description, input_variables, prompts)

    table = PrettyTable()
    table.field_names = ["Prompt", "Rating"] + [var["variable"] for var in input_variables]
    for prompt, rating in sorted(prompt_ratings.items(), key=lambda item: item[1], reverse=True):
        # Use the first test case as an example for displaying the input variables
        example_test_case = test_cases[0]
        table.add_row([prompt, rating] + [example_test_case.get(var["variable"], "") for var in input_variables])
        if use_wandb:
            wandb_table.add_data(prompt, rating, *[example_test_case.get(var["variable"], "") for var in input_variables])

    if use_wandb:
        wandb.log({"prompt_ratings": wandb_table})
        wandb.finish()
    print(table)

def generate_test_cases(description, input_variables, num_test_cases):
    variable_descriptions = "\n".join(f"{var['variable']}: {var['description']}" for var in input_variables)

    messages = [{
        "role": "system",
        "content": f"""You are an expert at generating test cases for evaluating AI-generated content.
Your task is to generate a list of {num_test_cases} test case prompts based on the given description and input variables.
Each test case should be a JSON object with a 'test_design' field containing the overall idea of this test case, and a list of additional JSONs for each input variable, called 'variables'.
The test cases should be diverse, covering a range of topics and styles relevant to the description.
Here are the input variables and their descriptions:
{variable_descriptions}
Return the test cases as a JSON list, with no other text or explanation.""",
    }, {
        "role": "user",
        "content": f"Description: {description.strip()}\n\nGenerate the test cases. Make sure they are really, really great and diverse:"
    }]

    response = openai.chat.completions.create(
        model=CANDIDATE_MODEL,
        max_tokens=1500,
        temperature=CANDIDATE_MODEL_TEMPERATURE,
        messages=messages,
    )
    response_text = response.choices[0].message.content

    test_cases = json.loads(response_text)

    print('Here are the test cases:', test_cases)

    return test_cases

# --------------------------------------------------------------------------------

## Example usage
description = "Given a prompt, generate a personalized email response." # this style of description tends to work well

input_variables = [
    {"variable": "SENDER_NAME", "description": "The name of the person who sent the email."},
    {"variable": "RECIPIENT_NAME", "description": "The name of the person receiving the email."},
    {"variable": "TOPIC", "description": "The main topic or subject of the email. One to two sentences."}
]

# --------------------------------------------------------------------------------

if use_wandb:
    wandb.config.update({"description": description,
                         "input_variables": input_variables,
                         "num_test_cases": NUMBER_OF_TEST_CASES,
                         "number_of_prompts": NUMBER_OF_PROMPTS})

# --------------------------------------------------------------------------------

generate_optimal_prompt(description, input_variables, NUMBER_OF_TEST_CASES, NUMBER_OF_PROMPTS, use_wandb)