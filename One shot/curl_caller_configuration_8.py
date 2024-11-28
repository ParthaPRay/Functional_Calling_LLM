# This curl caller code calls the simple functional call xecuted by 'simple_functaion_call_8.py'
# Partha Pratim Ray
# 15 November, 2024


import requests
import json

# API endpoint
api_url = "http://localhost:5000/process_prompt"

# Two test prompts for each function
# List of prompts with exactly two function instructions
prompts_with_two_functions = [
    "Find the factorial of 5.",
    "What is the factorial of 3?",
    "Get the coordinates of Los Angeles.",
    "Get coordinates of Kolkata",
    "Perform a heap sort on [10, 2, 7, 6].",
    "Heap sort these numbers: [4, 1, 3, 2].",
    "List files in the /var/log directory.",
    "List files in the /home/pi directory.",
    "What is the current time in Asia/Singapore?",
    "Current time in Asia/Kolkata?",
    "Exchange rate from GBP to INR",
    "Find exchange rate from INR to USD",
    "Analyze the sentiment of: I hate you",
    "Analyze the sentiment of: I love India my country",    
    "Read content of /home/pi/Desktop/requirements.txt",
    "Read content of /home/pi/Desktop/test.txt"
]

# Double the number of functions in the server code
# List of irrelevant prompts to get 'no_route' response 
irrelevant_prompts = [
    "What is the capital city of Australia?",
    "What is the boiling point of water?",
    "How do I fix a leaky faucet?",
    "Who painted the Mona Lisa?",
    "Tell me about black holes.",
    "How do I get to the nearest gas station?",
    "Can you help me with my homework?",
    "How many continents are there?",
    "What are the symptoms of the common cold?",
    "Tell me about black holes.",
    "How do I get to the nearest gas station?",
    "Can you help me with my homework?",
    "How many continents are there?",    
    "What are the symptoms of the common cold?",
    "How do I change my email password?",
    "Tell me a story."   
]


    # "How do I set up a new printer?",
    # "What is the best way to learn a new language?",
    # "What is the difference between a noun and a verb?",
    # "Explain Newton laws of motion.",
    # "What is the capital of Egypt?",
    # "What is the meaning of the word serendipity?",
    # "What is the recipe for pancakes?",
    # "Who is the author of 1984 book?",
    # "What is the latest score in the baseball game?" 
    # "What is the speed of light?",
    # "How do you say hello in Spanish?",    
    # "What is on TV tonight?",
    # "Tell me the plot of Hamlet.",
    # "Can you recommend a good book?",
    # "Explain the Pythagorean theorem.",
    # "Who won the Nobel Prize in Physics in 2020?"    
    # "Explain quantum physics"
    # "Who won the football match yesterday?"
    # "How do airplanes fly?"
    # "Define the meaning of life"

# Combine the prompts
all_prompts = prompts_with_two_functions + irrelevant_prompts

# Function to send a POST request to the API
def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        result = response.json()
        return (prompt, result)
    except Exception as e:
        return (prompt, f"Error: {str(e)}")

# Main function to execute the script
if __name__ == "__main__":
    # Loop through each prompt and send to the API
    for prompt in all_prompts:
        prompt_text, response = send_prompt(prompt)
        print(f"Prompt: {prompt_text}")
        print(f"Response: {json.dumps(response, indent=2)}\n")
