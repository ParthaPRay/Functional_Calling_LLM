# Simple Function Calling
# Partha Pratim Ray, Sikkim University
# 15/11/2024
#
# Configuration 4
# One shot

from fastapi import FastAPI
import numpy as np
import requests
import threading
import psutil
import time
import csv
import os
from pydantic import BaseModel
from queue import Queue
from statistics import mean
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess

# Curl commands for testing:

# Mathematical Operations

# Add: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Add 7 and 8."}'
# Mutiply: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Multiply 6 by 9."}'
# Factorial: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Find the factorial of 6."}'

# Algorithms and Data Structures

# Heap sort: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Sort the numbers [10, 3, 2, 8] using heap sort."}'

# Natural Language Processing

# Analyze sentiment: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Analyze the sentiment of: This is the best day ever! "}'

# External API Interaction

# Get coordinates: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Get the coordinates of Los Angeles."}'
# Get exchange rates: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "What is the exchange rate from GBP to USD?"}'

# Time and Date Handling

# Get current in city: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "What time is it now in Asia/Singapore?"}'

# System and OS Operations

# List files directories: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "List files in the /var/log directory."}'
# Read files in a path: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Read the content of /etc/hostname."}'



app = FastAPI()

# Define the threshold for similarity score 
THRESHOLD = 0.4950  # Adjust based on embedding model

# Define the embedding model and LLM
embed_model = "all-minilm:33m"  # Embedding model
model_name = "qwen2.5:0.5b-instruct"  # LLM for dynamic routes  # qwen2.5:0.5b-instruct  # llama3.2:1b-instruct-q4_K_M  # smollm2:1.7b-instruct-q4_K_M 
OLLAMA_API_URL = "http://localhost:11434/api/embed"
OLLAMA_LLM_URL = "http://localhost:11434/api/chat"

# CSV file setup
csv_file = 'simple_function_call_logs.csv'
csv_headers = [
    'timestamp', 'model_name', 'embed_model', 'prompt', 'response', 'route_type', 'route_selected',
    'semantic_similarity_score', 'similarity_metric', 'vector', 'total_duration', 
    'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 
    'eval_duration', 'tokens_per_second', 'avg_cpu_usage_during', 'memory_usage_before', 
    'memory_usage_after', 'memory_allocated_for_model', 'network_latency', 'total_response_time', 
    'route_selection_duration', 'llm_invoked', 'function_execution_time_total_ns', 
    'function_execution_times_ns', 'llm_response_parsing_duration_ns', 'number_of_functions_called'
]

csv_queue = Queue()
cpu_usage_queue = Queue()
memory_usage_queue = Queue()
is_monitoring = False
memory_allocated_for_model = 0

# CSV writer thread to log the data
def csv_writer():
    while True:
        log_message_csv = csv_queue.get()
        if log_message_csv is None:  # Exit signal
            break
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(csv_headers)
            writer.writerow(log_message_csv)

csv_thread = threading.Thread(target=csv_writer)
csv_thread.start()

class Prompt(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    # Measure memory usage for the model
    global memory_allocated_for_model
    memory_allocated_for_model = load_model_and_measure_memory(model_name)
    print(f"Memory Allocated for Model '{model_name}': {memory_allocated_for_model / (1024 * 1024):.2f} MB")

# Function to measure memory allocated after loading the model
def load_model_and_measure_memory(model_name):
    # Load the model by making an empty prompt request
    payload = {
        "model": model_name,
        "prompt": "",
        "stream": False
    }
    response = requests.post(OLLAMA_LLM_URL, json=payload)
    if response.status_code == 200:
        print(f"Model '{model_name}' loaded successfully.")
    else:
        print(f"Failed to load model '{model_name}'. Response: {response.text}")

    # Get the list of loaded models using the /api/ps endpoint
    ps_response = requests.get("http://localhost:11434/api/ps")
    if ps_response.status_code == 200:
        models_info = ps_response.json().get('models', [])
        for model_info in models_info:
            if model_info.get('name') == model_name:
                model_size = model_info.get('size', 0)
                return model_size  # Size is in bytes
        print(f"Model '{model_name}' not found in the loaded models.")
        return 0
    else:
        print(f"Failed to retrieve models using /api/ps. Response: {ps_response.text}")
        return 0

# Modified Route class to include functions and function_schemas
class Route:
    def __init__(self, name, utterances, responses=None, dynamic=False, function_schemas=None, functions=None):
        self.name = name
        self.utterances = utterances
        self.responses = responses  # Predefined static responses
        self.dynamic = dynamic
        self.function_schemas = function_schemas or []  # For dynamic routes
        self.functions = functions or []  # For dynamic routes

# Define functions for dynamic routes

def factorial(n: int) -> int:
    """Compute factorial of a number."""
    from math import factorial as math_factorial
    return math_factorial(n)


def get_coordinates(city_name: str) -> dict:
    """Retrieve the latitude and longitude for a specified city using the OpenWeatherMap Geocoding API."""
    try:
        # Replace with your actual API key
        api_key = "4a265265d7ea421d0cc3f782ad8ba67e"
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = data[0]["lat"]
                lon = data[0]["lon"]
                return {"latitude": lat, "longitude": lon}
            else:
                return {"error": f"No data found for the city: {city_name}"}
        else:
            return {"error": f"Error: Unable to fetch data for {city_name}. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}



def heap_sort(numbers: list) -> list:
    """Perform heap sort on a list of numbers."""
    import heapq
    heapq.heapify(numbers)
    return [heapq.heappop(numbers) for _ in range(len(numbers))]


def list_files(directory: str) -> list:
    """List files in a directory."""
    import os
    try:
        return os.listdir(directory)
    except Exception as e:
        return f"Error listing files: {e}"
        
        

# Define function schemas

factorial_schema = {
    "name": "factorial",
    "description": "Compute factorial of a number.",
    "parameters": {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "The number to compute factorial of."
            }
        },
        "required": ["n"]
    }
}


get_coordinates_schema = {
    "name": "get_coordinates",
    "description": "Get the latitude and longitude of a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city_name": {
                "type": "string",
                "description": "The name of the city."
            }
        },
        "required": ["city_name"]
    }
}


heap_sort_schema = {
    "name": "heap_sort",
    "description": "Sort a list of numbers using heap sort.",
    "parameters": {
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The list of numbers to sort."
            }
        },
        "required": ["numbers"]
    }
}

list_files_schema = {
    "name": "list_files",
    "description": "List files in a directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "The directory to list files from."
            }
        },
        "required": ["directory"]
    }
}



# Define routes with static and dynamic options
routes = [
    Route(
        name="dynamic_functions",
        utterances=[
            "Find the factorial of 7",
            "Get coordinates of New York",
            "Heap sort these numbers [6, 4, 2, 8]",
            "List files in the /home/pi directory"            
        ],
        dynamic=True,
        functions=[factorial, get_coordinates, heap_sort, list_files],
        function_schemas=[factorial_schema, get_coordinates_schema, heap_sort_schema, list_files_schema]
    )
]

# Function to create multi-shot prompt for the LLM
def create_multi_shot_prompt(prompt: str, route: Route) -> str:
    examples = ""
    if route.name == "dynamic_functions":
        examples = """
    Example 1:
    User: Find the factorial of 6
    Assistant:
    Function: factorial
    Arguments: {"n": 6}
    Result: 720

    Example 2:
    User: Get coordinates of New York
    Assistant:
    Function: get_coordinates
    Arguments: {"city_name": "New York"}
    Result: {"latitude": 40.7128, "longitude": -74.0060}

    Example 3:
    User: Heap sort these numbers: [10, 2, 7, 6]
    Assistant:
    Function: heap_sort
    Arguments: {"numbers": [10, 2, 7, 6]}
    Result: [2, 6, 7, 10]

    Example 4:
    User: List files in the /home/user directory
    Assistant:
    Function: list_files
    Arguments: {"directory": "/home/user"}
    Result: ["file1.txt", "file2.txt", "documents"]

    """
    else:
        # Handle other routes or provide default examples
        pass

    prompt_template = examples + f"""
    Now, respond to this query:
    User: {prompt}
    Assistant:
    """
    return prompt_template

# Resource monitoring thread to track CPU and memory usage
def monitor_resources():
    global is_monitoring
    process = psutil.Process()
    while is_monitoring:
        cpu_usage = psutil.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss  # Memory in bytes
        cpu_usage_queue.put(cpu_usage)
        memory_usage_queue.put(memory_usage)
        time.sleep(0.01)  # Poll every 10ms

# Function to call LLM with multi-shot prompt and extract metrics
def call_llm_with_multi_shot(prompt, route):
    response = requests.post(
        OLLAMA_LLM_URL,
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False}
    )
    
    response_json = response.json()

    # Check if the expected keys are present in the response
    if 'message' in response_json and 'content' in response_json['message']:
        generated_response = response_json['message']['content']
    else:
        generated_response = "Error: Unexpected response structure from LLM"

    # Measure the time to parse LLM output
    parse_start_time = time.time()
    # Parse the LLM's response to extract function calls and arguments
    function_calls = parse_llm_response(generated_response, route.functions)
    parse_end_time = time.time()
    llm_response_parsing_duration = (parse_end_time - parse_start_time) * 1e9  # Convert to nanoseconds

    response_texts = []
    
    # Initialize list to hold function execution times
    function_execution_times = []
    
    # Number of functions called
    number_of_functions_called = len(function_calls)
    
    # Process each function call in sequence
    for function, arguments in function_calls:
        if function and arguments is not None:
            # Measure the function execution time
            func_start_time = time.time()
            try:
                result = function(**arguments)
                func_end_time = time.time()
                execution_time = (func_end_time - func_start_time) * 1e9  # Convert to nanoseconds
                function_execution_times.append({
                    'function_name': function.__name__,
                    'execution_time_ns': execution_time
                })
                response_texts.append(f"Function: {function.__name__}\nResult: {result}")
            except Exception as e:
                func_end_time = time.time()
                execution_time = (func_end_time - func_start_time) * 1e9  # Convert to nanoseconds
                function_execution_times.append({
                    'function_name': function.__name__,
                    'execution_time_ns': execution_time
                })
                response_texts.append(f"Function: {function.__name__}\nError executing function: {e}")
        else:
            response_texts.append(f"Could not parse function call from LLM response.")

    # Combine results from all tasks
    response_text = '\n'.join(response_texts)

    # Extract the relevant metrics
    total_duration = response_json.get('total_duration', 0)
    load_duration = response_json.get('load_duration', 0)
    prompt_eval_count = response_json.get('prompt_eval_count', 0)
    prompt_eval_duration = response_json.get('prompt_eval_duration', 0)
    eval_count = response_json.get('eval_count', 0)
    eval_duration = response_json.get('eval_duration', 1)  # Avoid division by zero

    # Return the response and extracted metrics
    return {
        "generated_response": response_text,
        "metrics": {
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "function_execution_times": function_execution_times,
            "llm_response_parsing_duration_ns": llm_response_parsing_duration,
            "number_of_functions_called": number_of_functions_called
        }
    }

# Function to parse LLM response and map to valid functions
def parse_llm_response(response_text, available_functions):
    # Extract multiple function calls from the LLM response
    lines = response_text.strip().split('\n')
    function_calls = []
    function = None
    arguments = None

    # Map function names to the actual function objects
    function_mapping = {func.__name__: func for func in available_functions}

    for line in lines:
        if line.startswith('Function:'):
            if function and arguments is not None:
                function_calls.append((function, arguments))
            raw_function_name = line[len('Function:'):].strip()
            function = function_mapping.get(raw_function_name, None)
            arguments = None
        elif line.startswith('Arguments:'):
            args_text = line[len('Arguments:'):].strip()
            try:
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                arguments = None

    if function and arguments is not None:
        function_calls.append((function, arguments))  # Append the last function call

    return function_calls

# Main route processing API endpoint
@app.post("/process_prompt")
async def process_prompt(request: Prompt):
    global is_monitoring
    start_time = time.time()
    data = request.dict()
    prompt = data['prompt']
    
    llm_invoked = 0  # Initialize llm_invoked to 0 to know whether llm is invoked or not, useful for dynamic routing

    try:
        # Start resource monitoring
        is_monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Capture the start time for route selection
        route_start_time = time.time()

        # Get embedding for the prompt
        prompt_embedding, embed_metrics = get_embedding(prompt)

        # Find the best route (Static or Dynamic) based on the prompt
        best_route, similarity = find_best_route(prompt_embedding, routes)

        # Calculate the time taken for route selection
        route_selection_duration = (time.time() - route_start_time) * 1e9  # Convert to nanoseconds

        # Check if the similarity score is below the threshold
        if similarity < THRESHOLD:
            best_route = None

        if best_route is None:
            print("No matching route found.")
            route_name = "no_route"
            response = "No route found"
            route_type = "none"
            route_selected = route_name
            # Default values for missing metrics in case no route is found
            prompt_eval_duration = 0
            eval_count = 0
            eval_duration = 0
            prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
            total_duration = embed_metrics.get('total_duration', 0)
            load_duration = embed_metrics.get('load_duration', 0)
            function_execution_time_total = 0
            function_execution_times = []
            llm_response_parsing_duration = 0
            number_of_functions_called = 0
            print(f"No Route Response: {response}")  # Debugging print statement
        else:
            print(f"Selected Route: {best_route.name} with similarity: {similarity}")
            route_name = best_route.name
            route_selected = route_name
            route_type = "dynamic" if best_route.dynamic else "static"

            if best_route.dynamic:
                llm_invoked = 1  # Set llm_invoked to 1 since LLM is called
                # For dynamic routes, trigger LLM with multi-shot prompt
                multi_shot_prompt = create_multi_shot_prompt(prompt, best_route)
                llm_response = call_llm_with_multi_shot(multi_shot_prompt, best_route)
                response = llm_response['generated_response']
                # Extract the metrics for dynamic routes
                dynamic_metrics = llm_response['metrics']
                prompt_eval_duration = dynamic_metrics['prompt_eval_duration']
                eval_count = dynamic_metrics['eval_count']
                eval_duration = dynamic_metrics['eval_duration']
                prompt_eval_count = dynamic_metrics['prompt_eval_count']
                total_duration = dynamic_metrics['total_duration']
                load_duration = dynamic_metrics['load_duration']
                function_execution_times = dynamic_metrics.get('function_execution_times', [])
                function_execution_time_total = sum([entry['execution_time_ns'] for entry in function_execution_times])
                llm_response_parsing_duration = dynamic_metrics.get('llm_response_parsing_duration_ns', 0)
                number_of_functions_called = dynamic_metrics.get('number_of_functions_called', 0)
                print(f"Dynamic LLM Response: {response}")  # Debugging print statement
            else:
                # For static routes, log the predefined response based on route
                predefined_responses = best_route.responses
                # Find the closest response based on similarity to utterances
                similarities = []
                for utt in best_route.utterances:
                    utt_embedding, _ = get_embedding(utt)
                    sim = cosine_similarity(prompt_embedding, utt_embedding)
                    similarities.append(sim)
                closest_utterance_index = np.argmax(similarities)
                response = predefined_responses[closest_utterance_index]
                # Static responses have default values for dynamic metrics
                prompt_eval_duration = 0
                eval_count = 0
                eval_duration = 0
                prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
                total_duration = embed_metrics.get('total_duration', 0)
                load_duration = embed_metrics.get('load_duration', 0)
                function_execution_time_total = 0
                function_execution_times = []
                llm_response_parsing_duration = 0
                number_of_functions_called = 0
                print(f"Static Route Response: {response}")  # Debugging print statement

        # Stop resource monitoring
        is_monitoring = False
        monitor_thread.join()

        # Measure resource statistics
        process = psutil.Process()
        memory_usage_before = memory_usage_queue.queue[0] if not memory_usage_queue.empty() else process.memory_info().rss
        memory_usage_after = memory_usage_queue.queue[-1] if not memory_usage_queue.empty() else process.memory_info().rss
        avg_cpu_usage = calculate_average_cpu()
        similarity = round(similarity, 2) if similarity is not None else None

        # Network latency: time spent in network communication for the embedding request
        network_latency = total_duration - load_duration

        # Total response time: time from receiving the request to sending the response
        total_response_time = (time.time() - start_time) * 1e9  # Convert to nanoseconds

        # Calculate tokens per second for the response
        tokens_per_second = eval_count / eval_duration * 1e9 if eval_duration > 0 else 0
        tokens_per_second = round(tokens_per_second, 2)  # Round to 2 decimal points

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, model_name, embed_model, prompt, response, route_type, route_selected,
            similarity, "cosine", str(prompt_embedding), total_duration, load_duration, prompt_eval_count,
            prompt_eval_duration, eval_count, eval_duration, tokens_per_second,
            avg_cpu_usage, memory_usage_before, memory_usage_after, memory_allocated_for_model,
            network_latency, total_response_time, route_selection_duration, llm_invoked,
            function_execution_time_total, json.dumps(function_execution_times),
            llm_response_parsing_duration, number_of_functions_called
        ]

        # Put the log message into the CSV queue
        print(f"Logging to CSV: {log_message_csv}")  # Debugging print statement
        csv_queue.put(log_message_csv)

        # Return the response to the client
        return {
            "status": "success" if best_route else "no_match",
            "route_selected": route_name,
            "semantic_similarity_score": similarity,
            "similarity_metric": "cosine",
            "response": response
        }

    except Exception as e:
        is_monitoring = False  # Ensure monitoring stops in case of an error
        return {"status": "error", "message": str(e)}

# Function to get embeddings from the embedding model
def get_embedding(text, model=embed_model):
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": model, "input": text}
    )
    response_json = response.json()
    return response_json["embeddings"][0], response_json

# Function to calculate average CPU usage during the request
def calculate_average_cpu():
    cpu_usages = []
    while not cpu_usage_queue.empty():
        cpu_usages.append(cpu_usage_queue.get())
    return round(mean(cpu_usages), 2) if cpu_usages else 0

# Function to find the best route based on the prompt
def find_best_route(prompt_embedding, routes):
    best_route = None
    best_similarity = -1

    for route in routes:
        for utterance in route.utterances:
            utterance_embedding, _ = get_embedding(utterance)
            similarity = cosine_similarity(prompt_embedding, utterance_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route

    # If best similarity is below the threshold, set best_route to None
    if best_similarity < THRESHOLD:
        best_route = None

    return best_route, best_similarity

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
