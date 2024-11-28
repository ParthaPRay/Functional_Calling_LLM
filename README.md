# functional_calling_LLM
This repo shows the codes of function calling

Two Approaches of function calling involvng localized quantized LLMs have been used in Raspberry PI 4B with FastAPI + Ollama frameworks:

# One Shot

We use one shot approach (examples) to instruct LLM how to respond for function calling.

1. Make a virtual environment.
2. First run python3 simple_function_call_X.py in one terminal under the virtual envirnment.
3. Run then curl_caller_configuration_X.py in other terminal (with / without virtual environment). cURL should be used for such RESTful API call.

Here, **X** denotes the configuration such as 2, 4, 6, 8 for repeatative study varying LLMs.   

# Few Shot

We use few shot approach (examples) to instruct LLM how to respond for function calling.

1. Make a virtual environment.
2. First run python3 simple_function_call_X.py in one terminal under the virtual envirnment.
3. Run then curl_caller_configuration_X.py in other terminal (with / without virtual environment). cURL should be used for such RESTful API call.

Here, **X** denotes the configuration such as 2, 4, 6, 8 for repeatative study varying LLMs.   
