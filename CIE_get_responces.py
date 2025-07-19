import json

# Load the exported Postman run results JSON
with open('postman_run_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Loop through each execution in the run
for i, execution in enumerate(data.get('run', {}).get('executions', []), start=1):
    # Extract the raw request body (JSON string inside "raw")
    raw_request = execution.get('request', {}).get('body', {}).get('raw', '')
    
    # Extract the response body (string, usually JSON string)
    response_body = execution.get('response', {}).get('body', '')

    print(f"Execution #{i}:")
    print("Raw Request Body:")
    print(raw_request)
    print("Response Body:")
    print(response_body)
    print('-' * 40)
