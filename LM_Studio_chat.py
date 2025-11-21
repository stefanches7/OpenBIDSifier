import requests
import json

url = "http://localhost:1234/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# Initialize conversation history with the system message
conversation_history = [
    {
        "role": "system",
        "content": "Be a helpful assistant. Be concise."
    }
]

while True:
    # Get user input
    user_input = input("You: ")

    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        print("Ending conversation...")
        break

    # Add user's message to the conversation history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # Prepare the data for the API call
    data = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b",
        "messages": conversation_history,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }

    # Make the POST request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Get the model's response
    model_response = response.json()

    # Extract and print the last model response (the assistant's content)
    last_message = model_response['choices'][0]['message']['content']
    print(f"Model: {last_message}")

    # Add model's response to the conversation history for the next round
    conversation_history.append({
        "role": "assistant",
        "content": last_message
    })
