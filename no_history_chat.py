import boto3
import json

client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')  

titan_model_id = 'amazon.titan-text-express-v1'

history = []

def get_history():
    return "\n".join(history)


def get_configuration(prompt:str):
    return json.dumps({
        "inputText": get_history(),
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 0.5,
        }
    })

print("BOT: Hello! I am a chatbot. How can I assist you today?")

while True:
    user_input = input("USER: ")
    history.append(f"USER: {user_input}")
    if user_input.lower() == "exit":
        print("BOT: Goodbye")
        break
    response = client.invoke_model(
        body=get_configuration(user_input),
        modelId=titan_model_id,
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response['body'].read())
    output_text = response_body.get('results')[0].get('outputText').strip()
    print(f"BOT: {output_text}")
    history.append(output_text)


