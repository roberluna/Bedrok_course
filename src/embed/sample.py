import boto3
import json

client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

fact = "The Eiffel Tower is located in Paris."

animal = "Dog"

response = client.invoke_model(
    body=json.dumps({
        "inputText": animal,
    }),
    modelId='amazon.titan-embed-text-v1',
    accept='application/json',
    contentType='application/json'
)

response_body = json.loads(response.get('body').read().decode())
print(response_body)