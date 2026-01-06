import boto3
import json
import base64
import random
import os

client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
modelId='amazon.titan-image-generator-v2:0'
prompt =  "A photo of a dragon, please"
seed = random.randint(0,2147483647)


native_request = {
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
       "text": prompt,
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "quality": "standard",
        "cfgScale": 8.0,
        "height": 512,
        "width": 512,
        "seed": seed,
    } 
}

request = json.dumps(native_request)

response = client.invoke_model(
    modelId=modelId,
    body=request,
)

model_response = json.loads(response['body'].read())

base64_image_data = model_response['images'][0]

i, output_dir = 1, 'output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
while os.path.exists(f'{output_dir}/stability_image_{i}.png'):
    i += 1

image_data = base64.b64decode(base64_image_data)

image_path = os.path.join(output_dir, f'stability_image_{i}.png')
with open(image_path, 'wb') as image_file:
    image_file.write(image_data)

print(f'Image saved to {image_path}')