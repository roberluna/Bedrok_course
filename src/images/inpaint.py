import boto3
import json
import base64
import random

client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
seed = random.randint(0,2147483647)

def get_configuration(inputImage: str):
    return json.dumps({
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "image": inputImage,
            "text": "Make the dragon red and fierce",
            "negativeText": "bad quality, low res",
            "maskPrompt": "dragon",
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 7.0,
            "seed": seed,
        }
    })

with open('output/stability_image_1.png', 'rb') as image_file:
    base_image = base64.b64encode(image_file.read()).decode('utf-8')

response = client.invoke_model(
    modelId='amazon.titan-image-generator-v2:0',
    body=get_configuration(base_image),
    accept='application/json',
    contentType='application/json',
)

response_body = json.loads(response['body'].read())
base64_image_data = response_body['images'][0]

base64_image_data = base64_image_data.encode('utf-8')
image_data = base64.b64decode(base64_image_data)

with open('output/inpainted_image.png', 'wb') as image_file:
    image_file.write(image_data)

print('Inpainted image saved to output/inpainted_image.png')
