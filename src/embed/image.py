import boto3
import json
import base64


import numpy as np
from scipy.spatial.distance import cosine

# Instead of importing, we define it here:
def cosine_similarity(v1, v2):
    # scipy calculates distance, so similarity is 1 - distance
    return 1 - cosine(v1, v2)

client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

#output/inpainted_image.png
images = [
    "./output/inpainted_image.png",
    "./output/stability_image_1.png",
    "./output/stability_image_2.png"
]

def getImageEmbedding(image_path: str):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.invoke_model(
        body=json.dumps({
                "inputImage": encoded_string,
            }
        ),
        modelId='amazon.titan-embed-image-v1',
        accept='application/json',
        contentType='application/json'
    )

    response_body = json.loads(response.get('body').read().decode())
    return response_body['embedding']

# 1. Get embeddings for existing images
image_with_embeddings = []
for image_path in images:
    image_with_embeddings.append({
        'path': image_path,
        'embedding': getImageEmbedding(image_path)            
    })

test_image = "./output/stability_image_2.png"

test_image_embedding = getImageEmbedding(test_image)

similarities = []
for item in image_with_embeddings:
    similarities.append({
        'path': item['path'],
        'similarity': cosine_similarity(item['embedding'], test_image_embedding)
    })

similarities.sort(key=lambda x: x['similarity'], reverse=True)

print(f"Images most similar to: '{test_image}'\n")
for sim in similarities:
    print(f"Similarity: {sim['similarity']:.4f} | Image Path: {sim['path']}")


