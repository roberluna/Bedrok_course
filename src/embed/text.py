import boto3
import json
import numpy as np
from scipy.spatial.distance import cosine

# Instead of importing, we define it here:
def cosine_similarity(v1, v2):
    # scipy calculates distance, so similarity is 1 - distance
    return 1 - cosine(v1, v2)

facts = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "The Amazon River is the largest river by discharge volume of water in the world.",
    "Mount Everest is the highest mountain above sea level.",
    "The Sahara Desert is the largest hot desert in the world."
]

new_fact = "what is the biggest river in mexico?"

client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

def get_embedding(input_text: str):
    response = client.invoke_model(
        body=json.dumps({
            "inputText": input_text,
        }),
        modelId='amazon.titan-embed-text-v1',
        accept='application/json',
        contentType='application/json'
    )

    response_body = json.loads(response.get('body').read().decode())
    return response_body['embedding']

# 1. Get embeddings for existing facts
fact_with_embeddings = []
for fact in facts:
    fact_with_embeddings.append({
        'text': fact,
        'embedding': get_embedding(fact)            
    })

# 2. Get embedding for the new fact
new_fact_embedding = get_embedding(new_fact)

# 3. Calculate similarities
similarities = []
for item in fact_with_embeddings:
    similarities.append({
        'text': item['text'],
        # Fixed the variable name here from fact['embedding'] to item['embedding']
        'similarity': cosine_similarity(item['embedding'], new_fact_embedding)
    })

print(f"Facts most similar to: '{new_fact}'\n")
similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

for sim in similarities:
    print(f"Similarity: {sim['similarity']:.4f} | Fact: {sim['text']}")