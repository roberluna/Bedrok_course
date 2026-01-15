import boto3
import json

AWS_REGION = "us-east-1"

client = boto3.client(service_name="bedrock-agent-runtime", region_name=AWS_REGION)

def handler(event, context):
    body = json.loads(event['body'])
    question = body.get('question', '')
    
    if question:
        response = client.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "4KJO0F3DJ5",
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0",
                    # Added the generationConfiguration below
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": "You are a helpful assistant. Use the following search results to answer the user's question: $search_results$. Question: $query$"
                        }
                    }
                },
            }
        )
        answer = response.get("output").get("text")
        return {
            'statusCode': 200,
            'body': json.dumps({'answer': answer})
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Question not provided'})
        }