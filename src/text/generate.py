import boto3
import json
import pprint

client = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1')

model_id = 'amazon.titan-text-express-v1'
prompt = "tell me a storey about a dragon"

titan_config = json.dumps({
  "inputText": prompt,
  "textGenerationConfig": {
    "maxTokenCount": 3072,
    "stopSequences": [],
    "temperature": 1.0,
    "topP": 1.0
  }
})


llama_model_id = 'meta.llama3-8b-instruct-v1:0'
llama_config = json.dumps({
  "prompt": prompt,
  "max_gen_len": 512,
  "temperature": 0,
  "top_p": 0.9,
})

try: 
    response = client.invoke_model(
        modelId=llama_model_id,
        body=llama_config,
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response['body'].read()) 
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(response_body.get('results'))
    pp.pprint(response_body.get('generation'))
except Exception as e:
    print(f"Error invoking model: {e}")


