import boto3
import pprint

bedrock = boto3.client(
    service_name='bedrock', 
    region_name='us-east-2')
pp = pprint.PrettyPrinter(indent=4)

models = bedrock.list_foundation_models()
for model in models['modelSummaries']:
    pp.pprint(model)



