import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

AWS_REGION = "us-east-1"

# LangChain puede crear el cliente internamente, pero dejamos boto3 por claridad
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)

model = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",   # también: amazon.nova-lite-v1:0 / amazon.nova-pro-v1:0
    client=bedrock_runtime
)



def first_chain():
    prompt = ChatPromptTemplate.from_template(
        "Write a short, compelling product description for: {product_name}"
    )

    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"product_name": "bicycle"})
    print(response)

if __name__ == "__main__":
    first_chain()
