import boto3
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings # <--- IMPORTANTE: Usar ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# 1. Configuración del cliente
client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# 2. MODELO DE EMBEDDINGS
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", 
    client=client
)

# 3. MODELO LLM (Usando la interfaz moderna para Nova)
# ChatBedrockConverse formatea automáticamente el JSON con la clave [messages]
llm_model = ChatBedrockConverse(
    model_id="amazon.nova-micro-v1:0", 
    client=client,
    temperature=0
)

# --- Datos y Lógica RAG ---
my_data = [
    "the weather is nice today.",
    "last night's game was exciting.",
    "don likes to eat pizza on weekends.",
    "don likes to eat pasta!"
]

# Crear el vector store
vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

# Recuperar información
question = "what does don like to eat on weekends?"
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke(question)
context_text = "\n".join([r.page_content for r in results])

# 4. PROMPT (En formato de Chat para que funcione con Nova)
template = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the following context: {context}"),
    ("user", "{input}"),
])

# Crear la cadena
chain = template | llm_model

# Ejecutar
try:
    response = chain.invoke({"input": question, "context": context_text})
    print("Respuesta del modelo:")
    # Importante: Como es un objeto de Chat, el texto está en .content
    print(response.content)
except Exception as e:
    print(f"Error detectado: {e}")