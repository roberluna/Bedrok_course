import boto3
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. CONFIGURACIÓN INICIAL
# Usa la región donde tengas activos tus modelos (us-east-1 o us-west-2)
AWS_REGION = "us-east-1" 
client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

# 2. PREPARAR LOS MODELOS
# El "cerebro" que lee los fragmentos de PDF y genera la respuesta
llm_model = ChatBedrockConverse(
    model_id="amazon.nova-micro-v1:0", 
    client=client,
    temperature=0
)

# El "traductor" que convierte texto en números (vectores)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", 
    client=client
)

# 3. CARGA Y PROCESAMIENTO DEL PDF
# Cargamos el archivo
loader = PyPDFLoader("assets/books.pdf")
docs = loader.load()

# Picamos el PDF en trozos pequeños (chunks). 
# Imagina que el PDF es un filete largo; el LLM solo puede comer bocados pequeños.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # Tamaño del trozo
    chunk_overlap=100, # Un poquito del trozo anterior para no perder el hilo
    separators=["\n\n", "\n", " ", ""]
)
splitted_docs = splitter.split_documents(docs)

# 4. CREACIÓN DE LA BASE DE DATOS (VECTOR STORE)
# Aquí guardamos los trozos de PDF convertidos en números
vector_store = FAISS.from_documents(splitted_docs, embeddings)

# 5. BUSCADOR (RETRIEVER)
# Cuando hagas una pregunta, buscará los 3 trozos más parecidos en el PDF
question = "Givme a brief summary of the book in spanish. only one paragraph and 30 words"
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke(question)

# Juntamos los trozos encontrados en un solo texto
context_text = "\n\n".join([r.page_content for r in results])

# 6. EL PROMPT Y LA CADENA (CHAIN)
# Le damos instrucciones claras al modelo
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question using ONLY the following context: {context}"),
    ("user", "{input}"),
])

# Unimos el prompt con el modelo
chain = template | llm_model

# 7. EJECUCIÓN
try:
    response = chain.invoke({"input": question, "context": context_text})
    print("\n--- RESPUESTA DEL PDF ---")
    # Usamos .content porque ChatBedrockConverse devuelve un objeto mensaje
    print(response.content)
except Exception as e:
    print(f"Ups! Algo salió mal: {e}")