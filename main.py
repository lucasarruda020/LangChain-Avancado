from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ.get("CHAVE_API_OPENAI")
os.getenv("LANGSMITH_API_KEY")
os.getenv("LANGCHAIN_TRACING_V2")
api_key = os.getenv("CHAVE_API_OPENAI")

documento = TextLoader("documentos/GTB_gold_Nov23.txt", encoding="utf-8").load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

pedacos = splitter.split_documents(documento)

embeddings_model = OpenAIEmbeddings(api_key=api_key)

print(embeddings_model.model)

embeddings_model.embed_query(pedacos[0].page_content) 

vectorstore = InMemoryVectorStore.from_documents(
    documents=pedacos, embedding=embeddings_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

retriever.invoke("Seguro viagem")

query = "Como eu devo proceder caso tenha um item comprado roubado?"

query_embed = embeddings_model.embed_query(query)

similar_chunks = retriever.invoke(query)

similar_texts = [pedaco.page_content for pedaco in similar_chunks]

prompt = ChatPromptTemplate.from_messages(
   [ ("system", "responda usando exclusivamente os conteudos fornecidos. \n\nContexto:\n{contexto}"),
    ("human", "{query}")
])

modelo = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0.2,
    api_key=api_key
)


modelo.invoke(query)

cadeia = prompt | modelo | StrOutputParser()

trechos = retriever.invoke(query)
contexto = "\n\n".join(trecho.page_content for trecho in trechos)

resposta = cadeia.invoke({"query": query, "contexto": contexto})

print(resposta)