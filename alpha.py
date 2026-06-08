# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

# Permite acessar variáveis de ambiente e recursos do sistema operacional
import os

# Carrega documentos PDF para processamento pelo LangChain
from langchain_community.document_loaders import PyPDFLoader

# Divide textos grandes em pedaços menores (chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gera embeddings utilizando modelos da OpenAI
from langchain_openai import OpenAIEmbeddings

# Armazena embeddings em um banco vetorial na memória
from langchain_community.vectorstores import InMemoryVectorStore

# Cria templates para prompts enviados ao modelo
from langchain_core.prompts import ChatPromptTemplate

# Interface para utilização dos modelos de chat da OpenAI
from langchain_openai import ChatOpenAI

# Converte a saída do modelo para string simples
from langchain_core.output_parsers import StrOutputParser

# Carrega conteúdo de páginas web para processamento
from langchain_community.document_loaders import WebBaseLoader

# Carrega conteudo de um diretorio
from langchain_community.document_loaders import DirectoryLoader

# Juntar todos os codigos 
from langchain_community.document_loaders.merge import MergedDataLoader

# Rodar arquivos tipo texto
from langchain_community.document_loaders import TextLoader

# Baixa em paginas web 
from langchain_community.document_loaders import RecursiveUrlLoader

# Biblioteca para chunk de tokens
from langchain_text_splitters import CharacterTextSplitter

#hugging face
from transformers import AutoTokenizer, AutoModel

# Fazer forma semantica langchain com chunks
from langchain_experimental.text_splitter import SemanticChunker

# forma semantica usando HF
from langchain_huggingface import HuggingFaceEmbeddings

# Banco vetorial de embeddings FAISS (Comum igual chroma)
from langchain_community.vectorstores import FAISS

# Banco vetorial tambem 
from langchain_pinecone import PineconeVectorStore


load_dotenv()
os.environ.get("CHAVE_API_OPENAI")
api_key = os.getenv("CHAVE_API_OPENAI")


modelo = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0.2,
    api_key=api_key
)
# Ler apenas um doc
documento = PyPDFLoader("documentos/GTB_gold_Nov23.pdf").load()

# baixar url
#url = "https://g1.globo.com/tecnologia/noticia/2026/06/05/dona-do-claude-sugere-pausa-no-desenvolvimento-da-ia-por-risco-de-sistemas-sairem-do-controle-humano.ghtml"
#news = WebBaseLoader(web_path=url).load()
# Pegar tudo de um diretorio 
pdfs = DirectoryLoader("/Users/lucasarruda/Study/Alura/LangChain_Tecnicas_avancadas_de_RAG/documentos", glob="*.pdf").load()

# Juntar todos docs em um so
#all_docs = MergedDataLoader(loaders=[WebBaseLoader(web_path=url), TextLoader("/Users/lucasarruda/Study/Alura/LangChain_Tecnicas_avancadas_de_RAG/documentos/GTB_gold_Nov23.txt", encoding='utf-8')])

#all_loaders = all_docs.load()

#print(len(all_loaders))

## Baixar a pagina web
#loader = RecursiveUrlLoader("https://platform.openai.com/usage")

#lc_docs = loader.load()

## Chunk por letra palavra
#splitter = RecursiveCharacterTextSplitter(
#   chunk_size=1000, chunk_overlap=100
#)

#pedacos = splitter.split_documents(pdfs)




token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=8000, chunk_overlap=100
)

token_pedacos = token_splitter.split_documents(pdfs)

# Baixar arquivos HF para token
emb_tokenzier = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
emb_model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

hf_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=emb_tokenzier,
    chunk_size=512, chunk_overlap=50
)

hf_pedacos = hf_splitter.split_documents(pdfs)
len(hf_pedacos)


# pega por blocos do chunks por um setor logico pela OpenAI
semantic_openai_splitter = SemanticChunker(OpenAIEmbeddings())


semantic_openai_pedacos = semantic_openai_splitter.split_documents(pdfs)


print(len(semantic_openai_pedacos[2]))

#mUsarnsetor logico para os chunks porem com HF
hf_emb_model = HuggingFaceEmbeddings(model_name="infloat/multilingual-e5-small")
semantic_hf_splitter = SemanticChunker(emb_model)
semantic_hf_pedacos = semantic_hf_splitter.split_documents(pdfs)

len(semantic_hf_pedacos)



#from InMemoryVectorStore

vectorStore = InMemoryVectorStore.from_documents(
    documens=semantic_hf_pedacos, embedding=hf_emb_model
)

retriever = vectorStore.as_retriever(search_kwargs={"k":3})
print(retriever.invoke("Seguro viagem"))


# Utilizando FAISS & Demais ectore store


vectorstore = FAISS.from_documents(
    documents=semantic_hf_pedacos, embedding=hf_emb_model
)

retriever = vectorstore.as_retriever()

print(retriever.invoke("Sala VIP"))

pc_vector_store = PineconeVectorStore(
    host="https/www.example.com",
    pineconne_api_key="spk-3132190120312092",
    embedding=hf_emb_model
)

pc_vector_store.add_documents(semantic_hf_pedacos)
pc_vector_store.similarity_search_with_score("Seguro viagem") # sem usa retriever fazendo uma busca similar direto e dando resultado da distancia vetorial.
