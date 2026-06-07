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
from huggingface_hub import login
load_dotenv()

login(token=os.getenv("HF_TOKEN"))

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

emb_tokenzier = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
emb_model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

