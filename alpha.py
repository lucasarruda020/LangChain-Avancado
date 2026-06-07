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

load_dotenv()
os.environ.get("CHAVE_API_OPENAI")
os.getenv("LANGSMITH_API_KEY")
os.getenv("LANGCHAIN_TRACING_V2")
api_key = os.getenv("CHAVE_API_OPENAI")

modelo = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0.2,
    api_key=api_key
)

documento = PyPDFLoader("documentos/GTB_gold_Nov23.pdf").load()

url = "https://g1.globo.com/tecnologia/noticia/2026/06/05/dona-do-claude-sugere-pausa-no-desenvolvimento-da-ia-por-risco-de-sistemas-sairem-do-controle-humano.ghtml"

news = WebBaseLoader(web_path=url).load()

pdfs = DirectoryLoader("/Users/lucasarruda/Study/Alura/LangChain_Tecnicas_avancadas_de_RAG/documentos", glob="*.pdf").load()