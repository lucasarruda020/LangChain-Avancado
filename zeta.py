# baixa todos doc do diretorio
from langchain_community.document_loaders import DirectoryLoader

# chamando tokenzier do HF
from transformers import AutoTokenizer

# Chamando o splitter por tokens
from langchain_text_splitters import CharacterTextSplitter

#Banco vetorial local
from langchain_community.vectorstores import FAISS

# conexao local
from langchain_ollama import OllamaEmbeddings

# chamando modelo do ollama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

modelo = OllamaLLM(model="gemma3:4b")

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'responda exclusivamente usando os conteudos fornecidos em \n\n: {contexto}'),
        ('human', '{query}')
    ]
)

pdfs = DirectoryLoader("documentos", glob="*.pdf").load()

tokenzier = AutoTokenizer.from_pretrained("BAAI/bge-m3")

splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenzier,
    chunk_size=1250, chunk_overlap=150
)

pedacos = splitter.split_documents(pdfs)

embeddings = OllamaEmbeddings(model="bge-m3:567m")


vector_store = FAISS.from_documents(
    documents=pedacos,embedding=embeddings
)

retriever = vector_store.as_retriever()

pergunta = "Qual numero devo ligar para cartao roubado?"

trechos = retriever.invoke(pergunta)

contexto = "\n\n".join(trecho.page_content for trecho in trechos)

cadeia = prompt | modelo | StrOutputParser()

llm_generate = cadeia.invoke({"query": pergunta, "contexto": contexto})

print(llm_generate)