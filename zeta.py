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
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
# Chain mais complexas
from langchain_core.runnables import RunnablePassthrough
# Prompt de template comum 
from langchain_core.prompts import PromptTemplate

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

# cadeia completa do langchain! 
rag_chain = (
    {
        "contexto": RunnablePassthrough() | retriever,
        "query": RunnablePassthrough()
    } | prompt | modelo | StrOutputParser()
)

rag_chain.invoke(pergunta)

# Rewrite-Retrieve-Read

query_model = OllamaLLM(model="gemma3:1b")

rewrite_Retrieve_Read = """
Gere consulta de pesquisa para o banco de dados de vetores (Vector DB) a partir de ua pergunta do usuario

pergunta do usuario: {user_question}
consulta revisada do Vector DB:
"""

rewriter_prompt = PromptTemplate.from_template(rewrite_Retrieve_Read)

rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

rewriter_chain.invoke(pergunta)


rewritter_rag_chain = (
    {
        "contexto": RunnablePassthrough() | rewriter_chain | retriever,
        "query": RunnablePassthrough()
    } | prompt | modelo | StrOutputParser()
)

rewritter_rag_chain.invoke(pergunta)

## Generating Multiple queries

multi_query_prompt_template = """
Voce e um assistente de modelo de linguagem de IA. Sua tarefa e gerar versoes diferentes da pergunta do usuario para recuperar documentos relevantes de um banco de dados vetorial
ao gerar multiplas perspectivas sobre a pergunta do usuario
Pergunta original: {question}

"""

multi_query_prompt = PromptTemplate.from_template(multi_query_prompt_template)



multi_query_chain = multi_query_prompt | llm || CommaSeparatedListOutputParser()