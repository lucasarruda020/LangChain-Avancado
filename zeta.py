import os
import dotenv
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
# chamand openAI
from langchain_openai import ChatOpenAI
# nova biblioteca para fazer multiQuerys
from langchain.retrievers.multi_query import MultiQueryRetriever

# preparar o datase para treinamento

from langchain_community.evaluation.qa_evaluator import QAEvalChain
from langchain.evaluation.qa import QAGenerateChain

import json

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

## Generating Multiple queries para contextualizar a resposta

multi_query_prompt_template = """
Voce e um assistente de modelo de linguagem de IA. Sua tarefa e gerar versoes diferentes da pergunta do usuario para recuperar documentos relevantes de um banco de dados vetorial
ao gerar multiplas perspectivas sobre a pergunta do usuario
Pergunta original: {question}
"""

multi_query_prompt = PromptTemplate.from_template(multi_query_prompt_template)

multi_query_chain = multi_query_prompt | modelo | CommaSeparatedListOutputParser()

multi_query_chain.invoke(pergunta)



multi_retriever = MultiQueryRetriever(
    retriever=retriever, llm_chain=multi_query_chain
)

multi_rag_chain = (
    {
        "contexto": RunnablePassthrough() | multi_retriever,
        "query": RunnablePassthrough()
    } | prompt | modelo | StrOutputParser()
)

multi_rag_chain.invoke(pergunta)


## Tecnica hyde

hyde_prompt_template = """
escreva um paragrafo que possa responder a pergunta apresentada, nao adicione informacoes
Pergunta: {question}
Paragrafo:
"""

hyde_prompt = PromptTemplate.from_template(hyde_prompt_template)

hyde_chain = modelo | StrOutputParser(

)

hyde_rag_chain = (
        {
        "contexto": RunnablePassthrough() | hyde_chain | retriever,
        "query": RunnablePassthrough()
    }
)

multi_query_chain.invoke(pergunta)


## Aula de validacao cap 4 Criando um file que de acordo com o chunk ele gera uma pergunta + uma resposta
eval_chain = QAEvalChain.from_llm(modelo)

def avaliar(perguntas_respostas, geracoes):
    avaliacoes = eval_chain(perguntas_respostas, geracoes)
    corretas = 0
    for i in enumerate(perguntas_respostas):
        print(avaliacoes[i])
        corretas = corretas + (1 if avaliacoes[i]["results"].split("\n")[-1].split(":")[-1].strip() == "CORRECT" else 0)
    return corretas/len(perguntas_respostas)


qa_chain = QAGenerateChain.from_llm(modelo)

perguntas_resposta = qa_chain.apply_and_parse(
    [{"doc": p.page_content} for p in pedacos]
)

with open("qa_pais.json") as file:
    pairs = json.load(file)

len(pairs)

geracoes_sem_rag = []
for pr in perguntas_resposta[:10]:
    geracoes_sem_rag.append({"result": modelo.invoke(pr["query"])})


geracoes_sem_rag2 = [{"result": g["result"].content} for g in geracoes_sem_rag]

avaliar(perguntas_resposta[:10], geracoes_sem_rag2)


geracoes_multi_rag = []

for pr in perguntas_resposta[:10]:
    geracoes_sem_rag.append({"result": modelo.invoke(pr["query"])})

avaliar(perguntas_resposta[:10], geracoes_multi_rag)