# -*- coding: utf-8 -*-
"""
agente.py – Agente com 2 ramificações (responde ou pede mais detalhes)
Versão estável usando OpenAI + FAISS com a API moderna do LangChain (create_stuff_documents_chain + create_retrieval_chain).
Chaves e variáveis padronizadas para evitar erros de Missing input keys.
"""

import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from scipy.stats import norm
import re
from utils import calcular_sigma

# Garante que o event loop exista no Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


def carregar_documentos(folder_path: str = "DADOS"):
    """Carrega todos os PDFs da pasta fornecida e retorna lista de documentos."""
    docs = []
    for n in Path(folder_path).glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            print(f"Carregado com sucesso arquivo {n.name}")
        except Exception as e:
            print(f"Erro ao carregar arquivo {n.name}: {e}")
    print(f"Total de documentos carregados: {len(docs)}")
    return docs


def carregar_agente(folder_path: str = "DADOS"):
    """
    Inicializa o agente baseado nos documentos PDF da pasta especificada.
    Pipeline:
      PDFs -> chunks -> embeddings (OpenAI) -> FAISS -> retriever -> chain (stuff)
    Entradas/Saídas do chain:
      input: pergunta do usuário
      context: documentos recuperados (injetados no prompt)
      answer: resposta final
    """
    docs = carregar_documentos(folder_path)
    if not docs:
        raise ValueError("Nenhum documento PDF encontrado na pasta DADOS/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Prompt padronizado: usa 'context' para docs e 'input' para a pergunta
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um agente especialista em Six Sigma. \n"
         "Suas respostas devem seguir as regras abaixo:\n\n"
         "1. Responda APENAS com base no conteúdo fornecido na apostila que foi carregada no sistema.\n"
         "   - Se a pergunta não tiver resposta na apostila, diga claramente:\n"
         "     'Não encontrei essa resposta na apostila.'\n\n"
         "2. A resposta deve ser dada no mesmo idioma em que a pergunta foi feita.\n"
         "   - Se o usuário perguntar em português, responda em português.\n"
         "   - Se o usuário perguntar em inglês, responda em inglês.\n"
         "   - Nunca traduza literalmente; escreva de forma natural e clara no idioma do usuário.\n\n"
         "3. Seja objetivo e direto.\n"
         "   - Evite explicações excessivas ou floreios.\n"
         "   - Prefira respostas concisas, que tragam apenas o essencial.\n\n"
         "4. Caso a pergunta envolva:\n"
         "   - Identificação do nível Belt,\n"
         "   - Como iniciar o mapeamento de um processo,\n"
         "   - Benefícios em tratar um processo,\n"
         "   - Como calcular ganhos financeiros, erros ou ROI,\n"
         "   use as instruções e fórmulas práticas da apostila para responder.\n\n"
         "Contexto:  {context}"),
        ("human", "{input}")
    ])

    # Cria a corrente de 'stuff' que junta docs + prompt
    docs_chain = create_stuff_documents_chain(llm, prompt)

    # Cria o chain de recuperação (retriever -> docs_chain)
    chain = create_retrieval_chain(retriever, docs_chain)

    return chain


def responder_agente(agente, pergunta: str) -> str:
    """
    Roteia perguntas: se for sobre nível sigma, usa cálculo direto.
    Caso contrário, usa o agente normal (RAG).
    """
    # Detecta padrão de pergunta sobre sigma
    padrao = r"(?i)(nível sigma|calcular sigma|qual.*sigma)"
    if re.search(padrao, pergunta):
        try:
            # Procura números na pergunta
            numeros = [int(n) for n in re.findall(r"\d+", pergunta)]
            if len(numeros) >= 3:
                unidades, oportunidades, defeitos = numeros[0], numeros[1], numeros[2]
                sigma = calcular_sigma(unidades, oportunidades, defeitos)
                return f"O nível sigma do processo é aproximadamente **{sigma}**."
            else:
                return "Para calcular o nível sigma, preciso de três informações: número de unidades, oportunidades por unidade e defeitos."
        except Exception as e:
            return f"Não consegui calcular o sigma: {e}"
    
    # Se não for cálculo, cai no fluxo normal
    return agente.invoke({"input": pergunta})["answer"]
