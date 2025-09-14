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
    Roteia perguntas:
    - Se for sobre nível sigma -> usa cálculo direto.
    - Caso contrário -> injeta histórico no prompt.
    - Se a resposta vier vazia ou "não encontrei", complementa com conhecimento do LLM.
    """

    # 1. Detecta perguntas de nível sigma
    padrao = r"(?i)(nível sigma|calcular sigma|qual.*sigma)"
    if re.search(padrao, pergunta):
        try:
            matches = re.findall(r"\d+[.,]?\d*", pergunta)
            numeros = [float(m.replace(".", "").replace(",", ".")) for m in matches]

            unidades = oportunidades = defeitos = None

            # Busca por palavras-chave
            m_unid = re.search(r"(\d+[.,]?\d*)\s*unidade", pergunta, re.IGNORECASE)
            if m_unid:
                unidades = float(m_unid.group(1).replace(".", "").replace(",", "."))

            m_opp = re.search(r"(\d+[.,]?\d*)\s*oportunidade", pergunta, re.IGNORECASE)
            if m_opp:
                oportunidades = float(m_opp.group(1).replace(".", "").replace(",", "."))

            m_def = re.search(r"(\d+[.,]?\d*)\s*defeito", pergunta, re.IGNORECASE)
            if m_def:
                defeitos = float(m_def.group(1).replace(".", "").replace(",", "."))

            if unidades and oportunidades and defeitos is not None:
                sigma = calcular_sigma(int(unidades), int(oportunidades), int(defeitos))
                return f"O nível sigma do processo é aproximadamente **{sigma}**."
            else:
                return "Para calcular o nível sigma, preciso de três informações: número de unidades produzidas, oportunidades por unidade e defeitos encontrados."

        except Exception as e:
            return f"⚠️ Não consegui calcular o sigma: {e}"

    # 2. Injeta histórico no prompt
    historico = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]]
    )

    resposta = agente.invoke({
        "input": f"Histórico da conversa:\n{historico}\n\nNova pergunta: {pergunta}"
    })["answer"]

    # 3. Se não houver resposta da apostila, usa fallback no modelo
    if "não encontrei" in resposta.lower() or resposta.strip() == "":
        return (
            "⚠️ Não encontrei essa resposta na apostila. "
            "Aqui vai uma explicação com base em conhecimento geral de Six Sigma:\n\n"
            f"{agente.llm.invoke(pergunta).content}"
        )

    return resposta
