import streamlit as st
import os
from pathlib import Path
from agente import carregar_agente, responder_agente

# ----------------------------
# Configuração da página
# ----------------------------
st.set_page_config(page_title="RoBECC Agente", page_icon="🤖")

st.title("🤖 RoBECC Agente")
st.subheader("Assistente Especialista em Six Sigma")
st.write("Digite sua pergunta abaixo e eu responderei com base nos documentos fornecidos.")

# ----------------------------
# Aviso Legal na sidebar
# ----------------------------
st.sidebar.markdown(
    """
    ---
    📌 **Aviso Legal**  
    Este agente usa como base o material  
    *“Six Sigma” (Graeme Knowles, Bookboon, 2011)*.  
    Conteúdo de caráter **educacional e explicativo**,  
    sem finalidade comercial.  
    """
)

# ----------------------------
# Inicializa o agente (uma vez) e gerencia o estado da sessão
# ----------------------------
@st.cache_resource
def get_agente():
    try:
        if not os.path.exists("DADOS") or not any(Path("DADOS").glob("*.pdf")):
            st.error("❌ A pasta 'DADOS' não foi encontrada ou está vazia. Por favor, adicione os arquivos PDF.")
            st.stop()
        
        agente_instance = carregar_agente("DADOS")
        return agente_instance
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar o agente: {e}")
        st.stop()

# Carrega o agente e exibe um status
agente_instance = get_agente()
if agente_instance:
    st.success("✅ RoBECC pronta para o uso!")

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Processa a nova pergunta do usuário
# ----------------------------
if pergunta := st.chat_input("Sua pergunta:"):
    
    # Adiciona a pergunta ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Gera a resposta com o agente
    with st.chat_message("assistant"):
        with st.spinner("Buscando a resposta nas políticas..."):
            try:
                resposta = responder_agente(agente_instance, pergunta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
                st.markdown(resposta)
            except Exception as e:
                st.error(f"❌ Erro ao processar a pergunta: {e}")
