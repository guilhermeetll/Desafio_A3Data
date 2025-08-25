import streamlit as st
import requests
import os

# URL do seu backend FastAPI - usando variável de ambiente
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/chat")

# Inicializa o histórico do chat na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["author"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if prompt := st.chat_input("Como posso te ajudar?"):
    # Adiciona a mensagem do usuário ao histórico e exibe
    st.session_state.messages.append({"author": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara a mensagem para enviar ao backend
    payload = {
        "message": prompt,
        "conversation_history": st.session_state.messages[:-1]  # Envia o histórico anterior
    }

    # Exibe uma mensagem de "pensando..." enquanto espera a resposta
    with st.chat_message("assistant"):
        with st.spinner("Consultando os relatórios epidemiológicos..."):
            try:
                # Envia a requisição para o backend
                response = requests.post(BACKEND_URL, json=payload)
                response.raise_for_status()  # Lança um erro para respostas ruins (4xx ou 5xx)

                # Extrai a resposta do backend
                bot_response = response.json()
                response_content = bot_response.get("content", "Desculpe, ocorreu um erro.")
                sources = bot_response.get("sources", [])

                # Exibe e armazena a resposta do assistente
                st.markdown(response_content)
                
                # Exibe as fontes utilizadas (opcional)
                if sources:
                    with st.expander("Fontes consultadas"):
                        for i, source in enumerate(sources[:2]):  # Mostra até 2 fontes
                            st.markdown(f"**Fonte {i+1}**: {source.get('source', 'Documento')}")

                st.session_state.messages.append({"author": "assistant", "content": response_content})

            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao se conectar com o backend: {e}")
                st.error("Detalhes técnicos: " + str(e.response.text if e.response else ""))