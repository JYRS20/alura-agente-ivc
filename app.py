import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from app_logic.data_loader import load_documents
from app_logic.vector_store import process_documents_to_vectorstore
from app_logic.agent import setup_agent

# Configuración principal de la interfaz
st.set_page_config(
    page_title="Asistente IVC",
    layout="centered"
)

st.title("🏁 Asistente Virtual de Istmo Velocity Circuit")
st.write(
    "Consulta información sobre servicios, tarifas, inscripciones, "
    "categorías, vehículos, seguridad y políticas del autódromo."
)

@st.cache_resource
def initialize_agent():
    # Inicialización del pipeline RAG
    docs = load_documents()
    vectorstore = process_documents_to_vectorstore(docs)
    agent_chain = setup_agent(vectorstore)
    return agent_chain

# Validación de credenciales
if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "tu_clave_api_aqui":
    st.error("⚠️ Falta configurar tu GEMINI_API_KEY en el archivo .env")
    st.stop()

with st.spinner("Cargando documentos y preparando el asistente..."):
    agent = initialize_agent()

# Gestión del estado de la sesión para el historial
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("¿Qué deseas saber sobre el autódromo?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Convertir el historial al formato esperado por LangChain
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # Invocar al agente pasando el input y el historial de la conversación
            response = agent.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            answer = response["answer"]
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Ocurrió un error al consultar a la IA: {e}")
