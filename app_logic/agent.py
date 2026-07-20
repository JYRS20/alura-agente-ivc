from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_agent(vectorstore):
    """
    Configura el agente conectando el LLM con la base vectorial.
    Se implementa LCEL con historial de chat y recuperación semántica con umbral de relevancia.
    """
    print("Inicializando el pipeline RAG...")

    # Instanciar el LLM
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

    # Configurar el retriever descartando resultados de baja relevancia
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.35,
            "k": 4
        }
    )

    system_prompt = """
Eres el asistente virtual oficial de Istmo Velocity Circuit (IVC).

Responde preguntas sobre servicios, tarifas, inscripciones, pilotos,
vehículos, categorías, seguridad, reglamentos, reservas y políticas,
utilizando únicamente el contexto proporcionado.

No describas a la empresa como ficticia ni menciones proyectos,
desafíos, cursos o procesos de desarrollo, salvo que esa información
aparezca expresamente en el contexto vigente.

Si la respuesta no está disponible en los documentos, indica:
"No encuentro esa información en la documentación disponible".

Responde de manera completa pero concisa. Utiliza viñetas cuando
existan varios requisitos, pasos o condiciones.

Contexto:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Formateo de documentos para inyectar referencias precisas
    def format_docs(docs):
        if not docs:
            return "No se encontró información relevante en los documentos."
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Documento")
            source = source.replace("\\", "/").split("/")[-1]
            page = doc.metadata.get("page")
            
            reference = f"Fuente: {source}"
            if page is not None:
                reference += f", página {page + 1}"
                
            formatted.append(f"{reference}\nContenido:\n{doc.page_content}")
        return "\n\n".join(formatted)

    def create_rag_chain_with_history(chat_history, user_input):
        docs = retriever.invoke(user_input)
        context = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "input": user_input
        })

    class AgentWrapper:
        def invoke(self, inputs):
            chat_history = inputs.get("chat_history", [])
            answer = create_rag_chain_with_history(chat_history, inputs["input"])
            return {"answer": answer}

    print("Pipeline RAG inicializado con éxito.")
    return AgentWrapper()
