from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import os

FAISS_INDEX_PATH = "faiss_index"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def process_documents_to_vectorstore(documents):
    """
    Fragmenta los documentos y genera los embeddings correspondientes 
    almacenándolos en FAISS. Se implementa persistencia en disco y 
    procesamiento por lotes para respetar los límites de la API.
    """
    embeddings = get_embeddings()

    # Cargar índice desde disco si está disponible
    if os.path.exists(FAISS_INDEX_PATH):
        print("📂 Cargando índice FAISS existente desde disco...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("✅ Índice FAISS cargado exitosamente.")
        return vectorstore

    print("Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Se generaron {len(splits)} fragmentos de texto.")

    # Procesamiento por lotes para evitar error 429 (Rate Limit) de la API gratuita
    print("Generando embeddings en lotes (puede tardar unos minutos)...")
    BATCH_SIZE = 80
    vectorstore = None

    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Procesando lote {batch_num}/{total_batches} ({len(batch)} fragmentos)...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)

        if i + BATCH_SIZE < len(splits):
            print(f"  ⏳ Esperando 65s para respetar el límite de la API gratuita...")
            time.sleep(65)

    print("💾 Guardando índice FAISS en disco para futuros usos...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ ¡Base de datos vectorial creada y guardada exitosamente!")

    return vectorstore
