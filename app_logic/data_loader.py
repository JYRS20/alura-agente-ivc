import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

def load_documents(directory_path="docs"):
    """
    Lee todos los archivos PDF y CSV en la carpeta docs/ y los convierte 
    en documentos procesables para LangChain.
    """
    documents = []
    
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        if filename.endswith(".pdf"):
            print(f"Cargando PDF: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            
        elif filename.endswith(".csv"):
            print(f"Cargando CSV: {filename}")
            loader = CSVLoader(file_path=filepath, encoding='utf-8')
            documents.extend(loader.load())
            
    print(f"Total de fragmentos cargados: {len(documents)}")
    return documents
