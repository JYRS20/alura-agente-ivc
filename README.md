# Asistente Virtual de Istmo Velocity Circuit

Agente de inteligencia artificial que responde preguntas en lenguaje natural sobre los documentos internos de Istmo Velocity Circuit (IVC): manuales, reglamentos, políticas, tarifas y datos de pilotos y vehículos.

El proyecto fue desarrollado como desafio final del programa Alura Agente, y utiliza la tecnica RAG (Retrieval-Augmented Generation) para recuperar informacion relevante de los documentos antes de generar cada respuesta.

---

## Arquitectura de la solucion

1. Carga de documentos: los archivos PDF y CSV ubicados en la carpeta docs/ se leen con PyPDFLoader y CSVLoader de LangChain.
2. Fragmentacion: el texto se divide en fragmentos de 1000 caracteres con superposicion de 200, usando RecursiveCharacterTextSplitter.
3. Embeddings y base vectorial: cada fragmento se convierte en un vector numerico usando el modelo gemini-embedding-001 de Google. Los vectores se almacenan en FAISS y se guardan en disco para evitar reprocesar los documentos en cada arranque.
4. Recuperacion con umbral: cuando el usuario hace una pregunta, el sistema busca los 4 fragmentos mas relevantes con un umbral de similitud de 0.35. Si no se supera el umbral, el agente indica que no encontro informacion en los documentos.
5. Generacion de respuesta: los fragmentos recuperados se envian junto con la pregunta al modelo gemini-flash-latest de Google, que produce una respuesta en lenguaje natural. Cada respuesta incluye la fuente del documento y el numero de pagina.
6. Memoria conversacional: el historial de la conversacion se pasa al modelo en cada consulta, lo que permite referencias contextuales entre preguntas.
7. Interfaz: la aplicacion se sirve con Streamlit, que presenta un chat interactivo accesible desde el navegador.

---

## Tecnologias utilizadas

- Python 3
- LangChain (langchain, langchain-community, langchain-text-splitters)
- Google Gemini (langchain-google-genai, google-generativeai)
- FAISS (faiss-cpu)
- Streamlit
- pypdf
- pandas
- python-dotenv

---

## Instrucciones para ejecutar el proyecto

### Requisitos previos

- Python 3.10 o superior instalado
- Una clave de API de Google Gemini (se obtiene en https://aistudio.google.com)

### Pasos

1. Clonar el repositorio:

   git clone <url-del-repositorio>
   cd Proyecto-Oracle

2. Crear y activar el entorno virtual:

   En Windows:
   python -m venv venv
   .\venv\Scripts\activate

   En Mac/Linux:
   python -m venv venv
   source venv/bin/activate

3. Instalar dependencias:

   pip install -r requirements.txt

4. Configurar la clave de API:

   Crear un archivo .env en la raiz del proyecto con el siguiente contenido:
   GEMINI_API_KEY=tu_clave_aqui

5. Ejecutar la aplicacion:

   streamlit run app.py

   La primera vez procesara todos los documentos y guardara el indice en disco (puede tardar varios minutos segun el limite de cuota de la API gratuita). Las ejecuciones siguientes seran instantaneas.

---

## Ejemplos de preguntas y respuestas

Pregunta: Cuales son los requisitos de seguridad para participar en una fecha de competencia?

Respuesta del agente:
Para participar en una fecha de competencia en IVC, los vehiculos deben cumplir con los siguientes requisitos de seguridad:
- Jaula antivuelco certificada segun normativa FIA o equivalente
- Extintores vigentes dentro del habitaculo
- Cinturon de seguridad de cinco puntos en buen estado
- Casco certificado con homologacion vigente
- Equipo ignifugo para el piloto (traje, guantes y balaclava)
Fuente: 02_Reglamento_deportivo_tecnico_seguridad_IVC.pdf, pagina 8.

---

Pregunta: Cuanto tiempo antes debo cancelar una reserva para obtener reembolso completo?

Respuesta del agente:
Segun la politica de cancelaciones, se otorga reembolso completo si la cancelacion se notifica con al menos 7 dias habiles de anticipacion. Cancelaciones realizadas con menos de 7 dias generan una penalidad del 15% por gastos administrativos. No se realizan reembolsos por cancelaciones el mismo dia del evento.
Fuente: 04_Politica_reservas_pagos_cancelaciones_reembolsos_IVC.pdf, pagina 3.

---

Pregunta: Que categorias de vehiculos estan registradas en el circuito?

Respuesta del agente:
De acuerdo con el registro de pilotos y vehiculos, las categorias activas en IVC son: Turismo Nacional, GT3, Prototipo Libre y Karting Senior. Cada categoria tiene requisitos tecnicos especificos de cilindrada, peso minimo y configuracion de seguridad.
Fuente: registro_pilotos_vehiculos_categorias_IVC.csv.

