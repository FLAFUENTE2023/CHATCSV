"""
Chatbot CSV con LangChain y OpenAI GPT-4o-mini
==============================================

Aplicación completa de Streamlit que permite cargar archivos CSV y consultar
su contenido a través de un chatbot conversacional usando:
- LangChain para el procesamiento de documentos y cadenas conversacionales
- ChromaDB para almacenamiento vectorial persistente
- OpenAI GPT-4o-mini para generación de respuestas
- OpenAI text-embedding-3-large para embeddings semánticos

Arquitectura modular diseñada para facilidad de mantenimiento y extensión.
"""

import streamlit as st
import pandas as pd
import os
import toml
from pathlib import Path
from typing import List, Optional, Dict, Any

# Importaciones de LangChain para procesamiento de documentos
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importaciones para embeddings y modelos de lenguaje
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Importaciones para almacenamiento vectorial
from langchain_chroma import Chroma

# Importaciones para cadenas conversacionales
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Importaciones para manejo de variables de entorno
from dotenv import load_dotenv


def configurar_credenciales_openai() -> str:
    """
    Configura y valida las credenciales de OpenAI desde múltiples fuentes.
    
    Busca la clave API en el siguiente orden de prioridad:
    1. Variable de entorno OPENAI_API_KEY
    2. Archivo secrets/api_keys.toml
    3. Secretos de Streamlit Cloud (st.secrets)
    
    Returns:
        str: Clave API de OpenAI válida
        
    Raises:
        ValueError: Si no se encuentra ninguna clave API válida
    """
    # Cargar variables de entorno desde archivo .env si existe
    load_dotenv()
    
    # Prioridad 1: Variable de entorno
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Prioridad 2: Archivo local de secretos
    try:
        secrets_path = Path("secrets/api_keys.toml")
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            api_key = secrets.get("openai", {}).get("api_key")
            if api_key and api_key != "tu_clave_openai_aqui":
                return api_key
    except Exception as e:
        st.warning(f"Error al leer archivo de secretos local: {e}")
    
    # Prioridad 3: Secretos de Streamlit Cloud
    try:
        if hasattr(st, "secrets") and "openai" in st.secrets:
            api_key = st.secrets["openai"]["api_key"]
            if api_key:
                return api_key
    except Exception as e:
        st.warning(f"Error al leer secretos de Streamlit Cloud: {e}")
    
    # Si no se encuentra ninguna clave válida
    raise ValueError(
        "No se encontró una clave API de OpenAI válida. "
        "Por favor, configura OPENAI_API_KEY como variable de entorno, "
        "en secrets/api_keys.toml, o en los secretos de Streamlit Cloud."
    )


def inicializar_estado_sesion():
    """
    Inicializa las variables de estado de sesión de Streamlit para mantener
    la persistencia de datos durante la interacción del usuario.
    
    Variables inicializadas:
    - historial_conversacion: Lista de mensajes del chat (usuario/asistente)
    - vectorstore_csv: Almacén vectorial construido desde el CSV
    - cadena_conversacional: Cadena LangChain para consultas
    - datos_csv: DataFrame pandas con los datos cargados
    - archivo_csv_cargado: Booleano indicando si hay un CSV válido cargado
    """
    if "historial_conversacion" not in st.session_state:
        st.session_state.historial_conversacion = []
    
    if "vectorstore_csv" not in st.session_state:
        st.session_state.vectorstore_csv = None
    
    if "cadena_conversacional" not in st.session_state:
        st.session_state.cadena_conversacional = None
    
    if "datos_csv" not in st.session_state:
        st.session_state.datos_csv = None
    
    if "archivo_csv_cargado" not in st.session_state:
        st.session_state.archivo_csv_cargado = False


def validar_archivo_csv(archivo_subido) -> bool:
    """
    Valida que el archivo subido sea un CSV válido y no exceda el límite de tamaño.
    
    Args:
        archivo_subido: Objeto UploadedFile de Streamlit
        
    Returns:
        bool: True si el archivo es válido, False en caso contrario
    """
    if archivo_subido is None:
        return False
    
    # Verificar extensión de archivo
    if not archivo_subido.name.endswith('.csv'):
        st.error("Error: El archivo debe tener extensión .csv")
        return False
    
    # Verificar tamaño del archivo (límite: 5MB)
    tamaño_mb = archivo_subido.size / (1024 * 1024)
    if tamaño_mb > 5:
        st.error(f"Error: El archivo es demasiado grande ({tamaño_mb:.2f}MB). Límite máximo: 5MB")
        return False
    
    return True


def cargar_datos_csv(archivo_subido) -> Optional[pd.DataFrame]:
    """
    Carga un archivo CSV en un DataFrame de pandas con validación de errores.
    
    Args:
        archivo_subido: Objeto UploadedFile de Streamlit
        
    Returns:
        Optional[pd.DataFrame]: DataFrame con los datos o None si hay error
    """
    try:
        # Intentar cargar el CSV con encoding UTF-8
        df = pd.read_csv(archivo_subido, encoding='utf-8')
        
        # Validar que el DataFrame no esté vacío
        if df.empty:
            st.error("Error: El archivo CSV está vacío")
            return None
        
        # Validar que tenga al menos una columna
        if len(df.columns) == 0:
            st.error("Error: El archivo CSV no tiene columnas válidas")
            return None
        
        st.success(f"CSV cargado exitosamente: {len(df)} filas, {len(df.columns)} columnas")
        return df
        
    except UnicodeDecodeError:
        # Intentar con encoding alternativo si UTF-8 falla
        try:
            df = pd.read_csv(archivo_subido, encoding='latin-1')
            st.warning("Archivo cargado con encoding latin-1")
            return df
        except Exception as e:
            st.error(f"Error de codificación: {str(e)}")
            return None
            
    except pd.errors.EmptyDataError:
        st.error("Error: El archivo CSV está vacío o corrupto")
        return None
        
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {str(e)}")
        return None


def convertir_csv_a_documentos(df: pd.DataFrame) -> List[Document]:
    """
    Convierte un DataFrame de pandas en una lista de documentos de LangChain.
    Cada fila del CSV se convierte en un documento independiente.
    
    Args:
        df: DataFrame de pandas con los datos del CSV
        
    Returns:
        List[Document]: Lista de documentos LangChain para procesamiento vectorial
    """
    documentos = []
    
    for indice, fila in df.iterrows():
        # Crear contenido del documento concatenando todos los valores de la fila
        contenido_fila = []
        metadatos = {"fila_numero": indice + 1}
        
        for columna, valor in fila.items():
            # Agregar información de columna-valor al contenido
            if pd.notna(valor):  # Solo incluir valores no nulos
                contenido_fila.append(f"{columna}: {str(valor)}")
                # Guardar información clave en metadatos
                metadatos[columna] = str(valor)
        
        # Unir todo el contenido de la fila en un texto coherente
        contenido_completo = " | ".join(contenido_fila)
        
        # Crear documento LangChain con contenido y metadatos
        documento = Document(
            page_content=contenido_completo,
            metadata=metadatos
        )
        
        documentos.append(documento)
    
    return documentos


def dividir_documentos_grandes(documentos: List[Document]) -> List[Document]:
    """
    Divide documentos largos en fragmentos más pequeños para optimizar
    el procesamiento vectorial y la recuperación de información.
    
    Args:
        documentos: Lista de documentos originales
        
    Returns:
        List[Document]: Lista de documentos divididos en fragmentos óptimos
    """
    # Configurar divisor de texto con parámetros optimizados para datos tabulares
    divisor_texto = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Tamaño máximo de cada fragmento en caracteres
        chunk_overlap=100,      # Solapamiento entre fragmentos para mantener contexto
        separators=[" | ", "\n", " ", ""],  # Separadores priorizados para datos CSV
        length_function=len     # Función para medir longitud del texto
    )
    
    # Dividir todos los documentos
    documentos_divididos = []
    for documento in documentos:
        if len(documento.page_content) > 800:  # Solo dividir documentos largos
            fragmentos = divisor_texto.split_documents([documento])
            documentos_divididos.extend(fragmentos)
        else:
            documentos_divididos.append(documento)
    
    return documentos_divididos


@st.cache_resource
def construir_embeddings_openai(api_key: str) -> OpenAIEmbeddings:
    """
    Construye el modelo de embeddings de OpenAI con configuración optimizada.
    Utiliza cache de Streamlit para evitar reininicializaciones innecesarias.
    
    Args:
        api_key: Clave API de OpenAI
        
    Returns:
        OpenAIEmbeddings: Modelo de embeddings configurado para text-embedding-3-large
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-large",  # Modelo de alta calidad para embeddings semánticos
        openai_api_key=api_key,
        chunk_size=1000,                 # Procesar embeddings en lotes para eficiencia
        max_retries=3                    # Reintentos automáticos en caso de errores de red
    )


def construir_vectorstore_chroma(documentos: List[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Construye o carga un vectorstore persistente usando ChromaDB.
    Permite reutilización de embeddings previamente calculados para eficiencia.
    
    Args:
        documentos: Lista de documentos para vectorizar
        embeddings: Modelo de embeddings de OpenAI
        
    Returns:
        Chroma: Vectorstore persistente listo para consultas de similitud
    """
    directorio_chroma = "./chroma_db"
    
    try:
        # Intentar cargar vectorstore existente
        if os.path.exists(directorio_chroma) and os.listdir(directorio_chroma):
            st.info("Cargando vectorstore existente desde caché...")
            vectorstore = Chroma(
                persist_directory=directorio_chroma,
                embedding_function=embeddings
            )
            
            # Verificar si el vectorstore tiene datos
            if vectorstore._collection.count() > 0:
                st.success(f"Vectorstore cargado: {vectorstore._collection.count()} documentos")
                return vectorstore
            else:
                st.warning("Vectorstore vacío, recreando...")
        
        # Crear nuevo vectorstore si no existe o está vacío
        st.info("Creando nuevo vectorstore...")
        with st.spinner("Generando embeddings vectoriales... Esto puede tomar unos momentos."):
            vectorstore = Chroma.from_documents(
                documents=documentos,
                embedding=embeddings,
                persist_directory=directorio_chroma
            )
        
        st.success(f"Vectorstore creado exitosamente: {len(documentos)} documentos vectorizados")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error al construir vectorstore: {str(e)}")
        # Intentar limpiar directorio corrupto y recrear
        if os.path.exists(directorio_chroma):
            import shutil
            shutil.rmtree(directorio_chroma)
            st.warning("Directorio corrupto eliminado, reintentando...")
            return construir_vectorstore_chroma(documentos, embeddings)
        raise e


def limpiar_vectorstore_cache():
    """
    Elimina el vectorstore en caché para forzar reconstrucción con nuevos datos.
    Útil cuando se carga un nuevo archivo CSV.
    """
    directorio_chroma = "./chroma_db"
    if os.path.exists(directorio_chroma):
        import shutil
        shutil.rmtree(directorio_chroma)
        st.info("Cache de vectorstore eliminado")
    
    # Limpiar también cache de sesión
    if "vectorstore_csv" in st.session_state:
        st.session_state.vectorstore_csv = None


@st.cache_resource
def construir_modelo_chat_openai(api_key: str) -> ChatOpenAI:
    """
    Construye el modelo de chat GPT-4o-mini con configuración optimizada.
    Utiliza cache de Streamlit para reutilización eficiente.
    
    Args:
        api_key: Clave API de OpenAI
        
    Returns:
        ChatOpenAI: Modelo de lenguaje configurado para conversaciones
    """
    return ChatOpenAI(
        model="gpt-4o-mini",          # Modelo eficiente y de alta calidad
        openai_api_key=api_key,
        temperature=0,                # Determinístico para respuestas consistentes
        max_tokens=1000,              # Límite de tokens para controlar costos
        streaming=True                # Habilitar streaming para mejor UX
    )


def construir_cadena_conversacional(vectorstore: Chroma, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Construye una cadena conversacional que combina recuperación vectorial
    con generación de respuestas manteniendo memoria de conversación.
    
    Args:
        vectorstore: Almacén vectorial para búsqueda de similitud
        llm: Modelo de lenguaje para generación de respuestas
        
    Returns:
        Dict[str, Any]: Diccionario con retriever y cadena de procesamiento
    """
    # Configurar retriever con parámetros optimizados
    retriever = vectorstore.as_retriever(
        search_type="similarity",     # Búsqueda por similitud semántica
        search_kwargs={
            "k": 5                    # Recuperar top 5 documentos más relevantes
        }
    )
    
    # Crear prompt template para el contexto
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente especializado en análisis de datos CSV. "
                  "Usa el siguiente contexto para responder las preguntas del usuario de manera precisa y útil. "
                  "Contexto: {context}"),
        ("human", "{question}")
    ])
    
    # Crear cadena de procesamiento de documentos
    document_chain = prompt_template | llm | StrOutputParser()
    
    return {
        "retriever": retriever,
        "document_chain": document_chain,
        "prompt_template": prompt_template
    }


def procesar_consulta_usuario(cadena: Dict[str, Any], pregunta: str, historial_chat: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Procesa una consulta del usuario usando la cadena conversacional
    y retorna la respuesta con documentos fuente.
    
    Args:
        cadena: Diccionario con retriever y cadena de procesamiento
        pregunta: Pregunta del usuario
        historial_chat: Historial de conversación previa
        
    Returns:
        Dict[str, Any]: Diccionario con respuesta y documentos fuente
    """
    try:
        # Recuperar documentos relevantes
        retriever = cadena["retriever"]
        document_chain = cadena["document_chain"]
        
        documentos_relevantes = retriever.invoke(pregunta)
        
        # Preparar contexto desde los documentos
        contexto = "\n\n".join([doc.page_content for doc in documentos_relevantes])
        
        # Generar respuesta usando la cadena de documentos
        respuesta = document_chain.invoke({
            "context": contexto,
            "question": pregunta
        })
        
        return {
            "respuesta": respuesta,
            "documentos_fuente": documentos_relevantes,
            "exito": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "respuesta": f"Error al procesar la consulta: {str(e)}",
            "documentos_fuente": [],
            "exito": False,
            "error": str(e)
        }


def renderizar_interfaz_carga_archivo():
    """
    Renderiza la sección de la interfaz para carga y validación de archivos CSV.
    Maneja el estado de carga y proporciona feedback visual al usuario.
    """
    st.header("Paso 1: Cargar Archivo CSV")
    
    # Widget de carga de archivo
    archivo_subido = st.file_uploader(
        "Selecciona un archivo CSV para analizar",
        type=['csv'],
        help="Archivo máximo: 5MB. Formatos soportados: CSV con codificación UTF-8 o Latin-1"
    )
    
    # Procesar archivo si se subió uno nuevo
    if archivo_subido is not None:
        if validar_archivo_csv(archivo_subido):
            # Cargar datos del CSV
            df = cargar_datos_csv(archivo_subido)
            
            if df is not None:
                # Almacenar datos en estado de sesión
                st.session_state.datos_csv = df
                st.session_state.archivo_csv_cargado = True
                
                # Mostrar previsualización de datos
                st.subheader("Vista Previa de Datos")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Mostrar estadísticas básicas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Filas Totales", len(df))
                with col2:
                    st.metric("Columnas", len(df.columns))
                with col3:
                    st.metric("Memoria", f"{df.memory_usage().sum() / 1024:.1f} KB")
                
                # Botón para limpiar vectorstore anterior
                if st.button("Procesar Nuevo Archivo", type="primary"):
                    limpiar_vectorstore_cache()
                    st.success("Listo para procesar el nuevo archivo")
                    st.rerun()


def renderizar_interfaz_construccion_vectorstore():
    """
    Renderiza la sección de construcción del vectorstore y cadena conversacional.
    Maneja la creación de embeddings y configuración del sistema de chat.
    """
    if not st.session_state.archivo_csv_cargado:
        st.warning("Primero debes cargar un archivo CSV válido")
        return
    
    st.header("Paso 2: Configurar Sistema de Chat")
    
    try:
        # Obtener clave API
        api_key = configurar_credenciales_openai()
        
        # Mostrar estado de configuración
        col1, col2 = st.columns(2)
        with col1:
            st.success("Credenciales OpenAI configuradas")
        with col2:
            if st.session_state.vectorstore_csv is not None:
                st.success("Sistema de chat listo")
            else:
                st.info("Construyendo sistema de chat...")
        
        # Construir sistema si no existe
        if st.session_state.vectorstore_csv is None:
            with st.spinner("Procesando archivo CSV y creando vectorstore..."):
                # Convertir DataFrame a documentos
                documentos = convertir_csv_a_documentos(st.session_state.datos_csv)
                documentos_divididos = dividir_documentos_grandes(documentos)
                
                # Construir embeddings y vectorstore
                embeddings = construir_embeddings_openai(api_key)
                vectorstore = construir_vectorstore_chroma(documentos_divididos, embeddings)
                
                # Construir cadena conversacional
                llm = construir_modelo_chat_openai(api_key)
                cadena = construir_cadena_conversacional(vectorstore, llm)
                
                # Almacenar en estado de sesión
                st.session_state.vectorstore_csv = vectorstore
                st.session_state.cadena_conversacional = cadena
                
                st.success("Sistema de chat configurado exitosamente!")
                st.rerun()
        
    except ValueError as e:
        st.error(f"Error de configuración: {str(e)}")
        st.info("Por favor, configura tu clave API de OpenAI en el archivo secrets/api_keys.toml")


def renderizar_interfaz_chat():
    """
    Renderiza la interfaz principal de chat con historial de conversación
    y procesamiento de consultas en tiempo real.
    """
    if st.session_state.cadena_conversacional is None:
        st.warning("Primero debes completar la configuración del sistema de chat")
        return
    
    st.header("Paso 3: Chatear con tus Datos CSV")
    
    # Contenedor para el historial de chat
    contenedor_chat = st.container()
    
    # Mostrar historial de conversación
    with contenedor_chat:
        for mensaje in st.session_state.historial_conversacion:
            with st.chat_message(mensaje["role"]):
                st.markdown(mensaje["content"])
                
                # Mostrar documentos fuente si es respuesta del asistente
                if mensaje["role"] == "assistant" and "documentos_fuente" in mensaje:
                    if mensaje["documentos_fuente"]:
                        with st.expander("Ver documentos fuente"):
                            for i, doc in enumerate(mensaje["documentos_fuente"]):
                                st.text(f"Documento {i+1}: {doc.page_content[:200]}...")
    
    # Input para nueva consulta
    if pregunta_usuario := st.chat_input("Haz una pregunta sobre tus datos CSV..."):
        # Agregar mensaje del usuario al historial
        st.session_state.historial_conversacion.append({
            "role": "user",
            "content": pregunta_usuario
        })
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)
        
        # Procesar consulta y generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Analizando datos y generando respuesta..."):
                resultado = procesar_consulta_usuario(
                    st.session_state.cadena_conversacional,
                    pregunta_usuario,
                    st.session_state.historial_conversacion
                )
            
            # Mostrar respuesta
            st.markdown(resultado["respuesta"])
            
            # Agregar respuesta al historial
            mensaje_asistente = {
                "role": "assistant",
                "content": resultado["respuesta"],
                "documentos_fuente": resultado["documentos_fuente"]
            }
            st.session_state.historial_conversacion.append(mensaje_asistente)
            
            # Mostrar documentos fuente si existen
            if resultado["documentos_fuente"]:
                with st.expander("Ver documentos fuente"):
                    for i, doc in enumerate(resultado["documentos_fuente"]):
                        st.text(f"Documento {i+1}: {doc.page_content[:200]}...")
    
    # Botón para limpiar historial
    if st.session_state.historial_conversacion:
        if st.button("Limpiar Historial de Chat"):
            st.session_state.historial_conversacion = []
            st.rerun()


def main():
    """
    Función principal que orquesta toda la aplicación Streamlit.
    Configura la página, inicializa el estado de sesión y renderiza
    los módulos de la interfaz de usuario en secuencia lógica.
    """
    # Configuración de la página Streamlit
    st.set_page_config(
        page_title="CSV Chatbot - LangChain & OpenAI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Título principal y descripción
    st.title("Chatbot CSV con LangChain y OpenAI")
    st.markdown("""
    **Sistema inteligente de consultas sobre datos CSV usando:**
    - **LangChain** para procesamiento de documentos y cadenas conversacionales
    - **ChromaDB** para almacenamiento vectorial persistente y búsqueda semántica
    - **OpenAI text-embedding-3-large** para vectorización semántica de documentos
    - **OpenAI GPT-4o-mini** para generación de respuestas contextuales
    - **Streamlit** para interfaz de usuario interactiva
    """)
    
    # Inicializar estado de sesión
    inicializar_estado_sesion()
    
    # Barra lateral con información del sistema
    with st.sidebar:
        st.header("Información del Sistema")
        
        # Estado del archivo CSV
        if st.session_state.archivo_csv_cargado:
            st.success("Archivo CSV cargado")
            if st.session_state.datos_csv is not None:
                st.write(f"{len(st.session_state.datos_csv)} filas")
        else:
            st.info("Sin archivo CSV")
        
        # Estado del vectorstore
        if st.session_state.vectorstore_csv is not None:
            st.success("Vectorstore activo")
        else:
            st.info("Vectorstore pendiente")
        
        # Estado de la cadena conversacional
        if st.session_state.cadena_conversacional is not None:
            st.success("Chat configurado")
        else:
            st.info("Chat pendiente")
        
        # Número de mensajes en historial
        num_mensajes = len(st.session_state.historial_conversacion)
        st.write(f"{num_mensajes} mensajes en historial")
        
        st.markdown("---")
        st.markdown("**Configuración técnica:**")
        st.code("""
Modelo LLM: gpt-4o-mini
Embeddings: text-embedding-3-large
Vector DB: ChromaDB
Chunk size: 1000 caracteres
Temperature: 0 (determinístico)
        """)
    
    # Renderizar interfaces principales en secuencia
    try:
        # Paso 1: Carga de archivo
        renderizar_interfaz_carga_archivo()
        
        st.markdown("---")
        
        # Paso 2: Construcción del sistema
        renderizar_interfaz_construccion_vectorstore()
        
        st.markdown("---")
        
        # Paso 3: Interface de chat
        renderizar_interfaz_chat()
        
    except Exception as e:
        st.error(f"Error inesperado en la aplicación: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()