# Chatbot CSV con LangChain y OpenAI

## Descripción del Proyecto

Aplicación completa de Streamlit que permite cargar archivos CSV y consultar su contenido a través de un chatbot conversacional inteligente. El sistema utiliza tecnologías de vanguardia en procesamiento de lenguaje natural y recuperación de información.

### Arquitectura Técnica

- **Frontend**: Streamlit para interfaz de usuario interactiva
- **LLM**: OpenAI GPT-4o-mini para generación de respuestas contextuales
- **Embeddings**: OpenAI text-embedding-3-large para representación vectorial
- **Vector Database**: ChromaDB para almacenamiento persistente y búsqueda semántica
- **Framework NLP**: LangChain para orquestación de cadenas conversacionales

## Estructura del Proyecto

```
T8_-_Ejercicio 2/
├── app_csv_chatbot.py           # Aplicación principal (archivo único)
├── .streamlit/
│   └── config.toml              # Configuración de Streamlit
├── secrets/
│   └── api_keys.toml            # Claves API (NO subir a GitHub)
├── chroma_db/                   # Base de datos vectorial persistente
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Exclusión de archivos sensibles
├── README.md                    # Este archivo
└── venv_info/                   # Metadatos del entorno virtual
```

## Funcionalidades Principales

### 1. Carga y Procesamiento de Datos
- Validación automática de archivos CSV (máximo 5MB)
- Soporte para múltiples codificaciones (UTF-8, Latin-1)
- Conversión automática de filas CSV a documentos LangChain
- División inteligente de documentos largos para optimizar embeddings

### 2. Almacenamiento Vectorial
- Generación de embeddings semánticos usando OpenAI text-embedding-3-large
- Persistencia local con ChromaDB para reutilización eficiente
- Cache inteligente para evitar recomputación innecesaria
- Búsqueda por similitud semántica para recuperación contextual

### 3. Sistema Conversacional
- Cadena conversacional con memoria persistente de diálogo
- Recuperación de documentos fuente para transparencia
- Respuestas determinísticas (temperature=0) para consistencia
- Manejo de errores y recuperación graceful

### 4. Interfaz de Usuario
- Flujo de trabajo paso a paso intuitivo
- Previsualización de datos CSV cargados
- Chat en tiempo real con historial persistente
- Sidebar informativo con estado del sistema

## Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- Entorno virtual `chatcsv` (Conda)
- Clave API de OpenAI válida

### Instalación de Dependencias

```bash
# Activar entorno virtual
conda activate chatcsv

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de API

1. **Opción 1: Archivo local**
   Edita `secrets/api_keys.toml`:
   ```toml
   [openai]
   api_key = "tu_clave_openai_real"
   ```

2. **Opción 2: Variable de entorno**
   ```bash
   export OPENAI_API_KEY="tu_clave_openai_real"
   ```

3. **Opción 3: Streamlit Cloud**
   Configura en Settings → Secrets:
   ```toml
   [openai]
   api_key = "tu_clave_openai_real"
   ```

## Uso de la Aplicación

### Ejecución Local

```bash
# Activar entorno virtual
conda activate chatcsv

# Ejecutar aplicación
streamlit run app_csv_chatbot.py
```

### Flujo de Trabajo

1. **Paso 1: Cargar CSV**
   - Selecciona archivo CSV (máximo 5MB)
   - Visualiza previsualización y estadísticas
   - Confirma procesamiento

2. **Paso 2: Configurar Sistema**
   - El sistema valida credenciales OpenAI
   - Se generan embeddings vectoriales automáticamente
   - Se configura cadena conversacional

3. **Paso 3: Chatear**
   - Realiza consultas en lenguaje natural
   - Visualiza respuestas contextuales
   - Explora documentos fuente utilizados

### Ejemplos de Consultas

- "¿Cuántas filas tiene el dataset?"
- "Muéstrame un resumen de los datos de ventas por región"
- "¿Cuál es el valor promedio de la columna precio?"
- "Encuentra registros con ventas superiores a 1000"

## Características Técnicas Avanzadas

### Optimizaciones de Rendimiento
- **Cache de recursos**: `@st.cache_resource` para objetos pesados
- **Persistencia vectorial**: Reutilización de embeddings calculados
- **Streaming**: Respuestas en tiempo real del LLM
- **Chunking inteligente**: División óptima de documentos largos

### Manejo de Errores
- Validación exhaustiva de archivos CSV
- Recuperación automática de vectorstore corrupto
- Fallback de codificación para archivos problemáticos
- Mensajes de error informativos para el usuario

### Seguridad y Buenas Prácticas
- Exclusión de archivos sensibles con `.gitignore`
- Múltiples métodos de configuración de API keys
- No exposición de credenciales en código fuente
- Límites de tamaño para prevenir abuso de recursos

## Arquitectura de Software

### Modularización del Código

```python
# Módulo 1: Configuración inicial
configurar_credenciales_openai()
inicializar_estado_sesion()

# Módulo 2: Procesamiento de datos
cargar_datos_csv()
convertir_csv_a_documentos()
dividir_documentos_grandes()

# Módulo 3: Gestión vectorial
construir_embeddings_openai()
construir_vectorstore_chroma()
limpiar_vectorstore_cache()

# Módulo 4: Sistema conversacional
construir_modelo_chat_openai()
construir_cadena_conversacional()
procesar_consulta_usuario()

# Módulo 5: Interfaz de usuario
renderizar_interfaz_carga_archivo()
renderizar_interfaz_construccion_vectorstore()
renderizar_interfaz_chat()
```

### Flujo de Datos

```
CSV Input → Pandas DataFrame → LangChain Documents → 
Text Splitting → OpenAI Embeddings → ChromaDB Storage → 
Retrieval Chain → Conversational Memory → GPT-4o-mini → 
Streamlit UI
```

## Despliegue en Streamlit Cloud

1. Subir código a GitHub (excluir `/secrets/` con `.gitignore`)
2. Conectar repositorio en Streamlit Cloud
3. Configurar secrets en la plataforma:
   ```toml
   [openai]
   api_key = "tu_clave_openai_real"
   ```
4. La aplicación se desplegará automáticamente

## Limitaciones y Consideraciones

- **Tamaño de archivo**: Límite de 5MB para CSVs
- **Costo de API**: Uso de OpenAI genera costos por tokens
- **Memoria**: ChromaDB requiere RAM proporcional al dataset
- **Latencia**: Primera consulta más lenta por construcción de vectorstore

## Extensiones Futuras

- Soporte para múltiples formatos (Excel, JSON, Parquet)
- Interface para carga de múltiples archivos
- Análisis estadístico automático con visualizaciones
- Exportación de conversaciones y respuestas
- Configuración de parámetros avanzados desde UI

## Troubleshooting

### Problemas Comunes

1. **Error de API Key**
   - Verifica configuración en `secrets/api_keys.toml`
   - Confirma variable de entorno `OPENAI_API_KEY`

2. **Archivo CSV no carga**
   - Verifica codificación (UTF-8/Latin-1)
   - Confirma tamaño < 5MB
   - Revisa formato CSV válido

3. **Vectorstore corrupto**
   - Elimina directorio `chroma_db/`
   - Reinicia aplicación para reconstrucción

4. **Memoria insuficiente**
   - Reduce tamaño del CSV
   - Ajusta `chunk_size` en configuración

## Licencia

Proyecto educativo desarrollado para aprendizaje de tecnologías NLP y sistemas conversacionales.