# Chatbot CSV con Memoria Conversacional: Implementación Avanzada con LangChain y OpenAI

## 1. Propósito Académico

### Contexto del Ejercicio

Este proyecto constituye el **Ejercicio 2 del Tema 8** del módulo de Procesamiento de Lenguaje Natural (NLP) desarrollado como implementación de una evolución técnica del artículo de referencia "Build a Chatbot on Your CSV Data With LangChain and OpenAI" (Yvann, Better Programming, 2023).

### Problema Identificado

El ejercicio 1 del Team 8 (immediatamente anterior al presente ejercicio) ya implementa un sistema de consulta sobre datos mediante recuperación vectorial, pero adolece de una limitación fundamental: **la ausencia de memoria conversacional**. Cada consulta se procesa de forma aislada, sin considerar el contexto histórico de la interacción, lo que genera:

- Respuestas descontextualizadas en consultas consecutivas
- Incapacidad para referencias anafóricas ("¿y sobre lo anterior?")
- Pérdida de coherencia temática en diálogos extendidos
- Experiencia de usuario fragmentada y poco natural

### Desafío Técnico

El objetivo central de este ejercicio T8 - 2 consiste en **integrar memoria conversacional persistente** sin comprometer la funcionalidad de recuperación semántica sobre datos tabulares. Esto implica:

1. Mantener el historial de intercambios usuario-asistente
2. Incorporar dicho contexto en cada nueva consulta al modelo de lenguaje
3. Optimizar el uso de tokens para conversaciones extensas
4. Preservar la capacidad de análisis específico sobre datos CSV

## 2. Evolución Técnica Respecto al artículo original "Build a Chatbot on Your CSV Data With LangChain and OpenAI" (Yvann, Better Programming, 2023) que inspira este ejercicio.

### Arquitectura de Almacenamiento Vectorial

**Migración de FAISS a ChromaDB:**
- El artículo original emplea FAISS (Facebook AI Similarity Search) para indexación vectorial
- La implementación desarrollada sustituye FAISS por **ChromaDB**, proporcionando:
  - Persistencia automática de embeddings entre sesiones
  - Interface más estable con LangChain
  - Gestión simplificada de metadatos documentales
  - Mayor robustez ante fallos de sistema

**Modernización de Embeddings:**
- Sustitución de modelos de embedding locales obsoletos por **OpenAI text-embedding-3-large**
- Mejora significativa en calidad semántica y coherencia vectorial
- Compatibilidad garantizada con infraestructura OpenAI

### Sistema de Memoria Conversacional

**Implementación de Historial Dinámico:**
La innovación principal radica en la gestión inteligente del contexto conversacional mediante tres mecanismos complementarios:

1. **Almacenamiento en Session State:** Utilización de `st.session_state` para persistencia del historial durante la sesión activa
2. **Límite Dinámico de Memoria:** Mantenimiento automático de los últimos 20 intercambios para optimización de tokens
3. **Resumen Automático:** Compresión inteligente de conversaciones antiguas (resumen) mediante GPT-4o-mini cuando se excede el límite establecido

### Actualización de Framework

**Migración de Gradio a Streamlit:**
- Abandono de Gradio en favor de **Streamlit** para mayor control sobre la persistencia de sesión
- Interface reproducible tanto en ejecución local como en Streamlit Cloud
- Gestión robusta de estado entre interacciones

**Modernización de LangChain:**
- Actualización de importaciones conforme a la estructura modular actual de LangChain
- Sustitución de `ConversationalRetrievalChain` (deprecado) por composición manual de cadenas
- Adaptación a las APIs actuales de `langchain-core`, `langchain-openai`, y `langchain-chroma`

### Optimizaciones de Rendimiento

**Sistema de Caché Inteligente:**
- Implementación de `@st.cache_resource` para embeddings y vectorstore
- Reducción significativa de recomputación innecesaria
- Optimización de costos de API mediante reutilización de recursos

**Gestión Robusta de Errores:**
- Validación exhaustiva de claves API con múltiples fuentes de configuración
- Manejo de archivos CSV corruptos o mal codificados
- Recuperación automática ante fallos en vectorstore

## 3. Arquitectura Conceptual y Flujo Conversacional

### Pipeline de Procesamiento de Datos

1. **Carga y Validación:** El usuario carga un archivo CSV, sometido a validación de formato, codificación y tamaño
2. **Fragmentación Inteligente:** División del dataset en documentos individuales por fila, con división adicional para registros extensos
3. **Vectorización Semántica:** Generación de embeddings mediante OpenAI text-embedding-3-large
4. **Indexación Persistente local:** Almacenamiento en ChromaDB con persistencia (local) automática
5. **Indexacion Persistente en Streamlit Cloud:**
Streamlit guarda ChromaDB en memoria del contenedor que ejecuta la app.
Es un objeto cacheado en RAM del servidor. Si Streamlit reinicia el contenedor (por inactividad o redeploy) todo lo almacenado se pierde.


### Flujo Conversacional Integrado

**Procesamiento de Consulta:**
1. **Análisis de Intención:** Determinación automática de la necesidad de recuperación vectorial
2. **Recuperación Contextual:** Búsqueda de documentos relevantes en el vectorstore
3. **Construcción del Prompt:** Integración de:
   - Contexto de datos CSV recuperados
   - Historial conversacional reciente
   - Resumen de interacciones previas (si es aplicable)
4. **Generación de Respuesta:** Invocación de GPT-4o-mini con contexto completo
5. **Actualización de Memoria:** Incorporación del nuevo intercambio al historial persistente

**Gestión de Memoria Adaptativa:**
El sistema implementa una estrategia híbrida de memoria que combina:
- **Memoria Explícita:** Historial reciente de intercambios (últimos 20 mensajes)
- **Memoria Comprimida:** Resumen automático de conversaciones anteriores
- **Memoria Semántica:** Acceso a datos CSV mediante recuperación vectorial

### Coherencia Conversacional

El diseño garantiza coherencia temporal mediante:
- **Referencias Anafóricas:** Capacidad de resolver referencias a elementos mencionados previamente
- **Continuidad Temática:** Mantenimiento del hilo conversacional entre consultas
- **Contextualización Progresiva:** Acumulación inteligente de contexto sin degradación por exceso de información

## 4. Innovaciones Pedagógicas y Técnicas

### Documentación Académica Integral

**Comentarios Profesionales en Español:**
- Documentación a nivel universitario con explicaciones conceptuales rigurosas
- Comentarios inline que explican tanto el "qué" como el "por qué" de cada decisión técnica
- Docstrings exhaustivos con descrición de arquitectura, parámetros y comportamiento esperado

### Reproducibilidad Técnica

**Arquitectura Cloud-Ready:**
- Diseño compatible tanto para ejecución local como en Streamlit Cloud
- Gestión de credenciales mediante múltiples fuentes (variables de entorno, archivos locales, secretos de plataforma)
- Ausencia de dependencias de sistema específicas

**Modularización Funcional:**
- Separación clara de responsabilidades en funciones especializadas
- Interface uniforme entre módulos que facilita testing y mantenimiento
- Encapsulación de la lógica de negocio en un único archivo ejecutable

### Control de Calidad Avanzado

**Manejo Defensivo de Fallos:**
- Validación proactiva de inputs y estados del sistema
- Recuperación graceful ante errores de conectividad o API
- Feedback informativo al usuario en situaciones de error

**Optimización de Recursos:**
- Gestión inteligente de tokens mediante compresión de historial
- Caché de recursos computacionalmente costosos
- Límites configurables para prevención de abuso de recursos

## 5. Evaluación del Cumplimiento del Objetivo Docente

### Análisis de Funcionalidad Conversacional

**Evidencia de Memoria Efectiva:**
El sistema desarrollado demuestra cumplimiento del objetivo académico mediante:

1. **Persistencia Observable:** El historial conversacional permanece visible en la interface, permitiendo verificación directa de la memoria del sistema

2. **Coherencia Referencial:** El chatbot resuelve correctamente referencias a elementos mencionados en intercambios previos, evidenciando comprensión contextual

3. **Progresión Temática:** Las respuestas muestran continuidad lógica respecto a consultas anteriores, evitando repeticiones innecesarias o contradicciones

### Preservación de Funcionalidad Original

**Mantenimiento de Capacidades CSV:**
- La funcionalidad de análisis sobre datos tabulares se mantiene intacta
- La precisión de respuestas basadas en recuperación vectorial no se ve comprometida
- La capacidad de análisis estadístico y consulta específica permanece funcional

### Mejoras Arquitectónicas Verificables

**Superiodidad Técnica Respecto al Artículo Base:**
1. **Robustez:** Mayor estabilidad mediante gestión de errores y validaciones
2. **Escalabilidad:** Optimización de memoria que permite conversaciones extendidas
3. **Mantenibilidad:** Código modular y bien documentado que facilita extensiones futuras
4. **Usabilidad:** Interface más intuitiva con feedback claro de estado del sistema

### Cumplimiento del Objetivo Académico

**Conclusión Evaluativa:**
El proyecto alcanza satisfactoriamente el objetivo planteado de "mantener contexto histórico de conversación" mediante la implementación de un sistema de memoria híbrido que:

- **Preserva** la funcionalidad de consulta sobre CSV del artículo original
- **Extiende** las capacidades mediante memoria conversacional persistente
- **Mejora** la arquitectura técnica con optimizaciones modernas
- **Documenta** el proceso con rigor académico apropiado para el nivel universitario

### Direcciones de Investigación Futura

**Extensiones Conceptuales:**
1. **RAG Multifuente:** Integración de múltiples tipos de documentos (PDF, SQL, APIs externas)
2. **Memoria Episódica:** Implementación de memoria a largo plazo con persistencia entre sesiones
3. **Evaluación Cuantitativa:** Desarrollo de métricas específicas para calidad conversacional en contexto de análisis de datos
4. **Personalización Adaptativa:** Aprendizaje de preferencias de usuario para optimización de respuestas

El ejercicio establece una base sólida para la exploración de sistemas conversacionales más sofisticados en el ámbito del análisis de datos, cumpliendo tanto los objetivos pedagógicos inmediatos como proporcionando fundamentos para investigación avanzada en NLP aplicado.