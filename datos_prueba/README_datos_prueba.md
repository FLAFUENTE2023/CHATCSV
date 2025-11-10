# Datos de Prueba - CSV Chatbot

## Descripción del Directorio

Este directorio contiene archivos CSV de prueba diseñados específicamente para demostrar las capacidades del chatbot CSV desarrollado con LangChain y OpenAI.

## Archivos Incluidos

### `ventas_tecnologia_q1_2024.csv`
**Descripción:** Dataset de ventas de productos tecnológicos del primer trimestre de 2024

**Estructura de datos:**
- **id_producto:** Identificador único del producto (P001-P030)
- **nombre_producto:** Nombre comercial del producto
- **categoria:** Clasificación por tipo (Tecnologia, Accesorios, Audio, etc.)
- **precio_unitario:** Precio por unidad en USD
- **ventas_unidades:** Cantidad de unidades vendidas
- **fecha_venta:** Fecha de la transacción (formato YYYY-MM-DD)
- **region:** Zona geográfica (Norte, Sur, Este, Oeste)
- **vendedor:** Nombre del representante de ventas
- **metodo_pago:** Forma de pago utilizada
- **descuento_aplicado:** Porcentaje de descuento (0.00-0.20)
- **ingresos_totales:** Ingresos netos después de descuentos

**Características del dataset:**
- 30 registros de productos diversos
- Múltiples categorías para análisis segmentado
- Datos temporales para análisis de tendencias
- Información geográfica y de vendedores
- Métricas financieras calculadas

## Consultas de Ejemplo Recomendadas

### Análisis Básico
- "¿Cuántos productos hay en el dataset?"
- "¿Cuáles son las categorías de productos disponibles?"
- "Muestra los primeros 5 productos más caros"

### Análisis por Categoría
- "¿Cuántos productos de tecnología se vendieron?"
- "¿Cuál es el promedio de precio en la categoría Audio?"
- "Lista todos los accesorios ordenados por ventas"

### Análisis Temporal
- "¿Cuáles fueron las ventas de enero vs febrero?"
- "¿Qué día se registraron mayores ingresos?"
- "Muestra la evolución de ventas por fecha"

### Análisis Geográfico
- "¿Qué región genera más ingresos totales?"
- "Compara las ventas entre Norte y Sur"
- "¿Cuál es el producto más vendido por región?"

### Análisis de Vendedores
- "¿Quién es el vendedor con mejores resultados?"
- "¿Cuántos productos vendió Ana García?"
- "Ranking de vendedores por ingresos generados"

### Análisis Financiero
- "¿Cuál es el ingreso total del periodo?"
- "¿Qué productos tienen mayor descuento aplicado?"
- "Calcula el ticket promedio por transacción"

## Notas de Uso

- El dataset está optimizado para pruebas de recuperación semántica
- Contiene suficiente variedad para consultas complejas multi-criterio
- Los datos son realistas pero sintéticos
- Ideal para demostrar capacidades de análisis conversacional

## Extensión del Dataset

Para pruebas adicionales, se pueden agregar:
- Más registros temporales (Q2, Q3, Q4)
- Nuevas categorías de productos
- Métricas adicionales (costos, márgenes, inventario)
- Información de clientes
- Datos de satisfacción y retorno