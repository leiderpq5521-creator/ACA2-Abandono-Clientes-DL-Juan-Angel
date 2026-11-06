# GUION TÉCNICO Y LOCUCIÓN DE VIDEO (3 A 5 MINUTOS)
## PROYECTO ACA N° 2 – APRENDIZAJE AUTOMÁTICO · PILAR VENTAS
**Tema:** Del ML clásico al Deep Learning: predicción del abandono de clientes a partir de su secuencia semanal de compras (GRU, LSTM y Transformer) y despliegue en Streamlit
**Institución:** Corporación Unificada Nacional de Educación Superior (CUN) · Especialización en Inteligencia Artificial
**Docente:** Hamilton Rivera Flor
**Integrantes:** Juan Guillermo Angel Jaramillo · Luis Bernardo Correa Restrepo · María Alejandra Manga Betancourt

**Duración estimada:** ≈ 4.3 min (585 palabras a ≈ 135 palabras/min).
**Enlaces para la demo:**
- App: https://aca2-abandono-clientes-dl-juan-angel-tuqs2bahdtmtheu3bhrzfa.streamlit.app/
- Código: https://github.com/leiderpq5521-creator/ACA2-Abandono-Clientes-DL-Juan-Angel
- Póster: `entregables/Poster_ACA2_Ventas_DL.pdf`

> **Antes de grabar:** abre la app unos minutos antes (en Streamlit Cloud tarda ≈ 30 s en «despertar» si estuvo inactiva) y deja listo el póster en PDF.

---

## TABLA ESCENA POR ESCENA

| Escena y tiempo | Recurso visual en pantalla | Locutor sugerido | Texto de locución |
|---|---|---|---|
| **1 · 0:00–0:30 (30 s)** | Encabezado del póster (título, integrantes). | Juan Guillermo Angel | "Cordial saludo. Presentamos el ACA número 2 de Aprendizaje Automático de la Especialización en Inteligencia Artificial de la CUN, con el docente Hamilton Rivera Flor. Somos Juan Guillermo Angel, Luis Bernardo Correa y María Alejandra Manga. En el ACA 1 predijimos qué clientes de una tienda en línea dejarían de comprar, con un Random Forest que veía sólo tres números por cliente: recencia, frecuencia y gasto. Es como mirar una foto. En el ACA 2 la red neuronal ve la película completa: la historia de compras semana a semana." |
| **2 · 0:30–1:15 (45 s)** | `figuras/01_ventas_mensuales_pais.png` → en la app, pestaña **Historial (SQL)**, consulta «Ventas mensuales por país» → `figuras/03_mapa_clientes_semanas.png`. | Luis Bernardo Correa | "Usamos el dataset Online Retail de UCI: 541.909 líneas de factura entre diciembre de 2010 y diciembre de 2011. Todo se gestiona en una base de datos SQLite: la calidad, el análisis exploratorio y la limpieza se hacen con consultas y vistas SQL. Quitamos líneas sin cliente, duplicados, cancelaciones y precios no válidos. Después convertimos a cada uno de los 3.314 clientes en una secuencia de 40 semanas con siete canales: gasto, pedidos, unidades, productos, devoluciones, días sin comprar y si ya era cliente. En este mapa se ve la diferencia: quienes abandonan se van apagando antes del corte." |
| **3 · 1:15–2:05 (50 s)** | `figuras/04_aumento_datos.png` → diagrama de arquitectura del póster. | María Alejandra Manga | "Para que la comparación sea justa usamos la misma etiqueta y exactamente los mismos clientes de prueba del ACA 1, y lo verificamos en el código. Nada del futuro entra en las secuencias y cada cliente está en una sola partición. En entrenamiento aplicamos aumento de datos para series: escalado, ruido y recorte de la ventana. Comparamos tres redes: una GRU y una LSTM, que leen las semanas en orden y deciden qué recordar, y un Transformer temporal, en el que cada semana es un token y la autoatención relaciona todas las semanas entre sí. Cada red se entrenó con cinco semillas y elegimos la de mejor resultado en validación: la GRU." |
| **4 · 2:05–2:55 (50 s)** | Bloque de resultados del póster (KPIs y tabla) → `figuras/13_poster_modelo_app.png` → `figuras/09_desglose.png`. | Juan Guillermo Angel | "En los 663 clientes de prueba la GRU alcanza un área bajo la curva ROC de 0,73, frente a 0,72 del Random Forest. Las tres redes superan al modelo clásico por un margen parecido, pero el intervalo de confianza de la diferencia toca el cero, así que la mejora es pequeña y no concluyente: con compras tan esporádicas, buena parte de la señal ya estaba en recencia, frecuencia y gasto. Con la misma meta de detectar al menos el 80 por ciento de los abandonos, la red da 170 falsas alarmas frente a 188 del Random Forest. Donde más aporta la secuencia es en el segmento Recientes." |
| **5 · 2:55–4:00 (65 s)** | **Demo en vivo**: 1) pestaña *Cliente existente* con un ejemplo de prueba (probabilidad, gráfico semanal coloreado y explicación); 2) cambiar de modelo en la barra lateral; 3) *Secuencia manual* con las plantillas «Apagándose» y «Estable»; 4) *Lotes (CSV)* con el lote de ejemplo y descarga; 5) *Historial (SQL)*. | Luis Bernardo Correa | "Esta es la aplicación en Streamlit. Elegimos un cliente del conjunto de prueba: la red calcula su probabilidad de abandono y colorea cada semana según cuánto pesó en la decisión. Para eso borramos una semana a la vez y medimos cuánto cambia la predicción. Podemos cambiar de modelo y comparar con el Random Forest del ACA 1. En la pestaña de secuencia manual simulamos un cliente que se va apagando y otro estable, y vemos cómo cambia el riesgo. También procesamos un archivo CSV por lotes y descargamos el resultado. Cada predicción queda guardada en la base SQLite y se puede consultar con SQL desde la misma app." |
| **6 · 4:00–4:30 (30 s)** | Conclusiones del póster → enlaces de App, Video y Código. | María Alejandra Manga | "En conclusión, las redes recurrentes y de atención aprenden el ritmo de compra y la importancia de lo reciente, e igualan o superan ligeramente al Random Forest en el mismo problema. Su límite son los clientes con una sola compra. Lo valioso para el negocio es una herramienta que explica cada predicción semana a semana. El código, el notebook y la app están en los enlaces del póster. Muchas gracias." |

---

## TEXTO CONTINUO PARA VOZ CON IA (TTS)
*(Para ElevenLabs, Clipchamp, Fliki, HeyGen, etc. Copia este bloque tal cual.)*

> Cordial saludo. Presentamos el ACA número 2 de Aprendizaje Automático de la Especialización en Inteligencia Artificial de la CUN, con el docente Hamilton Rivera Flor. Somos Juan Guillermo Angel, Luis Bernardo Correa y María Alejandra Manga. En el ACA 1 predijimos qué clientes de una tienda en línea dejarían de comprar, con un Random Forest que veía sólo tres números por cliente: recencia, frecuencia y gasto. Es como mirar una foto. En el ACA 2 la red neuronal ve la película completa: la historia de compras semana a semana.
>
> Usamos el dataset Online Retail de UCI: 541.909 líneas de factura entre diciembre de 2010 y diciembre de 2011. Todo se gestiona en una base de datos SQLite: la calidad, el análisis exploratorio y la limpieza se hacen con consultas y vistas SQL. Quitamos líneas sin cliente, duplicados, cancelaciones y precios no válidos. Después convertimos a cada uno de los 3.314 clientes en una secuencia de 40 semanas con siete canales: gasto, pedidos, unidades, productos, devoluciones, días sin comprar y si ya era cliente. En este mapa se ve la diferencia: quienes abandonan se van apagando antes del corte.
>
> Para que la comparación sea justa usamos la misma etiqueta y exactamente los mismos clientes de prueba del ACA 1, y lo verificamos en el código. Nada del futuro entra en las secuencias y cada cliente está en una sola partición. En entrenamiento aplicamos aumento de datos para series: escalado, ruido y recorte de la ventana. Comparamos tres redes: una GRU y una LSTM, que leen las semanas en orden y deciden qué recordar, y un Transformer temporal, en el que cada semana es un token y la autoatención relaciona todas las semanas entre sí. Cada red se entrenó con cinco semillas y elegimos la de mejor resultado en validación: la GRU.
>
> En los 663 clientes de prueba la GRU alcanza un área bajo la curva ROC de 0,73, frente a 0,72 del Random Forest. Las tres redes superan al modelo clásico por un margen parecido, pero el intervalo de confianza de la diferencia toca el cero, así que la mejora es pequeña y no concluyente: con compras tan esporádicas, buena parte de la señal ya estaba en recencia, frecuencia y gasto. Con la misma meta de detectar al menos el 80 por ciento de los abandonos, la red da 170 falsas alarmas frente a 188 del Random Forest. Donde más aporta la secuencia es en el segmento Recientes.
>
> Esta es la aplicación en Streamlit. Elegimos un cliente del conjunto de prueba: la red calcula su probabilidad de abandono y colorea cada semana según cuánto pesó en la decisión. Para eso borramos una semana a la vez y medimos cuánto cambia la predicción. Podemos cambiar de modelo y comparar con el Random Forest del ACA 1. En la pestaña de secuencia manual simulamos un cliente que se va apagando y otro estable, y vemos cómo cambia el riesgo. También procesamos un archivo CSV por lotes y descargamos el resultado. Cada predicción queda guardada en la base SQLite y se puede consultar con SQL desde la misma app.
>
> En conclusión, las redes recurrentes y de atención aprenden el ritmo de compra y la importancia de lo reciente, e igualan o superan ligeramente al Random Forest en el mismo problema. Su límite son los clientes con una sola compra. Lo valioso para el negocio es una herramienta que explica cada predicción semana a semana. El código, el notebook y la app están en los enlaces del póster. Muchas gracias.

---

## LISTA DE CONTROL ANTES DE ENTREGAR
- [ ] Duración entre 3:00 y 5:00.
- [ ] Se ve la app en vivo y el póster.
- [ ] Subir el video (YouTube «no listado» o Google Drive «cualquiera con el enlace») y regenerar póster y documentos:
  `.venv/Scripts/python entregables/generar_poster.py --video https://TU-VIDEO` y `.venv/Scripts/python entregables/generar_documentos.py`
