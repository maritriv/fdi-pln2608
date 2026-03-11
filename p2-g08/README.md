# Práctica 2 - PLN

Grupo: G08

Estudiantes:
- Carlota Salzar Martín
- Marina Triviño de las Heras


## 1. Proceso seguido

NOTA: Cabe destacar que la práctica la hemos hecho por separado, pero que integramos ahora ambas partes para hacer una única entrega.

Para la realización de esta práctica, hemos seguido una metodología de síntesis por concatenación manual:

1. Grabación: Grabamos cada una los pangramas originales en español e inglés (también francés y chino, entre otros) en formato MP3.

2. Segmentación: Utilizamos Praat para visualizar el espectrograma y la onda, identificando los límites de cada fonema.

3. Extracción: Exportamos cada fonema individualmente buscando, en la medida de lo posible, estados estables de la señal.

4. Ensamblaje: Reconstruimos los nuevos pangramas utilizando Audacity (Carlota) o CapCut (Marina), intentando aplicar técnicas de crossfading y normalización para minimizar los ruidos y suavizar la transición entre fonemas.


## 2. Problemas encontrados y análisis técnico

A pesar de la precisión en el corte, el resultado sonoro presenta una falta de naturalidad significativa ("voz robótica" o "discontinua"). Tras analizar el proceso, hemos identificado las causas raíz:

- Ausencia de Coarticulación: Al extraer fonemas aislados, perdemos la transición natural de los órganos articulatorios (lengua, labios, velo del paladar). En el habla humana, un fonema se ve influido por el anterior y el posterior: cuando hablamos, los órgnaos articulatorios se están preparando para el siguiente sonido mientras emiten el actual. Al eliminar estas transiciones (especialmente visibles en el segundo formante F2), el cerebro detecta saltos antinaturales.

- Pérdida de la Prosodia y Curva Entonativa: El pangrama sintético carece de una línea melódica coherente. Cada fonema conserva el tono de su contexto original, lo que genera una entonación caótica que dificulta la comprensión sintáctica de la frase.

- Discontinuidad Espectral (Efecto Gestalt): Filosóficamente, el habla es una Gestalt donde el todo es más que la suma de sus partes. Al romper la señal en unidades discretas, rompemos la integridad del mensaje. Las pequeñas diferencias en el ruido de fondo y la intensidad entre clips rompen la ilusión de una voz única y continua.

## 3. Notas sobre el Inventario Fonético y Grabación

- Variación en el Tempo y la Articulación: Inicialmente, las grabaciones originales presentaban un tempo de habla demasiado rápido. Esto provocaba una hipoarticulación (relajación de los movimientos articulatorios), haciendo que los límites entre fonemas fueran difusos. Marina optó por regrabar los pangramas con una dicción enfática y lenta, facilitando la segmentación en Praat al obtener estados estables más largos y definidos.

- Aproximaciones Fonológicas (Sustituciones): Ante la ausencia del fonema africado /ks/ (la "x" de expensive) en el pangrama de origen, Carlota utilizó una estrategia de aproximación por rasgos compartidos, empleando la fricativa alveolar sorda /s/. Esto demuestra que la inteligibilidad sintética depende a veces de la jerarquía de rasgos: preferimos un sonido similar en punto de articulación aunque no sea idéntico. Pero no hemos usado otros idiomas para intentar extrear estos sonidos (por complejidad).

- Consonantes oclusivas: Ambas tuvimos dificultades ajustando los límites fonéticos en las consonantes oclusivas. Creemos que esto se debe a que la percepción de estas oclusivas dependen casi totalmente de las transiciones de los formantes de la vocal siguiente. Por ejemplo, la 'p' solo se indentifica por el sonido de la vocal que viene justo después, y no por el sonido de la explosión en sí. Entonces, al cortar la 'p' y separarla de su vocal original, le quitamos la información que el oído necesita para identificarla. Además, al reconstruir el nuevo pangrama, aunque pongamos el clip de la 'p' justo antes que el de la 'a', siguen siendo como dos piezas que no encajan. Hemos buscado la razón: el cerebro no está escuchando una 'p' seguida de una 'a', sino qeu está esperando ver cómo la 'p' deforma a la 'a'. Entonces va más allá que una constante + una vocal; esto es, la "psicoacústica".


## 4. Pequeña guía de tipos de sonido (y visualización en Praat)

Después de trabajar con los espectogramas en Praat, hemos llegado a estas conclusiones:

| Tipo de Sonido | Qué es | Cómo se ve en el espectrograma | Dificultad de extracción |
| :--- | :--- | :--- | :--- |
| **Vocales** | Aire sin obstáculos, cuerdas vocales vibrando. | Bandas horizontales oscuras (**formantes**). | **Fácil**: Son estables y musicales. |
| **Fricativas** (/s/, /f/, /j/) | Aire pasando por un canal estrecho (fricción). | Una mancha de "humo" o ruido en las frecuencias altas. | **Media**: Tienen mucha energía propia. |
| **Oclusivas** (/p/, /t/, /k/) | Cierre total y explosión de aire. | Un hueco blanco (silencio) y una línea vertical fina. | **Alta**: Sin la vocal que las sigue, apenas se entienden. |
| **Nasales** (/m/, /n/) | El aire sale por la nariz. | Parecidas a las vocales pero con menos energía (más claras). | **Media**: Se confunden fácilmente entre ellas. |


## 5. Estructura de archivos entregados

- ```/originales```: Grabaciones base de los integrantes Carlota y Marina.

- ```/sinteticos```: Frases reconstruidas (es.mp3, en.mp3).

- ```/data```: Material adicional (fonemas extraídos por Marina).

