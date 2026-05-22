# Informe P5 - Grupo 8

## 1. Introducción

En esta práctica hemos construido un pequeño sistema de procesamiento del lenguaje natural basado en Transformers. El trabajo tiene dos partes conectadas entre sí, un modelo de lenguaje causal capaz de generar textodespués y un modelo de reconocimiento de entidades nombradas (NER) que reutiliza el conocimiento aprendido por el modelo causal.

La idea principal no era entrenar un modelo grande ni competir con sistemas actuales, sino entender de forma práctica qué piezas hacen falta para construir un pipeline moderno de NLP: tokenización, embeddings, atención, bloques Transformer, entrenamiento autoregresivo, transferencia de pesos y evaluación de una tarea de clasificación secuencial.

Por eso el proyecto se ha planteado como una implementación pequeña pero completa. El sistema permite:

- entrenar el modelo causal
- generar texto a partir de un prompt
- preparar el corpus NER en formato CoNLL/BIO
- entrenar el modelo NER reutilizando el backbone causal
- evaluar métricas token-level y entity-level
- listar entidades nombradas encontradas en un fichero de texto

Todo queda integrado en un único ejecutable:

```bash
uv run fdi-pln-2608-p5
```

Además del modo por comandos, se ha añadido un menú interactivo con Typer y Rich para que la práctica sea más sencilla e intuitiva.

---

## 2. Organización del proyecto

La estructura general del proyecto es la siguiente:

```text
src/fdi_pln_2608_p5/
|-- cli.py                 # entry point del wheel
|-- main.py                # arranque mínimo de la app Typer
|-- checkpoint.py          # guardado y carga de checkpoints
|-- tokenizer.py           # tokenizador BPE propio
|-- utils.py               # semillas, dispositivo y utilidades 
|
|-- cli_app/               # interfaz Typer + Rich
|-- model/                 # atención, Transformer y modelo NER
|-- data/                  # dataset causal y conversión NER
|-- training/              # entrenamiento causal y NER
|-- evaluation/            # evaluación NER y análisis BPE
`-- generation/            # generación de texto e inferencia NER
```

Esta organización separa claramente las distintas responsabilidades del proyecto: la interfaz de línea de comandos, la lógica de entrenamiento, la generación de texto, la evaluación y los componentes internos del modelo. La idea era mantener una arquitectura modular donde cada parte pudiera evolucionar de forma relativamente independiente.

Además, esta separación facilita bastante la reutilización de componentes, las pruebas individuales de cada módulo y la trazabilidad del pipeline completo, desde la preparación de datos hasta la evaluación final de modelos.


---

## 3. Corpus utilizado

Para el modelo de lenguaje se utilizaron los textos proporcionados para la práctica:

- `resources/alice_in_wonderland.txt`
- `resources/looking_glass.txt`

Ambos textos pertenecen al universo literario de Lewis Carroll y son adecuados para esta práctica porque contienen diálogos, personajes recurrentes, nombres propios frecuentes y referencias a lugares. Esto permite que el modelo causal aprenda ciertos patrones de estilo narrativo y, al mismo tiempo, que la tarea NER tenga sentido dentro del mismo dominio textual.

El corpus no es especialmente grande, y eso condiciona los resultados obtenidos. Aun así, para una implementación didáctica entrenada en CPU resulta suficiente ya que permite entrenar el modelo, generar ejemplos, analizar errores reales y comprobar el funcionamiento completo del pipeline sin que el coste de entrenamiento sea inabordable.

Para NER se reutilizó el corpus anotado durante la preentrega. En la práctica final ya está integrado en:

```text
data/ner/merged.json
data/ner/final.conll

---

## 4. Tokenización BPE

El modelo no trabaja directamente con palabras completas. Se implementó un tokenizador BPE (*Byte Pair Encoding*) propio, entrenado sobre el corpus literario.

La razón de usar BPE es que ofrece un equilibrio entre caracteres y palabras. Si se tokeniza solo por caracteres, las secuencias se vuelven largas y al modelo le cuesta aprender unidades con significado. Si se tokeniza solo por palabras, cualquier palabra no vista se convierte en un problema. BPE permite representar palabras frecuentes con pocos tokens y palabras raras mediante fragmentos más pequeños.

El procedimiento seguido fue:

1. empezar con caracteres individuales;
2. contar pares consecutivos frecuentes;
3. fusionar los pares más comunes;
4. repetir hasta alcanzar el tamaño de vocabulario deseado;
5. reutilizar esas fusiones para codificar nuevos textos.

Un ejemplo real de análisis BPE:

```bash
uv run fdi-pln-2608-p5 analyze-bpe \
  --weights checkpoints/p5_causal_2608.pth \
  --text "Alice went to Wonderland"
```

Resultado resumido:

| Texto | Caracteres | Tokens | Ratio caracteres/token |
| --- | ---: | ---: | ---: |
| `Alice went to Wonderland` | 24 | 11 | 2.18 |

La segmentación obtenida no siempre es intuitiva. Por ejemplo, algunas palabras con mayúsculas o poco frecuentes se rompen en piezas pequeñas. Esto es una limitación esperable al entrenar BPE con un corpus reducido. Aun así, el tokenizador cumple su función: mantiene un vocabulario pequeño y permite codificar cualquier texto de entrada.

---

## 5. Arquitectura Transformer

El núcleo del proyecto es un Transformer pequeño implementado manualmente. La arquitectura incluye:

- embeddings de token;
- embeddings posicionales;
- bloques Transformer pre-norm;
- atención multi-cabezal;
- capas feed-forward con GELU;
- dropout;
- una cabeza final de lenguaje para generación;
- una cabeza BIO para NER.

La atención es el componente más importante. Cada posición de la secuencia genera tres representaciones: queries, keys y values. Con ellas se calcula qué tokens deben atender a qué otros tokens.

Para estabilizar el cálculo se usa atención escalada:

```text
softmax((Q @ K.T) / sqrt(head_dim)) @ V
```

El escalado por `sqrt(head_dim)` evita que los productos internos crezcan demasiado y saturen la softmax. Esta decisión es estándar en Transformers y resulta especialmente importante cuando se entrena con recursos limitados, porque ayuda a que el entrenamiento sea más estable.

El mismo backbone se usa de dos formas:

- en generación, con máscara causal, para impedir que el modelo vea tokens futuros;
- en NER, sin máscara causal, porque para clasificar una palabra interesa usar contexto izquierdo y derecho.

Esta diferencia es importante: la arquitectura base es la misma, pero la tarea cambia la forma de usar la atención.

---

## 6. Modelo causal de generación

El primer modelo entrenado fue el modelo de lenguaje causal. Su objetivo es predecir el siguiente token a partir de los tokens anteriores.

La configuración final fue pequeña para que el entrenamiento fuera viable en CPU:

| Parámetro | Valor |
| --- | ---: |
| `context_size` | 128 |
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `dropout` | 0.1 |
| `lr` | 0.0003 |

La decisión de mantener un modelo pequeño fue deliberada. Un modelo mayor podría parecer más ambicioso, pero con un corpus reducido y entrenamiento en CPU no necesariamente iba a generalizar mejor. En las pruebas, aumentar mucho la arquitectura hacía el entrenamiento más lento sin una mejora clara en la calidad de salida.

El checkpoint final es:

```text
checkpoints/p5_causal_2608.pth
```

Ejemplo de uso:

```bash
uv run fdi-pln-2608-p5 generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --max-new-tokens 80 \
  --top-k 20
```

Las generaciones capturan parcialmente el estilo del corpus: aparecen diálogos, puntuación literaria y nombres propios característicos. Sin embargo, también aparecen palabras deformadas, repeticiones y pérdida de coherencia en secuencias largas.

Esto no se interpreta como un fallo total del modelo, sino como una consecuencia razonable del tamaño del corpus y del tamaño del Transformer. El sistema demuestra que el pipeline funciona, aunque no tenga capacidad suficiente para producir texto largo de alta calidad.

---

## 7. Modelo NER

La segunda parte de la práctica consiste en adaptar el modelo causal a reconocimiento de entidades nombradas.

En vez de entrenar un modelo NER completamente nuevo, se reutiliza el backbone Transformer aprendido durante el entrenamiento causal. Se conservan:

- embeddings;
- posiciones;
- bloques Transformer.

La cabeza de lenguaje se sustituye por una capa lineal que predice etiquetas BIO por token.

Esta decisión tiene sentido porque el modelo causal ya ha aprendido cierta información sobre el corpus: patrones de caracteres, nombres frecuentes, estilo y contexto local. Aunque el modelo sea pequeño, reutilizar ese conocimiento es mejor que empezar NER desde cero.

El checkpoint final es:

```text
checkpoints/p5_ner_2608.pth
```

Entrenamiento:

```bash
uv run fdi-pln-2608-p5 train-ner \
  --data data/ner/final.conll \
  --causal-weights checkpoints/p5_causal_2608.pth \
  --output checkpoints/p5_ner_2608.pth
```

Evaluación:

```bash
uv run fdi-pln-2608-p5 eval-ner \
  --weights checkpoints/p5_ner_2608.pth \
  --data data/ner/final.conll
```

Resultados:

| Métrica | Valor |
| --- | ---: |
| Token accuracy | 0.8881 |
| Token precision | 0.2666 |
| Token recall | 0.7053 |
| Token F1 | 0.3869 |
| Entity precision | 0.1132 |
| Entity recall | 0.5909 |
| Entity F1 | 0.1901 |

Desglose por tipo:

| Tipo | Precision | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| LOC | 0.1027 | 0.8824 | 0.1840 | 17 |
| PER | 0.1168 | 0.5376 | 0.1919 | 93 |

La accuracy por token es alta porque la mayoría de tokens son `O`. Por eso no basta con mirar accuracy. Las métricas entity-level son más honestas: evalúan si el modelo detecta entidades completas y no solo etiquetas aisladas.

El modelo tiene un recall relativamente alto, pero baja precisión. Esto significa que encuentra muchas entidades, pero también marca demasiadas palabras que no deberían ser entidades. En la práctica esto se ve claramente:

```bash
uv run fdi-pln-2608-p5 ner \
  --weights checkpoints/p5_ner_2608.pth \
  --file examples/text.txt
```

Ejemplos de entidades detectadas:

| Entidad | Tipo predicho |
| --- | --- |
| Queen | LOC |
| Wonderland | PER |
| Hatter | PER |
| March | PER |
| Hare | LOC |
| Alice | PER |
| garden | LOC |

También aparecen falsos positivos como `the`, `ran` o `through`. Esto se debe principalmente al desbalance del dataset y a que hay pocas entidades anotadas. Se añadieron pesos de clase para que el modelo no ignorase todas las entidades, pero eso aumentó los falsos positivos. Es una decisión razonable si se quiere que el modelo al menos aprenda a detectar entidades, aunque la precisión final quede limitada.

---

## 8. Experimentos de generación

Además del entrenamiento principal, se realizaron pruebas variando `temperature` y `top-k`.

El objetivo no era encontrar una combinación perfecta, sino observar cómo cambia el comportamiento de un modelo pequeño según el muestreo.

Comando usado:

```bash
uv run fdi-pln-2608-p5 experiment-generate \
  --weights checkpoints/p5_causal_2608.pth \
  --prompt "Alice was" \
  --out reports/generation_experiments.md \
  --max-new-tokens 40
```

Resumen cualitativo:

| Temperature | Top-k | Comportamiento observado |
| ---: | ---: | --- |
| 0.5 | 10 | Más conservador, menos variado y algo repetitivo. |
| 0.8 | 10/20 | Mejor equilibrio entre variedad y control. |
| 1.2 | 20/50 | Más creativo, pero también más inestable. |

La conclusión principal fue que, en un modelo pequeño, subir mucho la temperatura aumenta rápidamente el ruido. El modelo empieza a explorar tokens menos probables, pero como no tiene una representación lingüística muy robusta, aparecen más deformaciones y saltos de coherencia.

Por eso, para ejemplos de entrega, los valores más razonables son `temperature=0.8` y `top-k` entre 10 y 20.

---

## 9. Interfaz de línea de comandos

La práctica se entrega como un CLI único. Esto es importante porque el enunciado pide poder ejecutar la práctica mediante un comando instalado desde el wheel.

El comando principal es:

```bash
uv run fdi-pln-2608-p5
```

Al ejecutarlo sin argumentos se abre un menú interactivo. Este menú permite probar las partes más importantes sin memorizar todos los comandos:

- generación de texto;
- detección de entidades NER;
- evaluación del modelo NER;
- análisis de tokenización BPE;
- ejemplos de comandos directos.

También se conservan los comandos reproducibles:

```bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
```

Se decidió usar Typer y Rich porque permiten una interfaz más clara sin añadir dependencias no permitidas. El CLI final no es solo una colección de scripts: actúa como una pequeña herramienta de laboratorio para entrenar, evaluar y probar el sistema.

---

## 10. Reproducibilidad

Los pasos principales para reproducir la práctica son:

```bash
uv sync
uv format --check
uv run fdi-pln-2608-p5 --help
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"
uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
uv build
```

El wheel generado es:

```text
dist/fdi_pln_2608_p5-1.0-py3-none-any.whl
```

Los checkpoints son autocontenidos: guardan pesos, configuración, métricas y tokenizador. Esto evita depender de configuraciones externas y facilita que el profesor pueda cargar directamente los modelos entregados.

---

## 11. Limitaciones

La limitación más importante es el tamaño del corpus. El modelo causal aprende ciertos patrones de estilo, pero no tiene datos suficientes para generar texto largo y coherente de forma consistente.

En generación aparecen:

- palabras deformadas;
- repeticiones;
- frases que empiezan con estilo literario pero pierden sentido;
- dependencia fuerte de `temperature` y `top-k`.

En NER, la principal dificultad es el desbalance. Hay muchos tokens `O` y pocos tokens de entidad. Además, el número de ejemplos de localizaciones es especialmente bajo. Esto hace que el modelo tenga problemas para delimitar entidades correctamente y para distinguir entre `PER` y `LOC`.

Otra limitación es que el tokenizador BPE se entrena sobre un corpus pequeño. Por eso algunas palabras se segmentan de manera poco natural, especialmente nombres propios o palabras con mayúsculas.

Estas limitaciones son importantes, pero también forman parte del aprendizaje de la práctica: muestran qué ocurre cuando se intenta construir un sistema completo con pocos datos y recursos limitados.

---

## 12. Conclusiones

La práctica ha permitido construir un pipeline completo de NLP desde cero, pasando por todas las fases importantes: tokenización, modelo causal, generación, adaptación a NER, evaluación y empaquetado como herramienta ejecutable.

La parte más valiosa del trabajo ha sido comprobar cómo se conectan las piezas. El tokenizador BPE condiciona la entrada del Transformer; el Transformer aprende representaciones durante el entrenamiento causal; esas representaciones se reutilizan después para NER; y las métricas finales muestran hasta qué punto esa transferencia funciona en un escenario pequeño.

Los resultados no son perfectos, pero sí son coherentes con el planteamiento. El modelo causal genera texto con rasgos reconocibles del corpus, aunque con errores. El modelo NER detecta bastantes entidades, pero con muchos falsos positivos. En ambos casos, los fallos se explican por el tamaño del dataset, el desbalance y la capacidad limitada del modelo.

Desde el punto de vista de ingeniería, el proyecto final queda empaquetado en un CLI reproducible y en un wheel instalable. Esto era importante para que la entrega no fuese solo código suelto, sino una herramienta completa que se puede ejecutar, probar y evaluar.

En conclusión, el sistema cumple el objetivo principal de la práctica: implementar y entender un mini Transformer funcional, aplicarlo a generación de texto y reutilizarlo para una tarea NER real construida a partir del corpus anotado en la preentrega. Aunque el modelo es pequeño, el recorrido completo refleja bien las decisiones y dificultades que aparecen en un proyecto real de procesamiento del lenguaje natural.

---

## 13. Integrantes

- Marina Triviño de las Heras
- Carlota Salazar Martín
