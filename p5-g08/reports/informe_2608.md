# Informe P5

## 1. Introducción

En esta práctica se ha desarrollado un modelo de lenguaje basado en la arquitectura Transformer y, a partir de él, un sistema de reconocimiento de entidades nombradas (NER) utilizando un esquema BIO simplificado.

El objetivo principal del proyecto no era competir con modelos grandes ni obtener resultados comparables a sistemas industriales actuales, sino comprender e implementar manualmente los componentes fundamentales que aparecen en modelos modernos de procesamiento del lenguaje natural.

A lo largo de la práctica se trabajó principalmente con:

- tokenización BPE;
- mecanismos de auto-atención;
- entrenamiento causal autoregresivo;
- transferencia de pesos entre tareas;
- y adaptación de un backbone Transformer a reconocimiento de entidades.

El proyecto final se entrega como un ejecutable único (`fdi-pln-2608-p5`) capaz de:

- entrenar modelos;
- generar texto;
- preparar datasets NER;
- evaluar métricas;
- e inferir entidades sobre texto arbitrario.

Todos los checkpoints generados son autocontenidos e incluyen tanto los pesos del modelo como la configuración necesaria para reutilizarlos posteriormente.

Además, uno de los objetivos importantes del proyecto fue mantener todo el pipeline completamente reproducible y ejecutable únicamente desde CLI.

---

## 2. Corpus utilizado

Para el entrenamiento del modelo causal se utilizaron fragmentos literarios de Lewis Carroll incluidos en:

- `resources/alice_in_wonderland.txt`
- `resources/looking_glass.txt`

Se decidió trabajar con estos textos porque ofrecen:

- un vocabulario relativamente variado;
- abundancia de diálogos;
- personajes recurrentes;
- y suficientes referencias espaciales como para reutilizar posteriormente el corpus en una tarea NER.

Aunque el tamaño total del dataset es relativamente pequeño comparado con corpus modernos, resultaba suficiente para implementar y validar todo el pipeline Transformer de la práctica.

Para la parte de reconocimiento de entidades se reutilizó el corpus anotado manualmente durante la preentrega del grupo 1 de etiquetado.

El fichero final utilizado fue:

```text
data/ner/merged.json
```

Este corpus fusionado se convierte automáticamente al formato CoNLL mediante:

```bash
uv run fdi-pln-2608-p5 prepare-ner-data --input ../pre-entrega_2601/merged.json --output data/ner/final.conll
```

Durante esta conversión se eliminan espacios y tokens irrelevantes para entrenamiento, manteniendo únicamente las secuencias útiles para el modelo NER.

Las estadísticas finales del corpus fueron:

| Medida NER | Valor |
| --- | ---: |
| Frases fusionadas | 59 |
| Tokens en la anotación original | 5750 |
| Tokens escritos en CoNLL | 3263 |
| Tokens blancos omitidos | 2487 |
| Tokens de entidad | 194 |
| Kappa de Cohen medio | 0.835 |
| Acuerdo medio | 98% |

Los resultados de acuerdo inter-anotador obtenidos en la preentrega fueron bastante altos. Un valor de κ superior a 0.8 suele considerarse un acuerdo fuerte, lo que indica que las anotaciones fueron relativamente consistentes entre participantes.

Las etiquetas originales de la preentrega se transformaron posteriormente al esquema BIO estándar utilizado por el modelo:

| Preentrega | BIO | Significado |
| --- | --- | --- |
| `o` | `O` | Fuera de entidad |
| `pi` | `B-PER` | Inicio de persona/personaje |
| `pc` | `I-PER` | Continuación de persona/personaje |
| `li` | `B-LOC` | Inicio de localización |
| `lc` | `I-LOC` | Continuación de localización |

---

## 3. Tokenización BPE

El modelo utiliza una tokenización basada en BPE (*Byte Pair Encoding*). En lugar de trabajar únicamente con palabras completas o caracteres individuales, el sistema aprende automáticamente fragmentos frecuentes del corpus.

El procedimiento seguido fue relativamente sencillo:

1. comenzar desde caracteres individuales;
2. contar pares consecutivos frecuentes;
3. fusionar progresivamente los pares más comunes;
4. y reutilizar esas fusiones durante la codificación.

Este enfoque permite representar palabras frecuentes mediante menos tokens y, al mismo tiempo, seguir siendo capaz de tokenizar palabras desconocidas utilizando fragmentos más pequeños.

Se decidió implementar BPE porque resulta mucho más flexible que una tokenización puramente por palabras, especialmente en corpus pequeños donde aparecen:

- nombres raros;
- deformaciones;
- puntuación irregular;
- o vocabulario poco frecuente.

Ejemplo real obtenido con el modelo entrenado:

```bash
uv run fdi-pln-2608-p5 analyze-bpe --weights checkpoints/p5_causal_2608.pth --text "Alice went to Wonderland"
```

| Texto | Caracteres | Tokens | Caracteres/token | Segmentación |
| --- | ---: | ---: | ---: | --- |
| `Alice went to Wonderland` | 24 | 11 | 2.18 | `\n \| lic \| e \| went \| to \| \n \| on \| d \| er \| l \| and` |

En este ejemplo puede verse una de las limitaciones del entrenamiento con corpus pequeño: algunas palabras se fragmentan de forma poco natural, especialmente aquellas menos frecuentes o con mayúsculas.

Aun así, el comportamiento general del tokenizador fue razonablemente bueno para el tamaño reducido del dataset y permitió mantener el vocabulario controlado.

---

## 4. Atención escalada

El núcleo del Transformer implementado en la práctica es el mecanismo de auto-atención.

Cada token genera tres representaciones distintas:

- queries;
- keys;
- values.

A partir de ellas se calcula la compatibilidad entre posiciones usando el producto:

```text
Q @ K.T
```

Sin embargo, cuando la dimensión interna aumenta, estos productos pueden crecer demasiado y provocar saturación en la softmax. Para evitarlo se aplica el factor:

```text
1 / sqrt(head_dim)
```

Este escalado estabiliza el entrenamiento y ayuda a mantener gradientes útiles.

Durante las pruebas se observó que pequeñas variaciones en esta parte afectaban bastante a la estabilidad del modelo, especialmente entrenando únicamente en CPU y con datasets relativamente pequeños.

Para generación de texto se utiliza máscara causal, impidiendo que el modelo vea tokens futuros. En cambio, para NER se utiliza atención bidireccional, ya que la clasificación de una palabra depende tanto del contexto izquierdo como del derecho.

---

## 5. Modelo causal

El modelo causal implementado sigue una arquitectura Transformer relativamente pequeña, diseñada para poder entrenarse en tiempos razonables dentro de las limitaciones de la práctica.

El modelo combina:

- embeddings de token;
- embeddings posicionales;
- bloques Transformer pre-norm;
- multi-head attention;
- capas feed-forward con GELU;
- dropout;
- y una cabeza final de lenguaje.

Durante el entrenamiento causal, el modelo recibe una secuencia y aprende a predecir el siguiente token.

La configuración utilizada finalmente fue:

| Parámetro | Valor |
| --- | ---: |
| `vocab_size` | 300 |
| `context_size` | 128 |
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `dropout` | 0.1 |
| `epochs` | 5 |
| `batch_size` | 64 |
| `lr` | 0.0003 |

Inicialmente se probaron configuraciones más grandes, pero el entrenamiento se volvía demasiado lento y los resultados no mejoraban significativamente debido al tamaño reducido del corpus.

Ejemplo de generación:

```bash
uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was" --max-new-tokens 80 --top-k 20
```

Salida observada:

```text
Alice was a worstanding large carriled it, and had put it: she stood out of
we frightened so day.

“suppose it might too say,” said the duchess, “how walk
about their day
```

Aunque el modelo todavía produce errores y palabras deformadas, sí consigue capturar parcialmente el estilo narrativo del corpus: aparecen diálogos, puntuación similar y nombres propios característicos del texto original.

Teniendo en cuenta que el entrenamiento se realizó sobre un corpus pequeño y sin recursos GPU, el comportamiento puede considerarse razonablemente satisfactorio.

---

## 6. Modelo NER

La parte NER reutiliza el backbone Transformer aprendido durante el entrenamiento causal.

En lugar de entrenar un modelo completamente nuevo desde cero, se reaprovechan:

- embeddings;
- posiciones;
- y bloques Transformer.

Únicamente se sustituye la cabeza final por una capa lineal de clasificación BIO.

Esta decisión se tomó porque incluso modelos pequeños suelen beneficiarse bastante de haber aprendido previamente ciertas regularidades del lenguaje.

Uno de los problemas principales encontrados fue el fuerte desbalance del corpus: la inmensa mayoría de tokens pertenecen a la clase `O`.

Para reducir este efecto se añadieron pesos de clase suaves durante el entrenamiento. Esto permitió mejorar bastante el recall de entidades, aunque introdujo también más falsos positivos.

Entrenamiento y evaluación:

```bash
uv run fdi-pln-2608-p5 train-ner --data data/ner/final.conll --causal-weights checkpoints/p5_causal_2608.pth --output checkpoints/p5_ner_2608.pth

uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll
```

Resultados obtenidos:

| Modelo | Token acc | Token P | Token R | Token F1 | Entity P | Entity R | Entity F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NER 2608 | 0.8881 | 0.2666 | 0.7053 | 0.3869 | 0.1132 | 0.5909 | 0.1901 |

Desglose por tipo:

| Tipo | Precision | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| LOC | 0.1027 | 0.8824 | 0.1840 | 17 |
| PER | 0.1168 | 0.5376 | 0.1919 | 93 |

La accuracy por token es relativamente alta debido al predominio de la clase `O`. Sin embargo, el F1 *entity-level* refleja mucho mejor la dificultad real de la tarea.

En general, el modelo consigue detectar bastantes entidades relevantes, especialmente nombres frecuentes de personajes. No obstante, todavía aparecen errores claros en:

- delimitación exacta de entidades;
- confusión entre PER y LOC;
- y falsos positivos en palabras comunes.

Ejemplo real:

```bash
uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt
```

| Entidad | Tipo predicho |
| --- | --- |
| the | PER |
| Queen | LOC |
| Wonderland | PER |
| The | PER |
| ran | LOC |
| through | PER |
| Hatter | PER |
| March | PER |
| Hare | LOC |
| drank | LOC |
| Alice | PER |
| garden | LOC |
| hall | PER |

Aunque aparecen errores evidentes (`the`, `ran`), también se observan aciertos razonables en entidades importantes del corpus como `Alice`, `Hatter` o `garden`.

Dado el tamaño del dataset y la simplicidad del modelo, los resultados obtenidos eran relativamente esperables.

---

## 7. Experimentos de generación

También se realizaron distintas pruebas modificando temperatura y `top-k` para analizar cómo afectaban a la generación.

```bash
uv run fdi-pln-2608-p5 experiment-generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was" --out reports/generation_experiments.md --max-new-tokens 40
```

Resumen cualitativo:

| Temperature | Top-k | Resultado observado |
| ---: | ---: | --- |
| 0.5 | 10 | Muy conservador y repetitivo, pero más coherente. |
| 0.5 | 20 | Algo más variado, aunque todavía rígido. |
| 0.5 | 50 | Empiezan a aparecer deformaciones frecuentes. |
| 0.8 | 10 | Buen equilibrio entre variedad y coherencia. |
| 0.8 | 20 | Resultados más creativos pero menos estables. |
| 0.8 | 50 | Más ruido y pérdida de control. |
| 1.2 | 10 | Generación bastante impredecible. |
| 1.2 | 20 | Saltos semánticos frecuentes. |
| 1.2 | 50 | Máxima diversidad pero muy poca coherencia. |

Durante las pruebas se observó claramente el comportamiento típico de modelos pequeños:

- temperaturas bajas producen texto más seguro pero repetitivo;
- temperaturas altas aumentan creatividad;
- pero también amplifican errores y palabras inexistentes.

El mejor equilibrio general apareció alrededor de `temperature=0.8` y `top-k=10/20`.

---

## 8. Reproducibilidad y CLI

Uno de los objetivos importantes de la práctica era mantener el proyecto completamente reproducible.

Todos los pasos principales pueden ejecutarse desde CLI:

```bash
uv sync
uv format --check
uv run fdi-pln-2608-p5 --help

uv run fdi-pln-2608-p5 generate --weights checkpoints/p5_causal_2608.pth --prompt "Alice was"

uv run fdi-pln-2608-p5 prepare-ner-data --input ../pre-entrega_2601/merged.json --output data/ner/final.conll

uv run fdi-pln-2608-p5 train-ner --data data/ner/final.conll --causal-weights checkpoints/p5_causal_2608.pth --output checkpoints/p5_ner_2608.pth

uv run fdi-pln-2608-p5 eval-ner --weights checkpoints/p5_ner_2608.pth --data data/ner/final.conll

uv run fdi-pln-2608-p5 ner --weights checkpoints/p5_ner_2608.pth --file examples/text.txt

uv build
```

El wheel final generado fue:

```text
dist/fdi_pln_2608_p5-1.0-py3-none-any.whl
```

---

## 9. Limitaciones encontradas

La principal limitación del proyecto fue el tamaño reducido del corpus.

En generación de texto esto provoca:

- vocabulario limitado;
- errores ortográficos;
- repeticiones;
- y pérdida relativamente rápida de coherencia.

En NER, el problema principal fue el fuerte desbalance entre entidades y tokens `O`. Además, el corpus anotado contiene relativamente pocas frases y pocos ejemplos de localizaciones.

También se observó que la tokenización BPE puede fragmentar palabras de manera poco intuitiva cuando aparecen mayúsculas o términos raros.

Por último, aunque los pesos de clase ayudaron a mejorar el recall, también incrementaron notablemente el número de falsos positivos.

En general, muchas de estas limitaciones son esperables cuando se trabaja con datasets pequeños y modelos entrenados únicamente en CPU.

---

## 10. Conclusiones

En conjunto, la práctica permitió implementar un pipeline completo y funcional de procesamiento del lenguaje natural utilizando únicamente componentes desarrollados manualmente.

A lo largo del proyecto se trabajó con:

- tokenización BPE;
- Transformers;
- entrenamiento causal;
- generación autoregresiva;
- transferencia de pesos;
- y reconocimiento de entidades.

Aunque los modelos obtenidos son pequeños y todavía presentan limitaciones claras, el sistema completo funciona de forma coherente y reproducible.

Además, una de las partes más interesantes del proyecto fue integrar la preentrega NER realizada manualmente por el grupo y reutilizarla posteriormente para entrenar un modelo real de reconocimiento de entidades.

El resultado final no pretende competir con modelos actuales, pero sí permite entender de forma práctica cómo se construyen y entrenan este tipo de arquitecturas, así como las dificultades reales que aparecen cuando se trabaja con corpus pequeños y recursos limitados.

En general, la práctica resultó especialmente útil para comprender mejor tanto la arquitectura Transformer como los problemas reales que aparecen durante entrenamiento, tokenización, evaluación y ajuste de modelos NLP relativamente pequeños.