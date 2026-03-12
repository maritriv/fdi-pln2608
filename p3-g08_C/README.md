# Práctica 3 — Detectives de criptoglifos (PLNCG26)

- Asignatura: Procesamiento de Lenguaje Natural  
- Grupo: G08 (Carlota)
- Script entregado: `fdi-pln-2608-p3.py`

Trabajo realizado en WSL (Ubuntu) utilizando Python 3.12 y uv.

---

# 1. Introducción
El objetivo de esta práctica es analizar un sistema de codificación desconocido y desarrollar un programa capaz de **convertir texto español UTF-8 a un formato binario específico (PLNCG26) y viceversa**.

Para ello se proporcionaron varios archivos binarios:

- `principal.bin`
- `largo.bin`
- `noticia.bin`

Todos utilizan el mismo sistema de codificación, variando únicamente la longitud del texto. Una vez comprendido el sistema, se puede decodificar cualquiera de ellos.

---

# 2. Contexto del trabajo

El análisis inicial se realizó en el laboratorio de clase. Como mi compañera de grupo Marina no pudo asistir ese día, empecé la exploración del problema junto con **Tom**, el compañero francés de la asignatura (aunque pertenece a otro grupo).

Durante la sesión analizamos conjuntamente la estructura del archivo binario y realizamos las primeras hipótesis sobre su codificación. Una vez identificado el sistema general, cada uno continuó el desarrollo del script de forma independiente.

Por esta razón, la implementación final se encuentra en la carpeta:
```
p3-g08_C
```
donde la **C** hace referencia a *Carlota*, ya que el desarrollo final se completó de forma individual.

---

# 3. Exploración inicial del archivo binario

El primer paso consistió en inspeccionar el archivo `largo.bin` desde la terminal de Linux.

Para visualizar su contenido utilizamos:

```
xxd largo.bin
```

que muestra el archivo en **formato hexadecimal**.

También se puede visualizar en binario:

```
xxd -b largo.bin
```

En este punto observamos que el archivo no parecía contener texto ASCII directo, sino **bytes aparentemente arbitrarios**.

---

# 4. Análisis de frecuencia

Siguiendo una técnica clásica de criptoanálisis, realizamos un **análisis de frecuencia de bytes**.

Esto permitió detectar patrones importantes:

| Byte | Hipótesis |
|-----|-----------|
| 0x0B | espacio |
| 0x18 | letra frecuente |
| 0x14 | otra letra frecuente |

Comparando estas frecuencias con la distribución de letras del español, llegamos a la hipótesis de que el sistema utilizaba **un desplazamiento similar al cifrado César** (pero no era César).

---

# 5. Descubrimiento de los modificadores

Al reconstruir partes del texto aparecieron elementos como:
```
e[53]n
```

Esto indicaba que algunos bytes **no representaban caracteres**, sino **modificadores del carácter anterior**.

Finalmente identificamos varios modificadores:

| Byte | Función |
|----|------|
| 0x35 | convertir la letra anterior en mayúscula |
| 0x32 | añadir acento agudo |
| 0x33 | añadir diéresis |
| 0x34 | añadir tilde (ñ) |
| 0x0A | salto de línea |
| 0x0B | espacio |

Ejemplo:

```
e + [53] → E
n + [73] → ñ
```

---

# 6. Codificación del alfabeto

El sistema utiliza un desplazamiento basado en una constante denominada **MAGIC**:

```
MAGIC = 20
SHIFT = ord('a') - MAGIC
```

Esto genera la siguiente correspondencia:

| letra | byte |
|------|------|
| a | 20 |
| b | 21 |
| c | 22 |
| d | 23 |
| e | 24 |
| ... | ... |
| z | 45 |

Diagrama conceptual:

```
UTF-8 texto
│
▼
letra → desplazamiento → byte
│
▼
PLNCG26
```


Por ejemplo:
```
a → 20
h → 27
o → 34
l → 31
a → 20
```


Por tanto:

```
hola → [27 34 31 20]
```

---

# 7. Sistema de modificadores

El sistema representa caracteres acentuados mediante **dos bytes**:
```
letra_base + modificador
```

Ejemplos:
```
á = a + ACUTE
ñ = n + TILDE
ü = u + UMLAUT
```


Representación conceptual:
```
á

 a  +  ACUTE
 │      │
 ▼      ▼
[20] + [32]
```

Este enfoque separa la **información de la letra** de la **información diacrítica**.

---

# 8. Truco conceptual de la práctica

Un aspecto interesante del sistema PLNCG26 es que **no utiliza directamente Unicode** para representar caracteres acentuados.

En su lugar usa lo que mencionamos antes de:
```
letra_base + modificador
```

Este diseño tiene varias ventajas:

- evita caracteres multibyte
- simplifica el procesamiento a nivel de bytes
- separa letra y diacrítico

Este tipo de representación recuerda a ciertos sistemas históricos de codificación de texto y a estrategias utilizadas en procesamiento lingüístico.

---

# 9. Implementación del script

El archivo entregado es:
```
fdi-pln-2608-p3.py
```

El script se ejecuta mediante **uv** y utiliza la biblioteca `typer` para la interfaz de línea de comandos.

---

# 10. Comandos implementados

## encode

Convierte texto UTF-8 a formato PLNCG26.

Ejemplo:

```
uv run fdi-pln-2608-p3.py encode texto.txt > salida.bin
```

Flujo:
```
texto UTF-8
│
▼
encode
│
▼
PLNCG26 (bytes)
```

Nota: `texto.txt` es un archivo de prueba sencillo creado en el momento para comprobar la buena ejecución del programa. El contenido era: 

"Hola mundo
Esto es una prueba"

---

## decode

Convierte texto PLNCG26 a texto UTF-8.

Ejemplo:

```
uv run fdi-pln-2608-p3.py decode largo.bin
```

En el caso de `largo.bin`, el resultado corresponde al inicio de **Don Quijote de la Mancha**.

---

## detect (opcional)

Calcula la probabilidad de que un fichero esté codificado en PLNCG26.

Ejemplo:
```
uv run fdi-pln-2608-p3.py detect largo.bin
```

Salida:
```
Probabilidad PLNCG26: 100.00%
```

---

# 11. Dependencias

El script utiliza únicamente:
```
typer
```

y se ejecuta como **script uv auto-contenible**:

```
#!/usr/bin/env -S uv run
```


No se utilizan dependencias externas adicionales.

---

# 12. Entorno de desarrollo

El desarrollo se realizó en:

- WSL (Ubuntu)
- Python 3.12
- gestor de dependencias `uv`

Estructura del proyecto:

```
fdi-pln2608
├─ p1-g08
├─ p2-g08
├─ p3-g08_C
│ └─ fdi-pln-2608-p3.py
├─ pyproject.toml
└─ uv.lock
```


---

# 13. Conclusión

La práctica ilustra cómo un sistema de codificación aparentemente complejo puede entenderse mediante:

- análisis de bytes
- análisis de frecuencia
- reconstrucción incremental del sistema

Una vez identificado el desplazamiento del alfabeto y los modificadores, resulta posible implementar un conversor completo entre texto UTF-8 y el formato binario PLNCG26.




