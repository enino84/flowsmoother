# Suavizador EnKF-MC con precisión dictada por el flujo: plan de diseño

Versión v0.3, 5 de noviembre de 2026. Documento de rumbo: qué se construye, por qué, cómo se
mide y qué resultado nos haría abandonar la idea. Las secciones 1 a 12 son el diseño; la
sección 13 dice qué quedó implementado en el repositorio, qué cambió respecto al diseño y
qué se midió ya. Los primeros resultados están en `PILOTO.md`. Se apoya en `FLOW_DICTATED_PRECISION_full_notes.md`
(las notas) y en `SWESPHERE_REFERENCE_v1.md` (la referencia de swesphere 0.6.0).

---

## 1. Objetivo e hipótesis

Construir un suavizador de Kalman por ensambles con Cholesky modificado (EnKS-MC) en el que
el conjunto de predecesores de cada fila lo fije la dinámica y no un radio.

**Hipótesis.** En un modelo donde el error se advecta (aguas someras), la covarianza entre
tiempos es el propagador, y su patrón de dispersión es la trayectoria de las parcelas. Por
tanto:

1. Poner la estela (trayectoria hacia atrás del viento medio) en el bloque entre tiempos
   mejora el campo transportado `h` frente a un cuadrado local.
2. La ventaja crece con el lag: a 6 h la estela casi coincide con el cuadrado; a 12 y 24 h
   ya no.
3. El viento y los enlaces entre campos siguen pidiendo estructura local (balance).

**Qué se sabe ya (notas, una semilla, N = 20, swesphere 0.1.0 con esponja).** Con `h`
observado al 10 %, el suavizador con estela da en `h` 0.202 en t-6h contra 0.213 con
cuadrado y 0.257 del mejor LETKS. En el total gana el LETKS al 10 % (por el viento) y el
EnKS-MC al 3 % y al 1 %. Sin prior climatológico, sin estadística sobre semillas, sin lags
mayores. Este plan cubre esos tres huecos.

---

## 2. Formulación

### 2.1 Estado conjunto y factorización

Dos tiempos, `a = t - lag` y `b = t`. Estado conjunto `z = [x_a, x_b]`, con cada bloque
ordenado `u, v, h` y fila mayor de norte a sur (el orden de swesphere).

```
B_z^{-1} = L^T D L,     L = | L_aa   0   |     D = diag(D_a, D_b)
                            | L_ba  L_bb |
```

Esto es la factorización `p(z) = p(x_a) p(x_b | x_a)`:

- `L_aa, D_a` describen el **prior** en el tiempo anterior.
- `L_ba, L_bb, D_b` describen la **transición**: cómo depende el estado nuevo del anterior
  (`L_ba`, el propagador) y de sí mismo (`L_bb`).

Los dos bloques se construyen por separado en el código. En el primer análisis ambos salen
del ensamble. En el ciclado (E-S2) el bloque del prior se estima del ensamble de análisis
del ciclo anterior, que es la muestra que el ensamble tiene de ese posterior, y el bloque
de transición se estima del par (ver la nota de E-S2).

**Por qué dos tiempos y no una cadena.** Para lags de 12 y 24 h se usa igualmente el par
`[x_{t-lag}, x_t]` con una sola estela larga. Una cadena de bloques cada 6 h tendría
estelas cortas en cada eslabón, casi iguales al cuadrado, y escondería el efecto que se
quiere medir. La cadena es la forma natural del ciclado (E-S2).

### 2.2 Regresión de cada fila

Para la fila `i` con predecesores `P(i)` y anomalías `X` (N miembros):

```
beta_i = (X_P^T X_P + a_i I)^{-1} (X_P^T y_i + a_i beta_i^clim)
a_i    = alpha * tr(X_P^T X_P) / p_i
L_{i,P(i)} = -beta_i^T,   L_ii = 1,   D_ii = 1 / sigma_i^2
```

Cresta relativa (la absoluta no hace nada en unidades del modelo, según las notas).
`beta_i^clim = 0` da el EnKF-MC clásico; distinto de cero da el prior climatológico. Las
anomalías se normalizan por campo con la desviación climatológica. Cuando `p_i > N` se
resuelve en forma dual (sistema N x N).

### 2.3 Análisis

Un solo sistema disperso con observaciones perturbadas; las observaciones son de `h` en `t`,
así que `H = [0, H_b]`:

```
(B_z^{-1} + H^T R^{-1} H) dZ = H^T R^{-1} (Y + E - H Z_b)
```

En el código se resuelve la forma equivalente en espacio de observaciones, que solo usa
resoluciones triangulares con `L` (sin ensamblar `B_z^{-1}`, sin relleno):

```
V = L^{-T} H^T,   S = V^T D^{-1} V + R,   dZ = L^{-1} D^{-1} V S^{-1} (Y + E - H Z_b)
```

La actualización de `x_a` llega solo a través de `L_ba`. Sin enlaces entre tiempos, `x_a`
no se mueve (esto se usa como prueba, sección 9).

### 2.4 Cuál es el primer prior

Hay dos cosas que se llaman "prior" y conviene no mezclarlas:

| prior | sobre qué | qué es en el primer análisis |
|---|---|---|
| del estado | `x_a`, es decir el bloque `L_aa, D_a` | el Cholesky modificado del ensamble inicial en `t - lag`, con predecesores en cuadrado de radio `r` |
| de los coeficientes | cada `beta_i` | el centro de la cresta: cero (EnKF-MC clásico) o `beta_i^clim` (prior climatológico) |

En el primer análisis no hay posterior anterior que heredar, así que el prior del estado se
estima del ensamble inicial en `t0` (sección 5). Ese ensamble nace de restas de estados del
registro y luego se propaga, de modo que sus correlaciones se parecen a las del clima pero
ya llevan la forma del flujo: `beta^clim` es un centro aproximado, no exacto. Desde el
segundo ciclo (E-S2) el bloque `L_aa, D_a` sale del ensamble de análisis del ciclo
anterior.

### 2.5 El papel del radio `r`

`r` es el único radio que queda y solo gobierna la parte local: los cuadrados dentro de un
mismo tiempo y el cuadrado entre tiempos de las filas de `u, v`. La estela no tiene radio:
su largo y su forma salen del viento, y solo queda el ancho `w`.

- Sin prior climatológico: `r = 2`. Con 20 a 40 miembros una fila más ancha paga varianza
  sin quitar sesgo (Bickel y Levina; medido en las notas).
- Con prior climatológico: se prueba también `r = 3`. En las notas el orden de los radios
  se invierte con el prior, porque los coeficientes ya no dependen solo de N miembros
  (en `two_jets`, `r = 3` con prior fue el mejor).
- En el ciclado `r` queda fijo. Lo que se actualiza en cada ciclo es la estela (del viento
  de ese momento) y los coeficientes; no hay barrido de radios.

---

## 3. Conjuntos de predecesores

| fila | mismo tiempo | tiempo anterior |
|---|---|---|
| `x_a` (u, v, h) | cuadrado de radio `r`, tres campos | no aplica |
| `h` en `x_b` | cuadrado de radio `r`, tres campos | **estela**: `u, v, h` de las celdas a distancia `w` de la trayectoria, más la celda propia |
| `u, v` en `x_b` | cuadrado de radio `r`, tres campos | cuadrado de radio `r`, tres campos |

- Dentro de un mismo tiempo solo cuentan las celdas anteriores en el orden (medio cuadrado).
  Entre tiempos todo `x_a` precede a `x_b`, así que se usa el vecindario completo.
- Brazo de control (`square`): la misma tabla con cuadrado en lugar de estela para `h`. Es
  la única diferencia entre los dos métodos, de modo que la comparación aísla el efecto de
  la estela.
- Control de tamaño (`patch`): parche isótropo que tiene, celda a celda, exactamente tantas
  celdas como la estela (las más cercanas en distancia euclídea). Separa "más predecesores"
  de "predecesores mejor puestos".
- Variante `wake<P>`: la estela calculada con el viento multiplicado por `P` por ciento
  (`wake50` es media velocidad). Sirve para probar si el patrón de error de `h` viaja más
  lento que el aire (ver sección 13).

### 3.1 La estela

Para cada celda se integra el viento medio del ensamble hacia atrás desde `t` hasta
`t - lag` (RK en lat-lon, periódico en longitud, tope de longitud de trayectoria). Los
predecesores son las celdas a distancia de Chebyshev menor o igual que `w` de algún punto
de la trayectoria.

- Ancho `w = 2` por defecto (en las notas, 1 es poca estructura y 3 no cambia nada).
- Viento: media del ensamble interpolada entre los estados guardados en `t - lag` y `t`
  (y los intermedios cada 6 h cuando el lag es mayor). El suavizador ya tiene esos estados.
- Longitudes esperadas según las notas: unas 2, 4 y 8 celdas en los chorros a 6, 12 y 24 h.
  En 0.6.0 el viento es más fuerte y serán más largas; se mide antes de lanzar nada.

### 3.2 Dominio estimado

Filas 10 a 55 (62.7 N a 60 S), como en las notas. En 0.6.0 no hay esponja, pero la fila
del polo tiene `u, v` multivaluados y las filas polares son rugosas, así que no se estiman
ni se puntúan. Las trayectorias que salen de la banda se cortan en el borde.

---

## 4. Prior climatológico en el suavizador

Para las filas de un mismo tiempo es igual que en el filtro: `beta^clim` sale de la
regresión sobre estados del registro climatológico.

Para los enlaces entre tiempos hacen falta **pares** `(S(t - lag), S(t))`. El registro se
genera guardando, para cada instantánea, también los estados a +6, +12 y +24 h (mismo
recorrido, sin costo extra de modelo).

**Punto abierto.** La estela cambia con el flujo del día, pero el coeficiente climatológico
para esas celdas es un promedio sobre todos los flujos. Puede que el prior sea informativo
para la parte local de la fila y débil para la parte de la estela. Por eso el prior se
prueba en tres variantes: apagado, solo en las filas de mismo tiempo, y en toda la fila.

---

## 5. Verdad, fondo, ensamble y observaciones

Experimento gemelo estándar. Tres objetos que llegan al mismo instante `t0`, todos
consistentes con la dinámica del modelo:

| objeto | construcción | papel |
|---|---|---|
| verdad (referencia) | un estado del registro, propagado con el modelo | de ella salen las observaciones y contra ella se mide el error |
| fondo | verdad + ruido en un tiempo anterior, propagado | centro del ensamble; sigue sin asimilar como **corrida libre**, que es la línea base |
| ensamble | fondo ya propagado + un ruido por miembro, propagado otra vez | lo que se asimila |

```
t0-T1-T2            t0-T2                 t0              t0+lag
   |-------- T1 ------|-------- T2 --------|------ lag ------|
verdad  ──────────────────────────────────────────────────────>  observaciones de h
fondo = verdad + ruido ───────────────────────────────────────>  corrida libre
                    miembros = fondo + ruido_k ───────────────>  ensamble
                                          x_a               x_b
```

- **El ruido no puede ser blanco.** En aguas someras un ruido gaussiano punto a punto está
  desbalanceado y se va en ondas de gravedad por toda la esfera en unas 3 horas (medido en
  las notas). Se usa ruido balanceado: `0.05 (S_i - S_j)`, la resta escalada de dos estados
  del registro (receta 11.3 de la referencia). Medido en la banda: unos 28 m de dispersión
  inicial en `h` (la desviación climatológica de `h` en la banda es 438 m), no los 19 m
  que cita la referencia para todo el globo.
- **Propagación de ajuste.** `T1 = T2 = 2 días` por defecto, dentro de un rango razonable
  de 1 a 5 días. Tres escalas lo acotan:
  - mínimo, unas horas: lo que tarda en irse la parte desbalanceada del ruido como ondas
    de gravedad (unas 3 h en las notas);
  - suficiente, 1 a 2 días: el viento ya transportó las perturbaciones varias celdas
    (8.7 celdas en 24 h en las notas), así que llevan la forma del flujo de ese momento;
  - máximo, unos 8 días: el tiempo de duplicación del error (7 a 8 días); más allá el
    ensamble crece demasiado y sale del régimen lineal.

  En 2 días el error crece menos de 1.2 veces. No alcanza para que el ensamble se alinee
  con las direcciones de crecimiento del error; eso lo hacen los ciclos (por eso los 4
  primeros no se puntúan). La prueba corta (`es0_spinup_check.py`, una semilla, N = 20) dio:

  | días | dispersión de `h` (m) | crecimiento | desbalance (div/vort) | dispersión / error |
  |---|---|---|---|---|
  | 0 | 28.2 | 1.00 | 0.025 | 0.72 |
  | 0.25 | 29.0 | 1.03 | 0.056 | 0.74 |
  | 1 | 29.2 | 1.04 | 0.039 | 0.72 |
  | 2 | 31.2 | 1.11 | 0.033 | 0.82 |
  | 4 | 34.6 | 1.23 | 0.027 | 0.87 |
  | 8 | 42.7 | 1.51 | 0.019 | 0.82 |

  El ruido es balanceado desde el principio (divergencia de 2 a 6 % de la vorticidad, con
  un pico pequeño a las 6 h que decae). La estructura de correlación no llega a
  estabilizarse: sigue cambiando despacio todo el tiempo, así que la prueba no señala un
  tiempo único. Se mantienen 2 días: desbalance ya decaído y crecimiento de 1.11.
- **Consistencia.** Fondo y miembros reciben ruido del mismo tamaño, así que el error de
  la media (unos 38 m en `h`) y la dispersión (unos 31 m) quedan del mismo orden, con el
  ensamble algo subdispersivo (cociente 0.7 a 0.9). El ensamble no queda centrado en la
  verdad (trampa 12).
- **Estado conjunto.** `x_a` es el ensamble en `t0` y `x_b` su propagación a `t0 + lag`;
  el miembro conjunto es `[x_a^k, M(x_a^k)]`.
- **Observaciones.** Solo `h` de la verdad en `t0 + lag`, en sitios aleatorios dentro de la
  banda, con error de 6 m (el del artículo de swesphere). Cuatro densidades: 1, 3, 10 y
  25 % de las celdas de `h` (unos 60, 180, 610 y 1520 sitios). Como `u, v` no se observan,
  eso es un tercio en fracción del estado: 0.3, 1, 3.3 y 8.3 %.
- **Lugar de las observaciones.** En E-S1 hay un solo análisis, así que cada semilla
  sortea su red y todos los métodos usan la misma. En el ciclado (E-S2) se prueban las dos
  opciones: red que se sortea de nuevo en cada ciclo y red fija.
- **Inflación.** En E-S1 no se aplica (1.00): hay un solo análisis y el ensamble nace con
  dispersión igual a su error. En E-S2 es un eje, igual para todos los métodos.
- **Modelo.** `presets.two_jets()`, L = 32, dt = 120 s, sin esponja. A `propagate` siempre
  se le pasa el tiempo corriente.

**El ruido se añade una sola vez.** Esta construcción es solo el arranque. Desde ahí, el
ensamble del ciclo `k` es el análisis del ciclo `k - 1` propagado con el modelo; no se
vuelve a perturbar el fondo en ningún ciclo (lo único que se aplica por ciclo es la
inflación). La verdad y la corrida libre siguen propagándose sin tocarse. En E-S1 cada
semilla es un arranque independiente con un solo análisis; en E-S2 hay un arranque por
semilla y luego solo ciclos.

**Consecuencia para el prior climatológico.** Tras la propagación de ajuste la covarianza
del ensamble ya no es la del clima por una constante: es la del clima transportada `T2`
por la dinámica. `beta^clim` deja de ser exacto y pasa a ser una aproximación, cuya
utilidad se mide (sección 4). Es el caso realista: un ensamble que ya conoce el flujo.

**Modo `climatologico` (solo comparación).** Miembros y verdad extraídos directamente del
registro, sin propagación de ajuste, con error de observación del 5 % de la desviación
climatológica. Se corre solo a 6 h para poner los resultados junto a los de las notas; ahí
el prior climatológico sí es exacto.

---

## 6. Experimentos

### E-S1. Primer análisis (primera entrega)

| eje | valores |
|---|---|
| lag | 6, 12, 24 h |
| N | 20, 40 |
| densidad de `h` | 1, 3, 10, 25 % |
| semillas | 20 (piloto con 5) |
| ensamble | gemelo con propagación de ajuste (sección 5); `climatologico` solo a 6 h |

Brazos:

1. LETKS, caja y Gaspari-Cohn, radios 3, 4, 6, 8, 10 y 14 (pesos del ETKF local del bloque
   `x_b` aplicados a los dos tiempos). Los radios llegan a 14 porque en este régimen el
   LETKS mejora hasta radios de 8 a 10 celdas.
2. EnKF-MC como filtro, solo sobre `x_b` (sin bloque entre tiempos), `r` en {2, 3}.
3. EnKS-MC, cuadrado entre tiempos, `r` en {2, 3}.
4. EnKS-MC, estela entre tiempos para `h`, `r` en {2, 3}, `w = 2`.
5. Control de tamaño: parche isótropo con tantas celdas como la estela.
6. Los brazos 3 y 4 con prior climatológico (variantes `same` y `full`).

Cresta relativa `alpha` en {0.003, 0.03, 0.3} sin prior y {0.03, 0.3, 3} con prior. La
exploración inicial mostró que en este régimen conviene una cresta mucho menor que la de
las notas (el error mejora de forma monótona al bajar de 3 a 0.003).

Métrica: RMSE del análisis sobre RMSE del fondo, en cada tiempo, total y por campo, con
peso `cos(phi)` y normalización por campo. Se reporta media y error estándar sobre
semillas, y la diferencia pareada estela menos cuadrado. La corrida libre se puntúa igual
y aparece como línea base en todas las tablas.

Costo de modelo: lo domina la propagación de ajuste. Con 40 miembros y 2 días son 80 días
de modelo, unos 3.5 minutos por semilla, cerca de 70 minutos para las 20 (el ensamble de
N = 20 es un subconjunto del de 40). La ventana del suavizador añade como mucho otro
minuto por semilla. El registro climatológico (60 días de arranque más 400 instantáneas cada 2 días) es menos
de una hora. El costo real está en las regresiones y el sistema disperso, y se mide en el
piloto.

**Advertencia de muestreo.** 400 instantáneas cada 2 días contienen unas 8 muestras
independientes de la amplitud de los remolinos (tau de 53 días). Veinte semillas del mismo
registro no son veinte flujos independientes. Si el piloto lo justifica, se alarga el
registro o se usan varios.

### E-S2. Ciclado con prior heredado (segunda entrega)

Ciclos de 6 h, **60 ciclos** (15 días de modelo, unos dos tiempos de duplicación del
error), 3 semillas. Los 4 primeros no se puntúan; se reportan por separado el transitorio
(ciclos 4 a 19) y el tramo final (ciclos 40 a 59). Una corrida larga de 120 ciclos con una
semilla comprueba si el tramo final ya es estado estacionario o todavía deriva. Costo: con
N = 20 cada corrida son 300 días de modelo, unos 13 minutos; el diseño completo (métodos
por inflación por densidad por red por semilla) son del orden de 150 a 200 corridas, así
que primero va un piloto de una semilla para descartar inflaciones y brazos que no
compiten. Arranca de la misma
construcción de la sección 5 y la corrida libre acompaña todos los ciclos como línea base.
El bloque `L_aa, D_a` se
estima del ensamble de análisis del ciclo anterior y la transición del par, con la estela
recalculada en cada ciclo. (Heredar el posterior de forma exacta, sin pasar por el
ensamble, exige la cadena completa de tiempos en forma de información, porque marginar el
tiempo viejo llena la matriz; queda como extensión.) Inflación {1.00, 1.02, 1.05} como eje para todos los métodos
(en las notas, las filas con memoria piden menos inflación y compararlas a 1.05 es injusto).
Opcional: filas bayesianas con memoria y peso del clima por evidencia.

| eje de E-S2 | valores | por qué |
|---|---|---|
| densidad de `h` | 10 % y 3 % | la red del caso base y una red rala, donde el EnKS-MC ganó en las notas |
| red | sorteada en cada ciclo; fija | en el QG las redes fijas y ralas acumularon error en los huecos y derivaron para todos los filtros; hay que ver si aquí la advección llena los huecos |
| inflación | 1.00, 1.02, 1.05, 1.10 | multiplicativa sobre las anomalías de pronóstico; cada método se reporta con su mejor valor y con la tabla completa (1.02 es la del artículo de swesphere) |

### E-S3. Control negativo en el QG (más adelante)

El mismo suavizador sobre `qg-python`. Las notas miden que allí el error no se advecta, así
que la predicción es que la estela no gane al cuadrado. Si se confirma, refuerza el
argumento físico.

---

## 7. Qué se guarda en cada corrida

Regla: **ninguna figura vuelve a correr el modelo ni el filtro.** Cada figura tiene su
script y lee solo archivos guardados. Todo en `float32`, con un `config.json` por corrida
(semilla, parámetros, versiones de paquetes).

Para **todas** las semillas (liviano):

| archivo | contenido |
|---|---|
| `metrics.csv` | una fila por semilla, lag, N, densidad y brazo: RMSE de fondo y de análisis por tiempo y por campo, dispersión, número medio de predecesores, tiempos de cómputo |
| `fields_<corrida>.npz` | verdad y corrida libre en los dos tiempos, media de fondo, media de análisis de cada brazo, dispersión de fondo y de análisis, sitios y valores observados |
| `wake_<corrida>.npz` | mapa de longitud de estela, número de predecesores por fila, trayectorias de un subconjunto de puntos |

Para las **semillas de galería** (una por lag, pesado):

| archivo | contenido |
|---|---|
| `ensemble_<corrida>.npz` | ensambles completos de fondo y de análisis en los dos tiempos, por brazo |
| `precision_<corrida>.npz` | `L` y `D` dispersas del estado conjunto, con estela y con cuadrado |
| `reference_<corrida>.npz` | ensamble grande de referencia (N = 200) propagado entre los dos tiempos, para tener la correlación entre tiempos "verdadera" |
| `trajectory_<corrida>.npz` | verdad cada hora entre `t - lag` y `t`, para animaciones |

Una vez por proyecto: el registro climatológico con pares (400 instantáneas por 4 tiempos,
unos 170 MB).

Tamaños: un estado son unos 105 kB en `float32`; el ensamble conjunto de 20 miembros son
unos 4 MB por brazo. Guardar ensambles completos solo en las semillas de galería mantiene
todo E-S1 en pocos GB.

---

## 8. Figuras

Paleta fija por método en todas las figuras (segura para daltonismo), salida en PDF
vectorial y PNG.

### 8.0 F0, la figura insignia: una partícula, su viaje y su fila de Cholesky

Se elige un punto `i` de `h` sobre el chorro y se cuenta toda la idea en una sola figura de
tres paneles enlazados por color.

| panel | contenido |
|---|---|
| A. El viaje | Mapa ampliado de la vorticidad en `t` (el campo en `t - lag` en contornos tenues). El punto `i` marcado; su trayectoria hacia atrás hasta `t - lag` como una línea con marcas cada hora que se van desvaneciendo. Sombreadas en un color las celdas de la estela (predecesores en `t - lag`) y en otro el cuadradito local (predecesores en `t`). |
| B. La fila | El patrón de `L` del estado conjunto con la fila `i` resaltada y, debajo, esa fila estirada como una tira dividida en `u_a, v_a, h_a, u_b, v_b, h_b`. Cada entrada no nula lleva el mismo color que su celda en el panel A; líneas finas unen algunas celdas del mapa con sus columnas. |
| C. Los pesos | Los coeficientes `beta` de esa fila devueltos al mapa: tres mapas pequeños (`u, v, h` en `t - lag`) con escala divergente. Se ve cuánto pesa cada predecesor y que el peso se concentra corriente arriba. |

Variantes de la misma figura:

- Un punto del chorro junto a uno de zona quieta: estela larga y curva contra parche local.
- Los tres lags en fila (6, 12, 24 h): la estela se alarga y la banda en la tira se aleja
  de la diagonal.
- La misma fila con cuadrado entre tiempos, como contraste: el cuadrado no alcanza el
  origen de la partícula.
- Versión animada: la partícula retrocede en el tiempo y las entradas de la fila se van
  encendiendo a medida que la trayectoria toca cada celda.

Cuidado estético: proyección ortográfica centrada en el punto para el panel A, un solo
color por tiempo (uno para `t - lag`, otro para `t`) usado igual en mapa y matriz, fondo
del campo en escala neutra para que resalten trayectoria y predecesores, tipografía y
grosores iguales a los del resto de figuras.

Datos que necesita (ya previstos en la sección 7): trayectoria horaria del punto, verdad
cada hora, y `L` del estado conjunto de la semilla de galería.

### 8.1 Figuras que explican la idea (semillas de galería)

| | figura | qué muestra |
|---|---|---|
| F1 | Vorticidad en `t` con estelas superpuestas en varios puntos, paneles a 6, 12 y 24 h | la estructura la dibuja el flujo: larga y curva en los chorros, un parche en las zonas quietas |
| F2 | Mapa de longitud de estela, un panel por lag | la estela "pinta" los dos chorros y nada más |
| F3 | Patrón de dispersión de `B_z^{-1}`: vista por bloques y acercamiento a las filas de un punto del chorro, estela contra cuadrado | dónde vive la dinámica en la matriz: el bloque entre tiempos |
| F4 | Correlación entre tiempos de un punto de `h`: referencia (N = 200), la implicada por el cuadrado y la implicada por la estela, tres mapas lado a lado | la figura más importante de este grupo: la estela reproduce la forma real de la correlación y el cuadrado no |
| F5 | Mapas de error de `h` en `t - lag`: cuadrado, estela y su diferencia, con los chorros en contorno | la ganancia está donde hay advección |

### 8.2 Figuras de resultados (todas las semillas)

| | figura | tipo |
|---|---|---|
| F6 | Ganancia pareada (estela menos cuadrado) en `h` en `t - lag`, por lag | cajas con los puntos de cada semilla encima y una línea en cero; **la figura titular**: debe crecer de 6 a 24 h |
| F7 | La misma ganancia frente al control de tamaño | cajas; separa posición de cantidad de predecesores |
| F8 | RMSE relativo de cada método por semilla, paneles por lag y por campo | cajas agrupadas por método (LETKS, cuadrado, estela, con y sin prior) |
| F9 | RMSE relativo frente a densidad de observación, una curva por método con banda de error estándar | líneas; muestra el cruce con el LETKS al ralear la red |
| F10 | Reducción de error por celda agrupada por longitud local de estela | cajas por intervalo de longitud; ata la ganancia a la dinámica punto a punto |
| F11 | Sensibilidad al prior: RMSE frente a `alpha` para las tres variantes | líneas con banda |
| F12 | Tabla de calor: método por densidad, un panel por campo y tiempo | resumen para el apéndice |

Sobre las cajas: con 20 semillas van siempre con los puntos individuales encima, y las
comparaciones entre métodos se muestran como diferencias pareadas (misma semilla, misma
red), que es donde se ve un efecto pequeño. Cajas de valores absolutos lado a lado
esconden la ganancia detrás de la variación entre flujos.

### 8.3 Figuras del ciclado (E-S2)

| | figura | tipo |
|---|---|---|
| F13 | RMSE por ciclo, media y banda sobre semillas, por método, con la corrida libre como referencia | líneas |
| F14 | Cociente dispersión sobre error por ciclo, un panel por inflación | líneas; muestra qué método pide menos inflación |
| F15 | Animación de la estela siguiendo el flujo durante los ciclos | para charlas |

---

## 9. Código

Paquete autocontenido sobre `swesphere` 0.6.0 (nombre provisional `flowsmoother`).

| archivo | contenido |
|---|---|
| `grid.py` | banda lat-lon periódica en longitud, índices, cuadrados, parche de tamaño dado |
| `wake.py` | trayectorias hacia atrás del viento medio y celdas de la estela (con factor de velocidad) |
| `rows.py` | regresión por fila con cresta relativa y centro; tres estimadores de la varianza residual; ensambla `L` y `D` |
| `joint.py` | grafo de predecesores del estado conjunto y del filtro; invariantes (`check_graph`) |
| `analysis.py` | `enks_mc` (espacio de observaciones), referencia en espacio de estados, `letks` |
| `clim.py` | registro climatológico con pares a +6, +12, +24 h; `beta^clim` |
| `ensemble.py` | verdad, fondo (corrida libre) y ensamble con propagación de ajuste; redes de observación |
| `smoother.py` | precisión conjunta para cada estructura y prior, con cachés entre brazos |
| `metrics.py`, `summary.py`, `io.py`, `plotting.py`, `model.py` | puntajes, estadística sobre semillas, archivos, estilo de figuras, propagación en paralelo |
| `experiments/es0_climatology.py`, `es0_spinup_check.py` | registro; prueba de la propagación de ajuste |
| `experiments/es1_first_analysis.py`, `summarize_es1.py` | E-S1 y sus tablas |
| `experiments/verify_predecessors.py` | invariantes de estructura y verificación contra el ensamble de referencia |
| `experiments/es2_cycled.py` | E-S2 |
| `figures/f00_...py` a `f15_...py` | un script por figura de la sección 8; leen solo archivos guardados |
| `tests/` | 44 pruebas |
| `Dockerfile`, `docker-compose.yml`, `run.sh` | imagen y etapas (`test`, `quick`, `clim`, `spinup`, `pilot`, `full`, `cycled`, `figures`) |

Pruebas antes de cualquier número:

1. Gaussiana sintética con precisión conocida: las filas la recuperan cuando N es grande.
2. Sin enlaces entre tiempos, `x_a` no cambia en el análisis.
3. Con todos los predecesores y N grande, el análisis coincide con el EnKS denso.
4. La estela de una rotación sólida (TC2) coincide con el desplazamiento analítico.
5. `beta^clim = 0` reproduce exactamente el brazo sin prior.

Las cinco están en `tests/` y pasan, junto con las de predecesores: todo predecesor es un
índice menor; conteos 12, 37 y 62 con r = 2; cada par de vecinos enlazado una sola vez;
costura periódica; bloques entre tiempos; dirección de la estela (viento del oeste, estela
al oeste; viento del sur, estela al sur); viento nulo da el cuadrado; corte por tope y por
borde de banda; el patrón de `L` es el grafo.

---

## 10. Criterios de éxito y de abandono

**Éxito.** En diferencia pareada sobre semillas, la estela mejora `h` en `t - lag` frente
al cuadrado y frente al control de tamaño, y la mejora crece de 6 a 24 h.

**Abandono de la hipótesis 1.** Si a 24 h la estela no supera al control de tamaño, lo que
ayudaba era el número de predecesores y no su posición.

**Resultado parcial aceptable.** La estela gana en `h` pero no en el total (como en las
notas). Entonces la pregunta pasa al ciclado (E-S2), que es donde un mejor `h` puede
volverse mejor viento.

**Lo que no se debe esperar.** Que los números coincidan con los de las notas: aquellos son
de la versión con esponja, con remolinos de unos 160 m; en 0.6.0 son de unos 367 m.

**Qué se espera del modelo nuevo.** Una señal más clara de la estela, no necesariamente
errores más bajos: el viento es más fuerte (unos 80 m/s) y las estelas más largas, pero el
error también es más rico frente a 20 miembros, lo que complica a todos los métodos. La
ventaja del prior climatológico debería crecer. Es una expectativa, no una medición.

---

## 11. Decisiones que necesito confirmadas

Marcar con `[x]` lo que queda aprobado; anotar al lado cualquier cambio.

- [x] **Construcción del experimento.** Gemelo estándar: verdad, fondo = verdad + ruido
      propagado (corrida libre), ensamble = fondo + ruido propagado, todos en el mismo
      instante (sección 5).
- [ ] **Ruido y tiempos de ajuste.** Ruido balanceado `0.05 (S_i - S_j)` (28 m de
      dispersión en `h` en la banda) y `T1 = T2 = 2 días`.
- [x] **Sitios de observación.** Aleatorios (no malla regular), solo `h`, error de 6 m.
- [ ] **Densidades e inflación.** 1, 3, 10 y 25 % en E-S1 sin inflación; en E-S2, 10 y
      3 %, red sorteada por ciclo y red fija, inflación 1.00, 1.02 y 1.05.
- [ ] **Ciclos.** 60 ciclos de 6 h, 4 sin puntuar, más una corrida larga de 120.
- [ ] **Dominio.** Banda de filas 10 a 55 aunque ya no haya esponja.
- [ ] **Viento de la estela.** Media del ensamble a lo largo del intervalo, no solo en `t`.
- [ ] **Tamaños.** N = 20 y 40; 20 semillas.
- [ ] **Nombre del paquete.** Provisional: `flowsmoother`.
- [ ] **Ensamble de referencia.** N = 200 en las semillas de galería (unos 8 minutos de
      modelo por semilla a 24 h); lo necesita F4.

## 12. Orden de trabajo

1. Registro climatológico con pares y medición de longitudes de estela a 6, 12 y 24 h
   (salen F1 y F2). **Hecho** (registro de 300 instantáneas para el piloto).
2. `grid`, `wake`, `rows`, `joint`, `analysis`, `io` con sus pruebas. **Hecho.**
3. Piloto de E-S1: 5 semillas, N = 20, 10 %, tres lags, con las semillas de galería.
   Primera lectura de la hipótesis (salen F3 a F6). **Hecho**; ver `PILOTO.md`.
4. E-S1 completo (salen F7 a F12). Por correr en learn-da.
5. E-S2 (ciclado con prior heredado).
6. E-S3 (QG).

---

## 13. Estado de la implementación (v0.3)

El repositorio implementa E-S1 y E-S2 completos, las 16 figuras y las pruebas. E-S3 (QG)
no está. Lo que cambió respecto al diseño de las secciones anteriores:

| tema | diseño | implementado | por qué |
|---|---|---|---|
| esponja polar | (pregunta abierta) | no se usa; banda de filas 10 a 55 | 0.6.0 no la necesita y la esponja reduce a menos de la mitad los remolinos; las filas polares se dejan fuera por la fila del polo y la rugosidad |
| análisis | sistema disperso en espacio de estados | forma equivalente en espacio de observaciones | evita ensamblar y factorizar `B^-1` (decenas de millones de no nulos); una prueba comprueba la igualdad |
| control de tamaño | cuadrado de radio ajustado | parche isótropo con el mismo número de celdas, celda a celda | iguala el número de predecesores de forma exacta |
| cresta | `alpha` de 1 a 30 | 0.003 a 0.3 (sin prior), 0.03 a 3 (con prior) | medido: el error baja de forma monótona al reducirla |
| radios del LETKS | 2 a 6 | 3 a 14, caja y Gaspari-Cohn | medido: el óptimo está en 10 a 14 |
| dispersión inicial | unos 19 m | 28 m en la banda | medido |
| prior heredado (E-S2) | herencia exacta del posterior | el bloque del prior se estima del ensamble de análisis anterior | la herencia exacta no es dispersa (sección 6) |
| inflación (E-S2) | 1.00, 1.02, 1.05 | se añade 1.10 | en el mini ciclado todos los ensambles quedaron subdispersivos con 1.02 (dispersión/error de 0.3 a 0.66) |
| brazos extra | no previstos | EnKF-MC como filtro; estela a velocidad reducida (`wake50`); varianza residual por dejar-uno-fuera (`--sigma loo`) | líneas base y diagnósticos que el piloto pidió |
| verificación | pruebas unitarias | además `verify_predecessors.py` contra un ensamble de 200 | para comprobar los predecesores contra el modelo |

No se probó aquí: construir la imagen de Docker (el entorno donde se escribió no tiene
acceso a Docker Hub; las mismas etapas de `run.sh` se corrieron sin contenedor), el preset
`full` de E-S1 y los presets `pilot` y `full` de E-S2 a tamaño completo.

Primera lectura de la hipótesis (detalle en `PILOTO.md`): con error inicial de gran escala
(diferencias escaladas del clima), la estela mejora `h` en t − lag frente al cuadrado a 6 y
12 h, no a 24 h, y no se separa del control de tamaño con 5 semillas; el LETKS con radio
grande gana a todos los EnKS-MC de r = 2. El experimento que queda por leer es el ciclado.
