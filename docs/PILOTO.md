# Piloto: primeras mediciones con el repositorio

5 de noviembre de 2026. Todo lo de este documento se midió con el código de este
repositorio en una máquina de 2 núcleos. Es un piloto: 5 semillas, un tamaño de ensamble,
una densidad, un radio. Sirve para leer la hipótesis por primera vez y para decidir qué
correr en learn-da, no para concluir.

Configuración: `two_jets`, L = 32, sin esponja; registro climatológico de 300 instantáneas
(60 días de arranque, una cada 2 días, con pares a +6, +12 y +24 h); verdad, corrida libre
y ensamble con propagaciones de ajuste de 2 días; N = 20; `h` observado en el 10 % de las
celdas de la banda (607 sitios aleatorios) con error de 6 m; r = 2; estela de ancho 2.

Reproducir: `./run.sh clim --n-snapshots 300` y
`./run.sh es1 pilot --structs square wake patch wake50 patch50`.

## 1. Resultado principal

RMSE del análisis sobre RMSE del fondo, media ± error estándar sobre 5 semillas. Para cada
familia se muestra su mejor configuración (elegida con estos mismos datos).

| lag | método | total en t − lag | `h` en t − lag | total en t | `h` en t |
|---|---|---|---|---|---|
| 6 h | LETKS (GC, radio 10) | **0.451 ± 0.011** | **0.227 ± 0.015** | **0.422 ± 0.011** | **0.111 ± 0.005** |
| | EnKS-MC cuadrado | 0.569 ± 0.019 | 0.333 ± 0.027 | 0.538 ± 0.012 | 0.172 ± 0.007 |
| | EnKS-MC parche | 0.565 ± 0.017 | 0.322 ± 0.022 | 0.540 ± 0.011 | 0.173 ± 0.007 |
| | EnKS-MC estela | 0.562 ± 0.014 | 0.309 ± 0.019 | 0.541 ± 0.011 | 0.174 ± 0.007 |
| 12 h | LETKS (GC, radio 14) | **0.502 ± 0.018** | **0.272 ± 0.013** | **0.443 ± 0.011** | **0.124 ± 0.006** |
| | EnKS-MC cuadrado | 0.663 ± 0.038 | 0.478 ± 0.024 | 0.582 ± 0.027 | 0.179 ± 0.008 |
| | EnKS-MC parche | 0.645 ± 0.037 | 0.459 ± 0.033 | 0.589 ± 0.030 | 0.179 ± 0.009 |
| | EnKS-MC estela | 0.636 ± 0.031 | 0.431 ± 0.011 | 0.582 ± 0.020 | 0.179 ± 0.010 |
| 24 h | LETKS (GC, radio 14) | **0.592 ± 0.016** | **0.411 ± 0.011** | **0.421 ± 0.017** | **0.113 ± 0.005** |
| | EnKS-MC cuadrado | 0.811 ± 0.024 | 0.728 ± 0.032 | 0.600 ± 0.033 | 0.183 ± 0.007 |
| | EnKS-MC parche | 0.807 ± 0.032 | 0.732 ± 0.041 | 0.612 ± 0.025 | 0.188 ± 0.010 |
| | EnKS-MC estela | 0.806 ± 0.025 | 0.714 ± 0.035 | 0.602 ± 0.024 | 0.189 ± 0.008 |

Todos los EnKS-MC de la tabla usan cresta α = 0.003 y sin prior climatológico. La corrida
libre queda en 0.98 a 1.00 del fondo: en 24 h el error casi no crece.

**Lectura.** En este montaje el LETKS con radio grande gana en todo, por 10 a 20 puntos en
el total. El EnKS-MC con r = 2 no compite todavía, con ninguna estructura.

## 2. Estela contra cuadrado y contra el control de tamaño

Diferencias pareadas (misma semilla, misma red, mismas observaciones perturbadas, misma
cresta), en puntos de RMSE/fondo ×100. Negativo: gana la estela.

| lag | estela − cuadrado, `h` en t − lag | semillas a favor | estela − parche, `h` en t − lag | semillas a favor |
|---|---|---|---|---|
| 6 h | −2.5 ± 1.4 | 4 de 5 | −1.3 ± 0.9 | 4 de 5 |
| 12 h | −4.7 ± 1.6 | 5 de 5 | −2.7 ± 2.6 | 4 de 5 |
| 24 h | −1.4 ± 3.8 | 3 de 5 | −1.8 ± 2.9 | 2 de 5 |

En el total en t − lag: −0.8 ± 0.6, −2.7 ± 1.0 y −0.5 ± 1.4 frente al cuadrado. En el
tiempo t (el análisis del filtro) las tres estructuras son indistinguibles.

**Lectura frente a los criterios del plan.**

- *"La estela mejora `h` en t − lag frente al cuadrado"*: sí a 6 y 12 h (a 12 h en las
  cinco semillas); a 24 h no se distingue de cero.
- *"La mejora crece de 6 a 24 h"*: no. Crece de 6 a 12 h y a 24 h se pierde en el ruido.
- *"Supera al control de tamaño"*: no demostrado. La diferencia frente al parche tiene el
  signo esperado pero no se separa de cero con 5 semillas en ningún lag. Parte de la
  ganancia frente al cuadrado es número de predecesores, no posición.

La estela a media velocidad (`wake50`) no es mejor que la estela completa (+1.3 ± 0.8,
+1.7 ± 1.3 y +0.7 ± 1.8 en `h` en t − lag).

## 3. Por qué: el error de este montaje es de gran escala

Tres mediciones de `verify_predecessors.py` (semilla de galería, ensamble de referencia de
200 miembros) explican la tabla.

**El error es de escala planetaria.** La correlación entre tiempos de `h` para un punto del
chorro no es una mancha alrededor de la trayectoria: son dipolos de ±0.75 que cubren 40
columnas (figura `f04`). El error inicial es una diferencia escalada de dos estados del
clima, es decir, un patrón del tamaño de los meandros. Por eso el LETKS mejora hasta
radios de 10 a 14 celdas.

**Con 20 miembros la covarianza muestral ya es buena, y la implicada por el Cholesky no.**
Error relativo de la covarianza entre tiempos de `h` (ventana de ±8 filas por ±20 columnas)
frente a la referencia:

| lag | muestral, 20 miembros | implicada, cuadrado | implicada, estela | implicada, parche | cuadrado + prior | estela + prior |
|---|---|---|---|---|---|---|
| 6 h | 0.27 | 1.72 | 1.68 | 1.65 | 0.70 | 0.69 |
| 12 h | 0.30 | 1.67 | 1.58 | 1.55 | 0.85 | 0.85 |
| 24 h | 0.37 | 2.02 | 1.94 | 1.77 | 1.04 | 1.05 |

Un error relativo de 1 equivale a suponer covarianza nula. La precisión dispersa con
r = 2 y cresta casi nula reproduce mal la covarianza de largo alcance, sea cual sea la
estructura; el prior climatológico la acerca, pero no mejora el análisis.

**Para predecir la fila de `h`, la posición no importa.** Fracción no explicada, fuera de
muestra, de la fila de `h` en `x_b` (ajustada con 20 miembros, evaluada en otros 100):

| lag | cuadrado | estela a media velocidad | estela | parche |
|---|---|---|---|---|
| 6 h | 0.0042 | 0.0047 | 0.0053 | 0.0055 |
| 12 h | 0.0049 | 0.0065 | 0.0086 | 0.0087 |
| 24 h | 0.0061 | 0.0107 | 0.0182 | 0.0184 |

La fila de `h` se predice al 98 a 99.5 % con cualquier conjunto, porque sus vecinos del
mismo tiempo ya la determinan casi por completo; la estela y el parche del mismo tamaño dan
lo mismo, y el cuadrado, con menos predecesores, da lo mejor. Los enlaces entre tiempos de
la fila de `h` aportan poco a la fila en sí; lo que la estela cambia es el camino por el
que una observación llega al tiempo anterior.

**`h` no es un trazador.** El pico de la correlación entre tiempos queda más cerca de la
celda propia (2.5 celdas a 6 h en las estelas largas) que del final de la trayectoria
(8 celdas). Lo que el viento transporta es la vorticidad potencial; `h` es su versión
invertida y suavizada, y sus patrones viajan a la velocidad de fase de los remolinos, menor
que el viento del núcleo del chorro.

## 4. Otras mediciones

- **Cresta.** El error del EnKS-MC baja de forma monótona al reducir α de 3 a 0.003 (una
  semilla, total en t − lag a 6 h: 0.77, 0.73, 0.68, 0.62, 0.60). Las notas usaban 1 a 30.
- **Prior climatológico.** No ayuda aquí (0.59 frente a 0.57 a 6 h; 0.87 frente a 0.81 a
  24 h), como era de esperar: tras la propagación de ajuste el ensamble ya no tiene las
  correlaciones del clima, y con 300 estados y filas de 100 a 700 predecesores el ajuste
  climatológico está casi saturado. La variante `same` es claramente peor (una semilla).
- **Radio.** Con r = 3 el EnKS-MC mejora (una semilla, 6 h: 0.55 frente a 0.60 en el
  total), y la ventaja de la estela se reduce.
- **Varianza residual.** Estimarla por dejar-uno-fuera (`--sigma loo`) en lugar de en
  muestra mejora un poco y de forma consistente (una semilla, 6 h: total 0.586 frente a
  0.608; `h` en t 0.151 frente a 0.178) sin cambiar el orden de las estructuras. El piloto
  se corrió con la opción clásica (`insample`).
- **Estelas.** Longitud media de 2.6, 5.1 y 10.0 celdas a 6, 12 y 24 h (percentil 90: 5.5,
  10.9 y 21.7; máximo 35). Entre 39 y 82 celdas por estela. Ninguna tocó el tope de 40
  celdas; del 2 al 6 % se cortan en el borde de la banda.
- **Ajuste.** Dispersión inicial de 28 m en `h`; a los 2 días, 31 m con desbalance del
  3 %. Tabla completa en la sección 5 del plan.

## 5. Ciclado (mini piloto)

Una sola corrida de cada brazo: una semilla, N = 20, 40 ciclos de 6 h (10 días), 10 % de
`h` con sitios sorteados en cada ciclo, inflación 1.02, α = 0.03, r = 2. RMSE normalizado
del análisis (media sobre los ciclos indicados):

| brazo | total, ciclos 5 a 20 | total, ciclos 21 a 40 | `h`, 21 a 40 | `u`, 21 a 40 | dispersión / error |
|---|---|---|---|---|---|
| corrida libre | 0.146 | 0.233 | | | |
| LETKS GC radio 4 | 0.0390 | 0.0268 | 0.0091 | 0.0287 | 0.66 |
| LETKS GC radio 8 | 0.0388 | **0.0224** | **0.0089** | **0.0234** | 0.34 |
| EnKS-MC cuadrado | 0.0569 | 0.0419 | 0.0212 | 0.0413 | 0.30 |
| EnKS-MC estela | 0.0524 | 0.0380 | 0.0203 | 0.0367 | 0.32 |

Reproducir: `./run.sh es2 pilot --cycles 40 --networks redraw --inflations 1.02 --arms
letks-gc-r4 letks-gc-r8 enks-square-r2-a0.03-off enks-wake-r2-a0.03-off --tag mini`.

**Lectura.** Todos los filtros convergen (el error baja de 2.5 a 5 veces mientras el de la
corrida libre se triplica) y siguen bajando al ciclo 40: aún no es estado estacionario. La estela
queda 9 % por debajo del cuadrado en el total y 11 % en el viento, que es la dirección que
el plan esperaba del ciclado (un mejor `h` hoy es un mejor viento mañana). El LETKS sigue
delante por un 40 %. Es una semilla y una inflación: indica dónde mirar, no demuestra nada.
Todos los ensambles quedan subdispersivos con 1.02 (cociente 0.3 a 0.66), así que el eje
de inflación de E-S2 debería extenderse hacia arriba (por ejemplo 1.05 y 1.10).

## 6. Qué propongo correr en learn-da

1. **E-S1 completo** (`./run.sh full`): 20 semillas, N = 20 y 40, cuatro densidades,
   r = 2 y 3. Decide si la ventaja de la estela sobre el parche existe. La hipótesis que
   más promete según las notas es la red rala (1 y 3 %), donde el LETKS necesita radios
   enormes.
2. **E-S2** (`./run.sh cycled pilot` y luego `full`): es el experimento que puede cambiar
   la lectura, porque tras unos ciclos el error deja de ser de gran escala y pasa a vivir
   en los huecos de la red, que es donde la advección tiene algo que decir.
3. Dos variantes baratas que el piloto sugiere: `--sigma loo`, y un montaje de primer
   análisis con error de pequeña escala (por ejemplo, arrancar E-S1 desde el ensamble de un
   ciclo avanzado de E-S2 en lugar de diferencias del clima).

## 7. Costo medido (2 núcleos, un hilo de BLAS por proceso)

| etapa | tiempo |
|---|---|
| modelo | 2.05 s por día de modelo |
| registro de 300 instantáneas | 21 min (el de 400 son unos 29 min) |
| una semilla del piloto (N = 20, 3 lags, 39 brazos) | 11 min |
| semilla de galería (además 200 miembros de referencia) | 21 min |
| ajuste de una precisión conjunta (36 432 filas) | 1 a 5 s; con prior de estela, hasta 45 s la primera vez |
| un análisis EnKS-MC, 607 observaciones | 4 a 5 s (1520 observaciones: 15 s) |
| un análisis LETKS | 0.3 a 7 s según el radio |
| todas las figuras de una corrida | 1.5 min |

## 8. Figuras del piloto

En `docs/figures/` (las demás se generan con `./run.sh figures es1 pilot`):

| archivo | qué muestra |
|---|---|
| `f00_particle_row_lag24_jet_wake.png` | una partícula, su viaje de 24 h y su fila de Cholesky (y `..._square.png`, el contraste) |
| `f01_wakes_on_flow.png`, `f02_wake_length.png` | las estelas sobre el flujo y el mapa de su longitud |
| `f03_sparsity_lag24.png` | el bloque entre tiempos con estela y con cuadrado |
| `f04_crosstime_correlation_lag24_jet.png` | correlación entre tiempos de `h`: referencia y la implicada por cada estructura |
| `f06_paired_gain_square.png`, `f07_paired_gain_patch.png` | ganancia pareada de la estela por lag |
| `f08_methods_boxes.png` | todos los métodos, cajas sobre semillas |
| `f10_gain_by_wake_length_square.png` | ganancia por longitud local de estela |
| `f13_cycled_rmse_an_tot.png` | el mini ciclado |

Los datos están en `docs/pilot/` (`metrics.csv`, `summary_best.csv`,
`verify_predecessors.csv`, `spinup_check.csv`, `es2_mini_metrics.csv`).
