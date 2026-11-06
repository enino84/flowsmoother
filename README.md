# flowsmoother

Suavizador de Kalman por ensambles con Cholesky modificado (EnKS-MC) en el que los
predecesores entre tiempos los dicta el flujo: para cada punto de `h` se integra hacia
atrás el viento medio del ensamble y las celdas de esa estela son los predecesores en el
tiempo anterior. Banco de pruebas: aguas someras en la esfera, `swesphere` 0.6.0
(`two_jets`, L = 32, sin esponja polar).

El diseño completo, los criterios de éxito y de abandono y las decisiones tomadas están en
[`docs/PLAN_SUAVIZADOR_FLUJO.md`](docs/PLAN_SUAVIZADOR_FLUJO.md). Los primeros números
medidos con este código están en [`docs/PILOTO.md`](docs/PILOTO.md).

## Arranque rápido (Docker)

```bash
mkdir -p results && export UID GID=$(id -g)   # el contenedor escribe con tu usuario
docker compose build
docker compose run --rm fs test          # 44 pruebas, unos 20 s
docker compose run --rm fs quick         # todo a tamaño de juguete, unos 9 min en 2 núcleos, en results/quick
WORKERS=16 docker compose run --rm fs pilot    # registro climatológico + piloto E-S1 + figuras
```

`WORKERS` es el número de procesos (por defecto, todos los núcleos). Cada proceso usa un
solo hilo de BLAS. Los resultados quedan en `./results` (volumen). Todas las etapas se
pueden reanudar: las semillas y corridas terminadas se saltan.

Sin Docker: `pip install -r requirements.txt && pip install -e .` y luego `./run.sh <etapa>`.

## Etapas (`./run.sh <etapa>`)

| etapa | qué hace | salida |
|---|---|---|
| `test` | pruebas unitarias | |
| `quick` | todas las etapas a tamaño reducido, para comprobar la instalación | `results/quick/` |
| `clim` | registro climatológico con pares: 60 días de arranque, 400 instantáneas cada 2 días, cada una con sus estados a +6, +12 y +24 h | `results/clim/record.npy` |
| `spinup` | cuánto deben durar las propagaciones de ajuste | `results/spinup/spinup_check.csv` |
| `pilot` | E-S1 piloto: 5 semillas, N = 20, 10 % de `h`, lags 6, 12 y 24 h; tablas, verificación de predecesores y figuras | `results/es1/pilot/` |
| `full` | E-S1 completo: 20 semillas, N = 20 y 40, densidades 1, 3, 10 y 25 %, r = 2 y 3 | `results/es1/full/` |
| `cycled [pilot\|full]` | E-S2: suavizador lag-1 ciclado, 60 ciclos de 6 h, inflación como eje, red fija y sorteada | `results/es2/<preset>/` |
| `figures es1\|es2 <corrida>` | todas las figuras de una corrida | `results/figures/` |

Cualquier parámetro se puede cambiar por línea de comandos; por ejemplo:

```bash
./run.sh es1 pilot --seeds 0 1 2 --N 40 --dens 0.03 0.10 --tag n40
./run.sh es1 full --structs square wake patch wake50      # estela a media velocidad
./run.sh es2 pilot --cycles 120 --seeds 0 --tag long      # corrida larga de control
./run.sh es1 pilot --sigma loo --tag loo                  # varianza residual por dejar-uno-fuera
```

Con `--tag` la corrida va a otra carpeta; para sus tablas y figuras:
`python experiments/summarize_es1.py --run n40` y `./run.sh figures es1 n40`.

## Qué se compara

Estado conjunto `z = [x(t0), x(t0 + lag)]`, precisión `B^-1 = L^T D L` estimada fila por
fila. Solo se observa `h` en `t0 + lag` (sitios aleatorios, error de 6 m).

| brazo | qué es |
|---|---|
| `letks-<box\|gc>-r<R>` | ETKS local: pesos del bloque `x_b` aplicados a los dos tiempos |
| `filter-r<R>-a<α>-<prior>` | EnKF-MC solo sobre `x_b` (sin bloque entre tiempos) |
| `enks-square-…` | EnKS-MC, cuadrado de radio `r` entre tiempos (brazo de control) |
| `enks-wake-…` | EnKS-MC, estela entre tiempos para las filas de `h` |
| `enks-patch-…` | control de tamaño: parche isótropo con, celda a celda, tantas celdas como la estela |
| `enks-wake50-…` | estela con el viento multiplicado por 0.5 (cualquier porcentaje) |

`a<α>` es la cresta relativa y el último campo el prior sobre los coeficientes: `off`
(centro cero), `same` (clima solo en la parte de mismo tiempo) o `full` (clima en toda la
fila, con los coeficientes entre tiempos sacados de los pares del registro).

Predecesores de cada fila (los verifica `tests/test_predecessors.py`):

| fila | mismo tiempo | tiempo anterior |
|---|---|---|
| `x_a` (u, v, h) | cuadrado `r`, tres campos | no hay |
| `h` en `x_b` | cuadrado `r`, tres campos | estela (o cuadrado, o parche), tres campos, celda propia incluida |
| `u, v` en `x_b` | cuadrado `r`, tres campos | cuadrado `r`, tres campos |

Dentro de un tiempo solo cuentan los índices menores (orden `u, v, h`, fila mayor), de
modo que una fila de `u` ve medio cuadrado de `u`; una de `v`, todo el de `u` y medio de
`v`; una de `h`, todo `u`, todo `v` y medio `h`. Con r = 2 son 12, 37 y 62 predecesores.

## Estructura

```
src/flowsmoother/
  grid.py       banda de latitud, índices, cuadrados, parche de tamaño dado
  wake.py       trayectorias hacia atrás del viento medio y celdas de la estela
  rows.py       regresiones por fila (cresta relativa, centro), ensamblado de L y D
  joint.py      grafo de predecesores del estado conjunto y sus invariantes
  analysis.py   EnKS-MC (forma en espacio de observaciones) y LETKS
  clim.py       registro con pares, coeficientes climatológicos
  ensemble.py   verdad, corrida libre, ensamble, redes y observaciones
  smoother.py   precisión conjunta para cada estructura y prior, con cachés
  metrics.py, summary.py, io.py, plotting.py, model.py
experiments/    es0_climatology, es0_spinup_check, es1_first_analysis, es2_cycled,
                summarize_es1, verify_predecessors
figures/        f00 … f15: un script por figura; leen solo archivos guardados
tests/          44 pruebas
docs/           plan de diseño y resultados del piloto
```

## Archivos que escribe cada corrida

Ninguna figura vuelve a correr el modelo ni el filtro.

| archivo | contenido |
|---|---|
| `config.json` | parámetros, versiones de paquetes, identificador del registro |
| `metrics.csv` | una fila por semilla, lag, N, densidad y brazo: RMSE de fondo, de análisis y de la corrida libre por tiempo y por campo, dispersión, predecesores, tiempos |
| `seedXX/fields_*.npz` | verdad, corrida libre, media y dispersión de fondo, redes y observaciones, media de análisis de los brazos de `save_arms` |
| `seedXX/wake_*.npz` | longitud, número de celdas, desplazamiento y trayectorias de la estela |
| semilla de galería | `ensemble_*.npz` (ensambles completos), `precision_*.npz` (`L` y `D`), `reference_*.npz` (ensamble de referencia), `trajectory.npz` (verdad cada hora) |
| `summary_best.csv`, `verify_predecessors.csv` | tablas de `summarize_es1.py` y `verify_predecessors.py` |

Las métricas son RMSE con peso `cos(latitud)` sobre la banda (filas 10 a 55), en unidades
normalizadas (cada campo dividido por su desviación climatológica); `tot` es la raíz de la
media de los tres errores cuadráticos. `ratio_*` es análisis sobre fondo.

## Notas de implementación

- **Sin esponja polar.** `swesphere` 0.6.0 corre con `n_pole_rows = 0`. Las filas polares
  no se estiman ni se puntúan (banda 62.7° N a 60° S): en la fila del polo `u, v` son
  multivaluados y las filas polares son rugosas. Fuera de la banda los miembros conservan
  su pronóstico.
- **Análisis sin ensamblar `B^-1`.** Se resuelve la forma equivalente en espacio de
  observaciones, `dZ = L^-1 D^-1 V (V^T D^-1 V + R)^-1 (Y + E − H Z_b)` con
  `V = L^-T H^T`, que solo necesita resoluciones triangulares con el factor disperso. Una
  prueba comprueba que coincide con el sistema en espacio de estados.
- **Mismas perturbaciones en todos los brazos.** Para cada semilla, lag, N y densidad, la
  red, las observaciones y las observaciones perturbadas son idénticas en todos los brazos
  EnKS-MC, de modo que las diferencias entre estructuras son pareadas.
- **Selección de configuración.** Las tablas y figuras muestran, para cada familia, la
  configuración con mejor media sobre semillas. Los mismos datos eligen y reportan, así
  que el valor es optimista por igual para todas las familias.
- **Ciclado.** El bloque del prior (filas de `x_a`) se estima del ensamble de análisis del
  ciclo anterior, que es la muestra que el ensamble tiene de ese posterior; el bloque de
  transición se estima del par, con la estela recalculada en cada ciclo.

## Estado

- Probado aquí: las 44 pruebas; todas las etapas a tamaño reducido (`quick`); el registro
  climatológico (300 instantáneas); la prueba de ajuste; el piloto de E-S1 completo con sus
  tablas, verificación y figuras; un ciclado corto de E-S2. Los números están en
  `docs/PILOTO.md` y los datos del piloto en `docs/pilot/`.
- No probado aquí: la construcción de la imagen de Docker (sin acceso a Docker Hub en el
  entorno donde se escribió), el preset `full` de E-S1 y los presets de E-S2 a tamaño
  completo.
- No implementado: E-S3 (control negativo en el QG).

## Licencia

MIT (ver `LICENSE`).
