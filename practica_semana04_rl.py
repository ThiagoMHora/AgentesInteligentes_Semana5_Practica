"""
Práctica Semana 04: Q-Learning y Reward Shaping en Gridworld Estocástico (FrozenLake 4x4)
=========================================================================================
Cumplimiento de Evaluaciones:
1. Implementación formal de la Ecuación de Bellman Model-Free (Q-Learning TD) y
   estrategia epsilon-greedy con decaimiento exponencial.
2. Comparación experimental cuantitativa entre el entorno estándar (recompensas
   dispersas) y el esquema de reward shaping (paso -0.01, hoyo -1.0, meta +1.0),
   evaluando si la penalización acelera la convergencia y reduce las trayectorias
   en bucle. Los veredictos se CALCULAN a partir de las métricas obtenidas.

Ejecución:  py practica_semana04_rl.py   (equivalente: py ejercicio2.py)
"""

import inspect
import numpy as np
import random
from typing import Tuple, Dict, Any, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Experimento de sensibilidad a alpha (fuera del enunciado). Se ejecuta DESPUÉS del
# experimento principal y con entornos/tablas propias, por lo que no altera sus resultados.
EJECUTAR_EXPERIMENTO_ADICIONAL = True

# Ancho de la primera columna de la tabla de métricas: debe alojar el nombre más largo
# ("8. Episodios truncados por límite de pasos (últimos 300 ep)", 59 caracteres).
ANCHO_COL_METRICA = 60
ANCHO_COL_VALOR = 16

# Umbral (en puntos porcentuales) por debajo del cual dos tasas se consideran "SIMILAR".
UMBRAL_SIMILAR_PP = 2.0


# ==============================================================================
# ENTORNO GRIDWORLD ESTOCÁSTICO
# ==============================================================================
class EntornoGridworldEstocastico:
    """Entorno Gridworld 4x4 Estocástico (Simulación reducida de FrozenLake)."""
    def __init__(self, reward_shaping: bool = False):
        self.filas = 4
        self.columnas = 4
        self.n_estados = 16
        self.n_acciones = 4  # 0: Izquierda, 1: Abajo, 2: Derecha, 3: Arriba
        self.agujeros = {5, 7, 11, 12}
        self.meta = 15
        self.estado_actual = 0
        self.reward_shaping = reward_shaping

    def reset(self) -> int:
        self.estado_actual = 0
        return self.estado_actual

    def step(self, accion: int) -> Tuple[int, float, bool]:
        # TAREA 1: Simular estocasticidad (33% acción deseada, 66% desvío lateral)
        if random.uniform(0, 1) < 0.66:
            accion = (accion + random.choice([-1, 1])) % 4
            
        # Transición determinista tras aplicar el viento
        r, c = divmod(self.estado_actual, 4)
        if accion == 0: c = max(0, c-1)
        elif accion == 1: r = min(3, r+1)
        elif accion == 2: c = min(3, c+1)
        elif accion == 3: r = max(0, r-1)
        nuevo_estado = r * 4 + c
        self.estado_actual = nuevo_estado

        # TAREA 2: Asignar recompensas según el esquema activado
        terminado = False
        if nuevo_estado in self.agujeros:
            terminado = True
            recompensa = -1.0 if self.reward_shaping else 0.0
        elif nuevo_estado == self.meta:
            terminado = True
            recompensa = 1.0
        else:
            recompensa = -0.01 if self.reward_shaping else 0.0
            
        return nuevo_estado, recompensa, terminado


# ==============================================================================
# EVALUACIÓN 1: BELLMAN MODEL-FREE Y EPSILON-GREEDY CON DECAIMIENTO
# ==============================================================================
def entrenar_q_learning(
    env: EntornoGridworldEstocastico,
    episodios: int = 2000,
    alpha: float = 0.8,
    gamma: float = 0.95,
    epsilon_inicial: float = 1.0,
    epsilon_min: float = 0.01,
    decay_rate: float = 0.995,
    max_pasos_por_episodio: int = 150,
    q_table_inicial: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Entrena el agente utilizando Q-Learning (Off-policy TD Control).

    Ecuación de Bellman Model-Free implementada:
      Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
      * Si s' es un estado terminal (hoyo o meta), max_a' Q(s', a') = 0.
      * Es model-free porque no requiere matrices de transición P(s'|s,a) ni R(s,a).

    Estrategia de Exploración:
      * Epsilon-Greedy con decaimiento exponencial:
          epsilon <- max(epsilon_min, epsilon * decay_rate)
      * Con los valores del enunciado (1.0, 0.995, 0.01): 0.995^n <= 0.01 exige
        n >= ln(0.01)/ln(0.995) ≈ 918.7, es decir, epsilon alcanza su mínimo al
        terminar el episodio 919; el resto del entrenamiento (≈54 %) es casi
        totalmente explotación con un 1 % de exploración residual.

    q_table_inicial: si se proporciona, el entrenamiento continúa desde una COPIA de
      esa tabla (permite entrenamiento acumulativo por lotes en la GUI sin modificar
      la tabla del llamador). Si es None, se parte de ceros (comportamiento original).
    """
    if q_table_inicial is None:
        q_table = np.zeros((env.n_estados, env.n_acciones))
    else:
        if q_table_inicial.shape != (env.n_estados, env.n_acciones):
            raise ValueError(
                f"q_table_inicial debe tener forma {(env.n_estados, env.n_acciones)}, "
                f"se recibió {q_table_inicial.shape}"
            )
        q_table = np.array(q_table_inicial, dtype=float, copy=True)
    epsilon = epsilon_inicial

    # Registro de métricas para la demostración experimental (Evaluación 2)
    metricas = {
        # Retorno no descontado del episodio (con shaping incluye las penalizaciones).
        "recompensas_por_episodio": [],
        # Longitud del episodio: indicador directo de eficiencia de la trayectoria.
        "pasos_por_episodio": [],
        # 1 si alcanzó la meta, 0 si cayó en hoyo o fue truncado. Base de la convergencia.
        "exitos_por_episodio": [],
        # 1 si algún estado se visitó >= 3 veces. Se conserva por compatibilidad, pero
        # en un entorno con 66 % de desvío casi todos los episodios lo cumplen (~90 %),
        # por lo que es poco discriminante entre esquemas.
        "bucles_por_episodio": [],
        # 1 si el episodio se cortó por max_pasos_por_episodio sin llegar a hoyo ni meta:
        # es la señal más directa de "deambular sin terminar" (bucle real).
        "truncados_por_episodio": [],
        # Valor de epsilon tras el decaimiento de cada episodio.
        "epsilons": [],
        # Índice (base 0) del primer episodio exitoso; None si nunca hubo éxito.
        "primer_exito": None
    }

    for ep in range(episodios):
        state = env.reset()
        done = False
        pasos = 0
        recompensa_acumulada = 0.0
        trayectoria_estados = [state]
        bucle_detectado = False

        # El límite de pasos evita episodios infinitos: sin penalización por paso (entorno
        # estándar) nada impide que el agente deambule indefinidamente entre casillas seguras.
        while not done and pasos < max_pasos_por_episodio:
            # ------------------------------------------------------------------
            # EVALUACIÓN 1: Selección de Acción Epsilon-Greedy con desempate aleatorio
            # ------------------------------------------------------------------
            if random.uniform(0, 1) < epsilon:
                # Exploración: Acción aleatoria uniforme en el espacio de acciones
                action = random.randint(0, env.n_acciones - 1)
            else:
                # Explotación con desempate aleatorio: np.argmax devuelve el PRIMER máximo,
                # así que con Q en ceros (o empates frecuentes en el entorno estándar, donde
                # casi todo vale 0) elegiría siempre la acción 0 "Izquierda", que en el
                # estado inicial choca contra la pared. Sortear entre los empatados evita
                # ese sesgo sistemático y mantiene la exploración implícita.
                max_q = np.max(q_table[state, :])
                candidatos = np.where(q_table[state, :] == max_q)[0]
                action = int(random.choice(candidatos))

            # Ejecución del paso muestral en el entorno (Model-Free)
            next_state, reward, done = env.step(action)
            pasos += 1
            recompensa_acumulada += reward

            # Detección de bucle: si un estado ya se visitó 2 o más veces
            if trayectoria_estados.count(next_state) >= 2:
                bucle_detectado = True
            trayectoria_estados.append(next_state)

            # ------------------------------------------------------------------
            # EVALUACIÓN 1: Ecuación de Bellman Model-Free (Sample-based TD update)
            # TD-Target: r + gamma * max_a' Q(s', a')  (0 en estado terminal)
            # En un estado terminal no hay recompensas futuras: su valor es 0 por
            # definición, así que el TD-Target es solo la recompensa inmediata. Además,
            # Q(hoyo/meta, ·) nunca se actualiza, por lo que arrancar desde ella sería
            # usar un valor que no significa nada.
            # ------------------------------------------------------------------
            # Decisión de diseño: el truncamiento por max_pasos_por_episodio NO se trata
            # como terminal. 'done' solo es True en hoyo o meta; si el bucle se corta por
            # el límite, la última transición ya se actualizó con bootstrapping, porque el
            # estado alcanzado NO es terminal en el MDP (el límite es artificial del
            # entrenamiento). Tratarlo como terminal sesgaría Q hacia 0 en estados
            # intermedios sin que el entorno lo justifique.
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(q_table[next_state, :])

            # TD-Error (delta_t)
            td_error = td_target - q_table[state, action]

            # Actualización estocástica de Bellman para Q-Learning:
            q_table[state, action] += alpha * td_error

            state = next_state

        # Evaluación 1: Decaimiento exponencial de epsilon
        epsilon = max(epsilon_min, epsilon * decay_rate)

        # Registro de métricas
        es_exito = 1 if (done and state == env.meta) else 0
        truncado = 1 if (not done and pasos >= max_pasos_por_episodio) else 0
        if es_exito == 1 and metricas["primer_exito"] is None:
            metricas["primer_exito"] = ep

        metricas["recompensas_por_episodio"].append(recompensa_acumulada)
        metricas["pasos_por_episodio"].append(pasos)
        metricas["exitos_por_episodio"].append(es_exito)
        metricas["bucles_por_episodio"].append(1 if bucle_detectado else 0)
        metricas["truncados_por_episodio"].append(truncado)
        metricas["epsilons"].append(epsilon)

    return q_table, metricas


# ==============================================================================
# EVALUACIÓN 2: DEMOSTRACIÓN EXPERIMENTAL (CONVERGENCIA Y TRAYECTORIAS EN BUCLE)
# ==============================================================================
def evaluar_politica_greedy(env: EntornoGridworldEstocastico, q_table: np.ndarray, n_evals: int = 200) -> Dict[str, Optional[float]]:
    """
    Evalúa la política aprendida en modo puramente greedy (epsilon = 0).

    Devuelve:
      tasa_exito          -> fracción de episodios que llegan a la meta.
      pasos_promedio      -> longitud media de todos los episodios (incluye truncados).
      tasa_bucles         -> fracción con algún estado repetido >= 3 veces (poco discriminante).
      tasa_truncados      -> fracción cortada a los 100 pasos sin hoyo ni meta (bucle real).
      pasos_promedio_exito-> longitud media SOLO de los episodios exitosos (eficiencia del
                             camino hacia la meta); None si no hubo ningún éxito.
    """
    exitos = 0
    pasos_totales = 0
    bucles_detectados = 0
    truncados = 0
    pasos_exitos = 0

    for _ in range(n_evals):
        s = env.reset()
        done = False
        steps = 0
        visitados = [s]
        hubo_bucle = False

        while not done and steps < 100:
            # Acción puramente greedy (explotación)
            max_q = np.max(q_table[s, :])
            candidatos = np.where(q_table[s, :] == max_q)[0]
            a = int(random.choice(candidatos))

            ns, r, done = env.step(a)
            steps += 1
            if visitados.count(ns) >= 2:
                hubo_bucle = True
            visitados.append(ns)
            s = ns

        if s == env.meta:
            exitos += 1
            pasos_exitos += steps
        if not done:
            truncados += 1
        if hubo_bucle:
            bucles_detectados += 1
        pasos_totales += steps

    return {
        "tasa_exito": exitos / n_evals,
        "pasos_promedio": pasos_totales / n_evals,
        "tasa_bucles": bucles_detectados / n_evals,
        "tasa_truncados": truncados / n_evals,
        "pasos_promedio_exito": (pasos_exitos / exitos) if exitos > 0 else None
    }


def calcular_exito_por_bloques(exitos: List[int], tamano_bloque: int = 250) -> List[Tuple[int, int, float]]:
    """
    Divide la serie de éxitos en bloques consecutivos y devuelve (inicio, fin, porcentaje)
    con episodios numerados desde 1. Si el último bloque queda incompleto, se calcula con
    los episodios que contenga.

    Bloques de 250: con 2000 episodios dan 8 filas exactas (tabla legible); cada bloque
    tiene muestra suficiente (1 éxito = 0.4 pp) para suavizar el ruido de un entorno con
    66 % de desvío, y el 4.º bloque (751-1000) contiene el final del decaimiento de
    epsilon (episodio 919), separando la fase exploratoria de la de explotación.
    """
    if tamano_bloque <= 0:
        raise ValueError("tamano_bloque debe ser un entero positivo")
    bloques: List[Tuple[int, int, float]] = []
    for inicio in range(0, len(exitos), tamano_bloque):
        segmento = exitos[inicio:inicio + tamano_bloque]
        porcentaje = 100.0 * sum(segmento) / len(segmento)
        bloques.append((inicio + 1, inicio + len(segmento), porcentaje))
    return bloques


def formatear_tabla_bloques(m_std: Dict[str, Any], m_shp: Dict[str, Any], tamano_bloque: int = 250) -> str:
    """Devuelve el texto de la tabla de tasa de éxito por bloques (reutilizable en consola y GUI)."""
    bloques_std = calcular_exito_por_bloques(m_std["exitos_por_episodio"], tamano_bloque)
    bloques_shp = calcular_exito_por_bloques(m_shp["exitos_por_episodio"], tamano_bloque)

    encabezado = f"{'BLOQUE (EPISODIOS)':<18} | {'ESTÁNDAR':^8} | {'REWARD SHAPING':^14} | MAYOR TASA"
    separador = "-" * len(encabezado)
    lineas = [
        "TASA DE ÉXITO MEDIA POR BLOQUES DE EPISODIOS (ENTRENAMIENTO)",
        separador,
        encabezado,
        separador,
    ]
    for (ini, fin, p_std), (_, _, p_shp) in zip(bloques_std, bloques_shp):
        # Se compara redondeado a 2 decimales para que el ganador coincida con lo mostrado.
        r_std, r_shp = round(p_std, 2), round(p_shp, 2)
        mayor = "Estándar" if r_std > r_shp else "Shaping" if r_shp > r_std else "Empate"
        rango = f"{ini} - {fin}"
        lineas.append(f"{rango:<18} | {f'{p_std:6.2f}%':^8} | {f'{p_shp:6.2f}%':^14} | {mayor}")
    lineas.append(separador)
    return "\n".join(lineas)


def _fmt_pct(valor: float) -> str:
    return f"{valor:6.2f}%"


def _fmt_pasos(valor: Optional[float]) -> str:
    return "N/A" if valor is None else f"{valor:6.2f} pasos"


def _fila_metrica(nombre: str, v_std: str, v_shp: str) -> str:
    return f"{nombre:<{ANCHO_COL_METRICA}} | {v_std:<{ANCHO_COL_VALOR}} | {v_shp:<{ANCHO_COL_VALOR}}"


def _pasos_promedio_exitosos(metricas: Dict[str, Any], ultimos: int = 300) -> Optional[float]:
    """Pasos medios de los episodios exitosos dentro de los últimos `ultimos` episodios."""
    pasos = metricas["pasos_por_episodio"][-ultimos:]
    exitos = metricas["exitos_por_episodio"][-ultimos:]
    pasos_ok = [p for p, e in zip(pasos, exitos) if e == 1]
    return float(np.mean(pasos_ok)) if pasos_ok else None


def _comparar_calidad(valor_std: float, valor_shp: float) -> str:
    """Veredicto de calidad: SIMILAR si la diferencia es menor a UMBRAL_SIMILAR_PP."""
    diferencia = valor_shp - valor_std
    if abs(diferencia) < UMBRAL_SIMILAR_PP:
        return "SIMILAR"
    return "Shaping MEJOR" if diferencia > 0 else "Shaping PEOR"


def _veredicto_dos_condiciones(cond_a: bool, cond_b: bool) -> str:
    if cond_a and cond_b:
        return "CONFIRMADA"
    if cond_a or cond_b:
        return "PARCIAL"
    return "NO CONFIRMADA"


def _si_no(condicion: bool) -> str:
    return "SÍ" if condicion else "NO"


def demostrar_evaluaciones_experimentales(episodios: int = 2000):
    """
    Compara cuantitativamente el entorno estándar y el de reward shaping para evaluar
    dos hipótesis (los veredictos se calculan a partir de los datos, no se asumen):
    1. La penalización por hoyo (-1.0) y por paso (-0.01) ACELERA LA CONVERGENCIA.
    2. La penalización por paso REDUCE LAS TRAYECTORIAS EN BUCLE.
    """
    print("\n" + "="*78)
    print("DEMOSTRACIÓN EXPERIMENTAL COMPARATIVA: ESTÁNDAR VS REWARD SHAPING")
    print("="*78)

    # 1. Entrenamiento de ambos entornos
    env_std = EntornoGridworldEstocastico(reward_shaping=False)
    env_shp = EntornoGridworldEstocastico(reward_shaping=True)

    print(f"\n[1/3] Entrenando Agente Estándar (Sparse Rewards: hoyo=0, paso=0, meta=+1)...")
    q_std, m_std = entrenar_q_learning(env_std, episodios=episodios)

    print(f"[2/3] Entrenando Agente Reward Shaping (hoyo=-1, paso=-0.01, meta=+1)...")
    q_shp, m_shp = entrenar_q_learning(env_shp, episodios=episodios)

    # 2. Evaluación de políticas finales (greedy)
    print(f"[3/3] Evaluando políticas finales aprendidas en 200 episodios independientes...")
    eval_std = evaluar_politica_greedy(env_std, q_std, n_evals=200)
    eval_shp = evaluar_politica_greedy(env_shp, q_shp, n_evals=200)

    # 3. Métricas de Convergencia
    # Fase temprana (1-500): mide la velocidad de aprendizaje mientras epsilon aún es alto.
    tasa_exito_std_500 = np.mean(m_std["exitos_por_episodio"][:500]) * 100
    tasa_exito_shp_500 = np.mean(m_shp["exitos_por_episodio"][:500]) * 100
    # Fase final (últimos 300): epsilon = 0.01, aproxima el rendimiento ya convergido.
    tasa_exito_std_final = np.mean(m_std["exitos_por_episodio"][-300:]) * 100
    tasa_exito_shp_final = np.mean(m_shp["exitos_por_episodio"][-300:]) * 100

    # 4. Métricas de Bucles y Eficiencia de Trayectorias
    # Métrica 4/6 (estado repetido >= 3 veces): se conserva por compatibilidad, pero se
    # considera POCO DISCRIMINANTE: con 66 % de desvío lateral casi cualquier episodio
    # revisita casillas (~90 % en ambos esquemas). Las métricas 8-10 la complementan.
    tasa_bucles_std = np.mean(m_std["bucles_por_episodio"]) * 100
    tasa_bucles_shp = np.mean(m_shp["bucles_por_episodio"]) * 100
    pasos_prom_std = np.mean(m_std["pasos_por_episodio"][-300:])
    pasos_prom_shp = np.mean(m_shp["pasos_por_episodio"][-300:])
    # Truncados: episodios que agotan el límite sin terminar (bucle "real").
    trunc_std_final = np.mean(m_std["truncados_por_episodio"][-300:]) * 100
    trunc_shp_final = np.mean(m_shp["truncados_por_episodio"][-300:]) * 100
    # Pasos en éxitos: eficiencia del camino cuando sí llega (no se mezcla con truncados).
    pasos_exito_std = _pasos_promedio_exitosos(m_std, 300)
    pasos_exito_shp = _pasos_promedio_exitosos(m_shp, 300)

    # 5. Imprimir Tabla Comparativa
    ancho_tabla = ANCHO_COL_METRICA + 2 * ANCHO_COL_VALOR + 6
    print("\n" + "-"*ancho_tabla)
    print(_fila_metrica("MÉTRICA EXPERIMENTAL", "ESTÁNDAR", "REWARD SHAPING"))
    print("-"*ancho_tabla)
    print(_fila_metrica("1. Episodio del 1er éxito alcanzado", str(m_std['primer_exito']), str(m_shp['primer_exito'])))
    print(_fila_metrica("2. Tasa de éxito en fase temprana (Ep 1-500)", _fmt_pct(tasa_exito_std_500), _fmt_pct(tasa_exito_shp_500)))
    print(_fila_metrica("3. Tasa de éxito final (últimos 300 ep)", _fmt_pct(tasa_exito_std_final), _fmt_pct(tasa_exito_shp_final)))
    print(_fila_metrica("4. Tasa de episodios con bucles repetidos", _fmt_pct(tasa_bucles_std), _fmt_pct(tasa_bucles_shp)))
    print(_fila_metrica("5. Longitud promedio de trayectoria (pasos)", _fmt_pasos(pasos_prom_std), _fmt_pasos(pasos_prom_shp)))
    print(_fila_metrica("6. Tasa de bucles en política final (eval greedy)", _fmt_pct(eval_std['tasa_bucles']*100), _fmt_pct(eval_shp['tasa_bucles']*100)))
    print(_fila_metrica("7. Éxito de política final greedy", _fmt_pct(eval_std['tasa_exito']*100), _fmt_pct(eval_shp['tasa_exito']*100)))
    print(_fila_metrica("8. Episodios truncados por límite de pasos (últimos 300 ep)", _fmt_pct(trunc_std_final), _fmt_pct(trunc_shp_final)))
    print(_fila_metrica("9. Pasos promedio en episodios exitosos (últimos 300 ep)", _fmt_pasos(pasos_exito_std), _fmt_pasos(pasos_exito_shp)))
    print(_fila_metrica("10. Truncados en política final (eval greedy)", _fmt_pct(eval_std['tasa_truncados']*100), _fmt_pct(eval_shp['tasa_truncados']*100)))
    print("-"*ancho_tabla)

    # 6. Tasa de éxito media por bloques de episodios (requisito explícito del enunciado)
    print("\n" + formatear_tabla_bloques(m_std, m_shp, tamano_bloque=250))

    # 7. Diagnóstico y Conclusiones de Evaluación (veredictos derivados de las métricas)
    print("\n📋 CONCLUSIONES DE EVALUACIÓN:")
    print("▶ EVALUACIÓN 1 (Bellman Model-Free y Epsilon-Greedy):")
    print("  • Implementada la actualización muestral TD(0) de Bellman: Q(s,a) <- Q(s,a) + alpha*TD_Error.")
    print("  • Decaimiento geométrico de epsilon de 1.0 a 0.01 implementado correctamente.")
    eps_hist = m_std["epsilons"]
    ep_eps_min = next((i + 1 for i, e in enumerate(eps_hist) if e <= 0.01 + 1e-12), None)
    print(f"    -> ε final = {eps_hist[-1]:.4f}; ε_min alcanzado al terminar el episodio {ep_eps_min}.")

    print("▶ EVALUACIÓN 2 (Reward Shaping vs Estándar, veredictos calculados):")

    # 7a. Aceleración de convergencia (fase temprana)
    pe_std, pe_shp = m_std['primer_exito'], m_shp['primer_exito']
    primer_exito_antes = pe_shp is not None and (pe_std is None or pe_shp < pe_std)
    exito_temprano_mayor = tasa_exito_shp_500 > tasa_exito_std_500
    aceleracion = _veredicto_dos_condiciones(primer_exito_antes, exito_temprano_mayor)
    print(f"  • Aceleración de convergencia (fase temprana): {aceleracion}")
    print(f"    -> 1er éxito (episodio, base 0): Estándar = {pe_std} | Shaping = {pe_shp}"
          f"  (Shaping más temprano: {_si_no(primer_exito_antes)})")
    print(f"    -> Éxito Ep 1-500: Estándar = {tasa_exito_std_500:.2f}% | Shaping = {tasa_exito_shp_500:.2f}%"
          f"  (Shaping mayor: {_si_no(exito_temprano_mayor)})")

    # 7b. Calidad final
    exito_greedy_std = eval_std['tasa_exito'] * 100
    exito_greedy_shp = eval_shp['tasa_exito'] * 100
    print(f"  • Calidad final (SIMILAR si la diferencia es < {UMBRAL_SIMILAR_PP:.0f} pp):")
    print(f"    -> Éxito últimos 300 ep: Estándar = {tasa_exito_std_final:.2f}% | Shaping = {tasa_exito_shp_final:.2f}%"
          f"  => {_comparar_calidad(tasa_exito_std_final, tasa_exito_shp_final)}"
          f" ({tasa_exito_shp_final - tasa_exito_std_final:+.2f} pp)")
    print(f"    -> Éxito política greedy: Estándar = {exito_greedy_std:.2f}% | Shaping = {exito_greedy_shp:.2f}%"
          f"  => {_comparar_calidad(exito_greedy_std, exito_greedy_shp)}"
          f" ({exito_greedy_shp - exito_greedy_std:+.2f} pp)")

    # 7c. Reducción de trayectorias en bucle
    menos_pasos = pasos_prom_shp < pasos_prom_std
    trunc_menor_igual = trunc_shp_final <= trunc_std_final
    reduccion_bucles = _veredicto_dos_condiciones(menos_pasos, trunc_menor_igual)
    print(f"  • Reducción de trayectorias en bucle: {reduccion_bucles}")
    print(f"    -> Pasos promedio (últimos 300 ep): Estándar = {pasos_prom_std:.2f} | Shaping = {pasos_prom_shp:.2f}"
          f"  (Shaping menor: {_si_no(menos_pasos)})")
    print(f"    -> Truncados (últimos 300 ep): Estándar = {trunc_std_final:.2f}% | Shaping = {trunc_shp_final:.2f}%"
          f"  (Shaping menor o igual: {_si_no(trunc_menor_igual)})")

    # 7d. Advertencia: Q(0, ·) ≈ -0.01/(1-gamma) es el valor de deambular para siempre
    # pagando -0.01 por paso; si TODAS las acciones lo rondan, la meta no domina el valor.
    gamma_usado = inspect.signature(entrenar_q_learning).parameters["gamma"].default
    limite_deambular = -0.01 / (1 - gamma_usado)
    if np.all(np.abs(q_shp[0, :] - limite_deambular) < 0.05):
        print(f"  ⚠️ ADVERTENCIA: Q_shaping(0) = {np.round(q_shp[0, :], 3)} está a < 0.05 de {limite_deambular:.3f}.")
        print("     Los valores Q cercanos a -0.01/(1-gamma) indican que el agente aprendió a deambular sin terminar el episodio.")

    print("\nFundamento teórico esperado (hipótesis; los veredictos de arriba indican si se cumple):")
    print("  • La penalización por hoyo (-1.0) da retroalimentación negativa inmediata a acciones peligrosas,")
    print("    lo que debería ayudar a descartarlas antes que con recompensa 0 en el hoyo.")
    print("  • Con recompensa 0 por paso, deambular no tiene coste. Con -0.01 por paso, deambular indefinidamente")
    print(f"    vale como mucho -0.01/(1-gamma) = {limite_deambular:.3f}, por lo que caminos más cortos a la meta")
    print("    deberían preferirse. En un entorno tan estocástico, sin embargo, el coste de -1.0 del hoyo puede")
    print("    hacer que evitar riesgos (aunque se alargue o no termine el episodio) resulte más valioso.")

    # 8. Generación de Gráficas Comparativas
    try:
        def media_movil(datos, ventana=100):
            return np.convolve(datos, np.ones(ventana)/ventana, mode='valid')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfica 1: Convergencia (Tasa de Éxito)
        mm_std_exitos = media_movil(m_std["exitos_por_episodio"]) * 100
        mm_shp_exitos = media_movil(m_shp["exitos_por_episodio"]) * 100
        ax1.plot(mm_std_exitos, label='Estándar (Sparse Rewards)', color='#e06c75', linewidth=2)
        ax1.plot(mm_shp_exitos, label='Reward Shaping (Penalización hoyo/paso)', color='#98c379', linewidth=2)
        ax1.set_title('Evaluación 2A: Velocidad de Convergencia\n(Tasa de Éxito - Media Móvil 100 ep)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('% Éxitos (Llegada a Meta)')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='lower right')

        # Gráfica 2: Longitud de Trayectorias
        mm_std_pasos = media_movil(m_std["pasos_por_episodio"])
        mm_shp_pasos = media_movil(m_shp["pasos_por_episodio"])
        ax2.plot(mm_std_pasos, label='Estándar (Sin penalización por paso)', color='#e06c75', linewidth=2)
        ax2.plot(mm_shp_pasos, label='Reward Shaping (-0.01 por paso)', color='#61afef', linewidth=2)
        ax2.set_title('Evaluación 2B: Longitud de Trayectorias\n(Pasos por Episodio - Media Móvil 100 ep)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Pasos Promedio')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('comparativa_convergencia_bucles.png', dpi=150)
        plt.close(fig)
        print("\n📊 Gráfica guardada exitosamente: comparativa_convergencia_bucles.png")
    except Exception as e:
        print(f"Nota: No se pudo exportar la imagen gráfica ({e}), pero las métricas numéricas fueron calculadas.")

    return q_std, q_shp


# ==============================================================================
# EXPERIMENTO ADICIONAL (FUERA DEL ENUNCIADO): SENSIBILIDAD A ALPHA
# ==============================================================================
def experimento_sensibilidad_alpha(
    episodios: int = 2000,
    alphas: Tuple[float, ...] = (0.8, 0.1),
    repeticiones: int = 5
) -> List[Dict[str, Any]]:
    """
    Repite el entrenamiento `repeticiones` veces por cada (alpha, esquema), manteniendo el
    resto de hiperparámetros del enunciado, y promedia los resultados. Con 66 % de desvío
    las transiciones son muy ruidosas: un alpha alto (0.8) hace que Q siga la última muestra
    en vez de su esperanza, por lo que interesa contrastarlo con un alpha bajo (0.1).
    Se promedian varias repeticiones porque una sola corrida tiene mucha varianza.
    Usa entornos y tablas propias: no modifica los resultados del experimento principal.
    """
    print("\n" + "="*78)
    print("EXPERIMENTO ADICIONAL (FUERA DEL ENUNCIADO): SENSIBILIDAD A ALPHA")
    print("="*78)
    print(f"Promedio de {repeticiones} repeticiones por configuración, {episodios} episodios cada una;")
    print("resto de hiperparámetros iguales al enunciado (gamma=0.95, ε 1.0 -> 0.01, decay 0.995).")

    resultados: List[Dict[str, Any]] = []
    for alpha in alphas:
        for shaping in (False, True):
            exitos_greedy, pasos_greedy, exitos_final = [], [], []
            for _ in range(repeticiones):
                env = EntornoGridworldEstocastico(reward_shaping=shaping)
                q, m = entrenar_q_learning(env, episodios=episodios, alpha=alpha)
                ev = evaluar_politica_greedy(env, q, n_evals=200)
                exitos_greedy.append(ev["tasa_exito"] * 100)
                pasos_greedy.append(ev["pasos_promedio"])
                exitos_final.append(np.mean(m["exitos_por_episodio"][-300:]) * 100)
            resultados.append({
                "alpha": alpha,
                "esquema": "Reward Shaping" if shaping else "Estándar",
                "exito_greedy": float(np.mean(exitos_greedy)),
                "pasos_greedy": float(np.mean(pasos_greedy)),
                "exito_ultimos_300": float(np.mean(exitos_final)),
            })

    encabezado = f"{'ALPHA':<6} | {'ESQUEMA':<15} | {'ÉXITO GREEDY':>12} | {'PASOS GREEDY':>12} | {'ÉXITO ÚLT. 300':>14}"
    print("\n" + "-"*len(encabezado))
    print(encabezado)
    print("-"*len(encabezado))
    for r in resultados:
        print(f"{r['alpha']:<6} | {r['esquema']:<15} | {r['exito_greedy']:>11.2f}% | "
              f"{r['pasos_greedy']:>12.2f} | {r['exito_ultimos_300']:>13.2f}%")
    print("-"*len(encabezado))
    return resultados


def main() -> None:
    print("=== EVALUACIÓN EJERCICIO 2: Q-LEARNING Y REWARD SHAPING ===")
    q_std, q_shp = demostrar_evaluaciones_experimentales(episodios=2000)
    
    print("\nTabla Q Entorno Estándar (Estado 0):", np.round(q_std[0, :], 3))
    print("Tabla Q Reward Shaping (Estado 0):", np.round(q_shp[0, :], 3))

    if EJECUTAR_EXPERIMENTO_ADICIONAL:
        experimento_sensibilidad_alpha(episodios=2000, alphas=(0.8, 0.1), repeticiones=5)


if __name__ == "__main__":
    main()
