"""
Práctica Semana 03 — Agente Deliberativo BDI con Inferencia Deductiva
y Reconsideración Dinámica.

Agente SRE autónomo que supervisa microservicios de e-commerce.
Implementa el intérprete BDI de Wooldridge:
    B := brf(B, p)
    D := options(B)
    I := filter(B, D)
    π := plan(B, I, Π)   # con Fallback Plans
    execute(π)

Decisiones de diseño:
  1. La creencia 'anomalia_fuga_memoria' es MONÓTONA: una vez inferida,
     no se resetea (una fuga de memoria no se auto-cura).
  2. El deseo RESTAURAR_SERVICIO_CACHE se condiciona a una creencia
     ('latencia_alta'), no se genera incondicionalmente.
  3. La conmutación al plan fallback se decide DENTRO del agente (en brf),
     no en el bloque main. El main solo orquesta.
  4. Se encapsula el ciclo BDI completo en 'ejecutar_ciclo'.
  5. Se guarda self.plan_activo y self.cola_acciones para dar continuidad
     entre ciclos.
  6. Los planes fallidos se guardan en una LISTA ('planes_fallidos'), no en
     un solo nombre, y se limpian cuando la cuota cloud se restablece.
  7. Solo se marcan como fallidos los planes que dependen de la cuota cloud
     (PlanBDI.requiere_cuota); un plan que no la usa no se descarta.
  8. Un deseo que lleva ciclos activo sin ser intención gana prioridad
     (envejecimiento), sin superar nunca a la seguridad (100).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------
@dataclass
class TelemetriaCluster:
    ciclo: int
    cpu_pct: float
    latencia_ms: float
    alerta_siem: bool
    cuota_agotada: bool = False


@dataclass
class Deseo:
    nombre: str
    prioridad: int
    postergado: int = 0       # ciclos seguidos activo sin ser intención
    envejece: bool = True     # el deseo de fondo no envejece
    incremento: int = 5       # puntos que gana por ciclo postergado
    tope: int = 99            # nunca supera a la seguridad (100)

    @property
    def prioridad_efectiva(self) -> int:
        if not self.envejece or self.postergado == 0:
            return self.prioridad
        return max(self.prioridad,
                   min(self.tope, self.prioridad + self.incremento * self.postergado))


@dataclass
class PlanBDI:
    nombre: str
    intencion_meta: str
    precondicion: Callable[[Dict[str, Any]], bool]
    cuerpo_acciones: List[str]
    requiere_cuota: bool = False   # True si el plan consume cuota cloud


# ---------------------------------------------------------------------------
# Agente BDI
# ---------------------------------------------------------------------------
class AgenteBDI_Estudiante:

    # Umbrales de negocio (evitan "números mágicos")
    VENTANA_CPU = 3
    UMBRAL_INCREMENTO_FUGA = 15.0
    UMBRAL_LATENCIA_ALTA = 200.0

    def __init__(self):
        self.creencias: Dict[str, Any] = {
            "historial_cpu": [],
            "anomalia_fuga_memoria": False,
            "bajo_ataque": False,
            "cuota_cloud_agotada": False,
            "latencia_alta": False,
            "planes_fallidos": [],
        }
        self.intencion_actual: Optional[Deseo] = None
        self.plan_activo: Optional[PlanBDI] = None
        self.cola_acciones: List[str] = []
        self.postergaciones: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # brf: Revisión de creencias con inferencia deductiva
    # ------------------------------------------------------------------
    def brf(self, p: TelemetriaCluster) -> None:
        """
        Actualiza creencias a partir de la telemetría p.

        Regla deductiva: si las últimas 3 lecturas de CPU son
        estrictamente crecientes y (p3 - p1) >= 15.0, se deriva
        proactivamente 'anomalia_fuga_memoria = True' (monótona).

        Reconsideración dinámica: si la cuota cloud se agotó y el plan
        activo depende de ella, se agrega a 'planes_fallidos' para forzar
        el fallback. Cuando la cuota se restablece, la lista se limpia.
        """
        # 1) Ventana deslizante de tamaño 3
        self.creencias["historial_cpu"].append(p.cpu_pct)
        if len(self.creencias["historial_cpu"]) > self.VENTANA_CPU:
            self.creencias["historial_cpu"].pop(0)

        # 2) Sincronizar creencias directas desde la telemetría
        self.creencias["cuota_cloud_agotada"] = bool(p.cuota_agotada)
        self.creencias["bajo_ataque"] = bool(p.alerta_siem)
        self.creencias["latencia_alta"] = p.latencia_ms > self.UMBRAL_LATENCIA_ALTA

        # 3) Inferencia deductiva sobre la ventana (sin else: monotonía)
        hist = self.creencias["historial_cpu"]
        if len(hist) == self.VENTANA_CPU:
            p1, p2, p3 = hist
            if p1 < p2 < p3 and (p3 - p1) >= self.UMBRAL_INCREMENTO_FUGA:
                self.creencias["anomalia_fuga_memoria"] = True

        # 4) Reconsideración dinámica de planes
        fallidos: List[str] = self.creencias["planes_fallidos"]
        if self.creencias["cuota_cloud_agotada"]:
            plan = self.plan_activo
            if plan is not None and plan.requiere_cuota and plan.nombre not in fallidos:
                fallidos.append(plan.nombre)
        else:
            fallidos.clear()   # la cuota se restableció: los planes vuelven a ser viables

    # ------------------------------------------------------------------
    # options: Generación de deseos según creencias
    # ------------------------------------------------------------------
    def options(self) -> Dict[str, Deseo]:
        """
        Genera el conjunto de deseos D a partir de las creencias.
        Jerarquía base:
            MITIGAR_INTRUSION_SEGURIDAD  -> 100
            RESTAURAR_SERVICIO_CACHE     ->  80
            PREVENIR_CAIDA_OOM           ->  75
            ESTABILIZAR_RENDIMIENTO_SLA  ->  60
        """
        post = self.postergaciones
        opciones: Dict[str, Deseo] = {}

        if self.creencias.get("bajo_ataque", False):
            opciones["MITIGAR_INTRUSION_SEGURIDAD"] = Deseo(
                "MITIGAR_INTRUSION_SEGURIDAD", 100,
                post.get("MITIGAR_INTRUSION_SEGURIDAD", 0))

        if self.creencias.get("latencia_alta", False):
            opciones["RESTAURAR_SERVICIO_CACHE"] = Deseo(
                "RESTAURAR_SERVICIO_CACHE", 80,
                post.get("RESTAURAR_SERVICIO_CACHE", 0))

        if self.creencias.get("anomalia_fuga_memoria", False):
            opciones["PREVENIR_CAIDA_OOM"] = Deseo(
                "PREVENIR_CAIDA_OOM", 75,
                post.get("PREVENIR_CAIDA_OOM", 0))

        # Homeostasis: deseo de fondo siempre presente (no envejece).
        opciones["ESTABILIZAR_RENDIMIENTO_SLA"] = Deseo(
            "ESTABILIZAR_RENDIMIENTO_SLA", 60, envejece=False)

        return opciones

    # ------------------------------------------------------------------
    # filter: Selección de intención (mente abierta)
    # ------------------------------------------------------------------
    def filter(self, opciones: Dict[str, Deseo]) -> Optional[Deseo]:
        """
        Selecciona la intención de mayor prioridad efectiva. Al recalcularse
        en cada ciclo, el compromiso es 'de mente abierta'.
        """
        if not opciones:
            return None
        return max(opciones.values(), key=lambda d: d.prioridad_efectiva)

    def _actualizar_postergaciones(self, opciones: Dict[str, Deseo],
                                   intencion: Optional[Deseo]) -> None:
        """Cuenta los ciclos que cada deseo lleva activo sin ser intención."""
        nuevas: Dict[str, int] = {}
        for nombre, deseo in opciones.items():
            if not deseo.envejece:
                continue
            if intencion is not None and nombre == intencion.nombre:
                nuevas[nombre] = 0
            else:
                nuevas[nombre] = self.postergaciones.get(nombre, 0) + 1
        self.postergaciones = nuevas

    # ------------------------------------------------------------------
    # Selección de plan con Fallback Plans
    # ------------------------------------------------------------------
    def seleccionar_plan_fallback(
        self, intencion: Deseo, biblioteca: List[PlanBDI]
    ) -> Optional[PlanBDI]:
        """
        Retorna el primer plan aplicable para la intención dada cuya
        precondición se cumpla y que NO esté en 'planes_fallidos'.
        """
        fallidos = self.creencias.get("planes_fallidos", [])

        for plan in biblioteca:
            if plan.intencion_meta != intencion.nombre:
                continue
            if plan.nombre in fallidos:
                continue
            try:
                if plan.precondicion(self.creencias):
                    return plan
            except Exception:
                # Una precondición mal formada no debe romper el agente.
                continue
        return None

    # ------------------------------------------------------------------
    # Ciclo BDI completo (intérprete de Wooldridge)
    # ------------------------------------------------------------------
    def ejecutar_ciclo(
        self,
        telemetria: TelemetriaCluster,
        biblioteca: List[PlanBDI],
    ) -> List[str]:
        """
        brf -> options -> filter -> plan (con fallback) -> execute
        Devuelve la cola de acciones del plan activo.
        """
        self.brf(telemetria)
        opciones = self.options()
        self.intencion_actual = self.filter(opciones)
        self._actualizar_postergaciones(opciones, self.intencion_actual)

        self.plan_activo = None
        if self.intencion_actual is not None:
            self.plan_activo = self.seleccionar_plan_fallback(
                self.intencion_actual, biblioteca
            )

        self.cola_acciones = (
            list(self.plan_activo.cuerpo_acciones) if self.plan_activo else []
        )
        return self.cola_acciones


# ---------------------------------------------------------------------------
# Biblioteca de planes Π
# ---------------------------------------------------------------------------
def construir_biblioteca_planes() -> List[PlanBDI]:
    """
    Biblioteca de runbooks del agente SRE.

    Para ESTABILIZAR_RENDIMIENTO_SLA se registran DOS planes:
      - Principal: Runbook_Autoescalado_Horizontal
                   (precondición: hay cuota cloud; requiere_cuota=True)
      - Fallback:  Runbook_Degradacion_Elegante_CircuitBreaker
                   (precondición: cuota cloud agotada)
    """
    return [
        PlanBDI(
            nombre="Runbook_Autoescalado_Horizontal",
            intencion_meta="ESTABILIZAR_RENDIMIENTO_SLA",
            precondicion=lambda c: not c.get("cuota_cloud_agotada", False),
            cuerpo_acciones=[
                "APROVISIONAR_NODO_REPLICA_K8S",
                "REBALANCEAR_TRAFICO_INGRESS",
            ],
            requiere_cuota=True,
        ),
        PlanBDI(
            nombre="Runbook_Degradacion_Elegante_CircuitBreaker",
            intencion_meta="ESTABILIZAR_RENDIMIENTO_SLA",
            precondicion=lambda c: c.get("cuota_cloud_agotada", False),
            cuerpo_acciones=[
                "ACTIVAR_CIRCUIT_BREAKER_RECOMENDACIONES",
                "HABILITAR_RESPUESTA_ESTATICA_CDN",
            ],
        ),
        PlanBDI(
            nombre="Runbook_Aislar_Nodo_Comprometido",
            intencion_meta="MITIGAR_INTRUSION_SEGURIDAD",
            precondicion=lambda c: c.get("bajo_ataque", False),
            cuerpo_acciones=[
                "AISLAR_POD_SOSPECHOSO",
                "ROTAR_CREDENCIALES_SECRETOS",
            ],
        ),
        PlanBDI(
            nombre="Runbook_Purga_Cache_Distribuida",
            intencion_meta="RESTAURAR_SERVICIO_CACHE",
            precondicion=lambda c: c.get("latencia_alta", False),
            cuerpo_acciones=[
                "PURGAR_CACHE_REDIS",
                "PRECALENTAR_CACHE_CDN",
            ],
        ),
        PlanBDI(
            nombre="Runbook_Mitigar_Fuga_Memoria",
            intencion_meta="PREVENIR_CAIDA_OOM",
            precondicion=lambda c: c.get("anomalia_fuga_memoria", False),
            cuerpo_acciones=[
                "REINICIAR_PODS_CON_MAYOR_CONSUMO",
                "AJUSTAR_LIMITES_MEMORIA_K8S",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Simulación principal
# ---------------------------------------------------------------------------
def _correr(titulo, lecturas, biblioteca):
    print(f"\n=== {titulo} ===")
    agente = AgenteBDI_Estudiante()
    for tel in lecturas:
        print(f"\n--- Ciclo {tel.ciclo} "
              f"(CPU: {tel.cpu_pct}%, Latencia: {tel.latencia_ms}ms, "
              f"SIEM: {tel.alerta_siem}, Cuota agotada: {tel.cuota_agotada}) ---")
        acciones = agente.ejecutar_ciclo(tel, biblioteca)
        c = agente.creencias
        print(f"  Creencias: fuga_memoria={c['anomalia_fuga_memoria']}, "
              f"latencia_alta={c['latencia_alta']}, "
              f"bajo_ataque={c['bajo_ataque']}, "
              f"cuota_agotada={c['cuota_cloud_agotada']}, "
              f"planes_fallidos={c['planes_fallidos']}")
        i = agente.intencion_actual
        print(f"  Intención : {i.nombre if i else None}"
              f" (prioridad efectiva {i.prioridad_efectiva if i else '-'})")
        print(f"  Plan      : {agente.plan_activo.nombre if agente.plan_activo else None}")
        print(f"  Acciones  : {acciones}")


if __name__ == "__main__":
    biblioteca_planes = construir_biblioteca_planes()

    # Escenario 1: CPU [40, 58, 75] -> fuga; cuota agotada; alerta SIEM.
    _correr("EVALUACIÓN EJERCICIO 1: AGENTE BDI", [
        TelemetriaCluster(1, 40.0, 120.0, False, False),
        TelemetriaCluster(2, 58.0, 150.0, False, False),
        TelemetriaCluster(3, 75.0, 180.0, False, False),
        TelemetriaCluster(4, 82.0, 210.0, False, True),
        TelemetriaCluster(5, 85.0, 230.0, True, True),
    ], biblioteca_planes)

    # Escenario 2: cuota agotada sostenida con el SLA como intención:
    # debe conmutar al plan fallback y volver al principal al restablecerse.
    _correr("ESCENARIO 2: CUOTA AGOTADA SOSTENIDA", [
        TelemetriaCluster(1, 35.0, 120.0, False, False),
        TelemetriaCluster(2, 36.0, 130.0, False, True),
        TelemetriaCluster(3, 34.0, 125.0, False, True),
        TelemetriaCluster(4, 33.0, 120.0, False, False),
    ], biblioteca_planes)
