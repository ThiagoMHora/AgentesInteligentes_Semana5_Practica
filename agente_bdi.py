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


  1. La creencia 'anomalia_fuga_memoria' es MONÓTONA: una vez inferida,
     no se resetea (una fuga de memoria no se auto-cura).
  2. El deseo RESTAURAR_SERVICIO_CACHE se condiciona a una creencia
     ('latencia_alta'), no se genera incondicionalmente.
  3. La conmutación al plan fallback se decide DENTRO del agente (en brf),
     no en el bloque main. El main solo orquesta.
  4. Se encapsula el ciclo BDI completo en 'ejecutar_ciclo'.
  5. Se guarda self.plan_activo y self.cola_acciones para dar continuidad
     entre ciclos.
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


@dataclass
class PlanBDI:
    nombre: str
    intencion_meta: str
    precondicion: Callable[[Dict[str, Any]], bool]
    cuerpo_acciones: List[str]


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
            "plan_fallido": None,
        }
        self.intencion_actual: Optional[Deseo] = None
        self.plan_activo: Optional[PlanBDI] = None
        self.cola_acciones: List[str] = []

    # ------------------------------------------------------------------
    # TAREA 1 — brf: Revisión de creencias con inferencia deductiva
    # ------------------------------------------------------------------
    def brf(self, p: TelemetriaCluster) -> None:
        """
        Actualiza creencias a partir de la telemetría p.

        Regla deductiva: si las últimas 3 lecturas de CPU son
        estrictamente crecientes y (p3 - p1) >= 15.0, se deriva
        proactivamente 'anomalia_fuga_memoria = True'.

        Decisión de diseño: la creencia derivada es MONÓTONA (no se
        resetea). Una fuga de memoria solo se repara con una acción
        (reinicio del pod), no por una bajada puntual de la CPU.
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

        # 4) Reconsideración dinámica: si la cuota cloud se agotó y hay
        #    un plan activo, marcarlo como fallido para forzar el fallback
        #    en la selección de plan del ciclo actual.
        if self.creencias["cuota_cloud_agotada"] and self.plan_activo is not None:
            self.creencias["plan_fallido"] = self.plan_activo.nombre

    # ------------------------------------------------------------------
    # TAREA 2 — options: Generación de deseos según creencias
    # ------------------------------------------------------------------
    def options(self) -> Dict[str, Deseo]:
        """
        Genera el conjunto de deseos D a partir de las creencias.
        Jerarquía exigida:
            MITIGAR_INTRUSION_SEGURIDAD  -> 100
            RESTAURAR_SERVICIO_CACHE     ->  80
            PREVENIR_CAIDA_OOM           ->  75
            ESTABILIZAR_RENDIMIENTO_SLA  ->  60
        """
        opciones: Dict[str, Deseo] = {}

        # Seguridad: solo si hay alerta SIEM activa.
        if self.creencias.get("bajo_ataque", False):
            opciones["MITIGAR_INTRUSION_SEGURIDAD"] = Deseo(
                "MITIGAR_INTRUSION_SEGURIDAD", 100
            )

        # Caché: solo si la latencia es alta (creencia derivada en brf).
        if self.creencias.get("latencia_alta", False):
            opciones["RESTAURAR_SERVICIO_CACHE"] = Deseo(
                "RESTAURAR_SERVICIO_CACHE", 80
            )

        # Prevención OOM: solo si se infirió fuga de memoria.
        if self.creencias.get("anomalia_fuga_memoria", False):
            opciones["PREVENIR_CAIDA_OOM"] = Deseo(
                "PREVENIR_CAIDA_OOM", 75
            )

        # Homeostasis: deseo de fondo siempre presente.
        opciones["ESTABILIZAR_RENDIMIENTO_SLA"] = Deseo(
            "ESTABILIZAR_RENDIMIENTO_SLA", 60
        )

        return opciones

    # ------------------------------------------------------------------
    # TAREA 3 — filter: Selección de intención (mente abierta)
    # ------------------------------------------------------------------
    def filter(self, opciones: Dict[str, Deseo]) -> Optional[Deseo]:
        """
        Selecciona la intención de mayor prioridad. Al recalcularse en
        cada ciclo, el compromiso es 'de mente abierta': el agente puede
        cambiar de foco si aparece un deseo más prioritario.
        """
        if not opciones:
            return None
        return max(opciones.values(), key=lambda d: d.prioridad)

    # ------------------------------------------------------------------
    # TAREA 4 — Selección de plan con Fallback Plans
    # ------------------------------------------------------------------
    def seleccionar_plan_fallback(
        self, intencion: Deseo, biblioteca: List[PlanBDI]
    ) -> Optional[PlanBDI]:
        """
        Retorna el primer plan aplicable para la intención dada cuya
        precondición se cumpla y que NO esté marcado como fallido.

        Esto implementa el Fallback Plan: si el plan principal falla
        (p. ej. cuota_cloud_agotada), se conmuta al alternativo sin
        abandonar la Intención.
        """
        plan_fallido = self.creencias.get("plan_fallido")

        for plan in biblioteca:
            if plan.intencion_meta != intencion.nombre:
                continue
            if plan_fallido is not None and plan.nombre == plan_fallido:
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
        Ejecuta un ciclo BDI completo:
            brf -> options -> filter -> plan (con fallback) -> execute
        Devuelve la cola de acciones del plan activo.
        """
        # 1) Revisión de creencias
        self.brf(telemetria)

        # 2) Generación de opciones
        opciones = self.options()

        # 3) Filtrado de intenciones
        self.intencion_actual = self.filter(opciones)

        # 4) Selección de plan con fallback
        self.plan_activo = None
        if self.intencion_actual is not None:
            self.plan_activo = self.seleccionar_plan_fallback(
                self.intencion_actual, biblioteca
            )

        # 5) Ejecución (encolar acciones)
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
                   (precondición: hay cuota cloud)
      - Fallback:  Runbook_Degradacion_Elegante_CircuitBreaker
                   (precondición: cuota cloud agotada)
    Así, cuando 'cuota_cloud_agotada == True', el agente conmuta solo.
    """
    return [
        # --- Plan principal: Autoescalado horizontal ---
        PlanBDI(
            nombre="Runbook_Autoescalado_Horizontal",
            intencion_meta="ESTABILIZAR_RENDIMIENTO_SLA",
            precondicion=lambda c: not c.get("cuota_cloud_agotada", False),
            cuerpo_acciones=[
                "APROVISIONAR_NODO_REPLICA_K8S",
                "REBALANCEAR_TRAFICO_INGRESS",
            ],
        ),
        # --- Plan fallback: Degradación elegante con Circuit Breaker ---
        PlanBDI(
            nombre="Runbook_Degradacion_Elegante_CircuitBreaker",
            intencion_meta="ESTABILIZAR_RENDIMIENTO_SLA",
            precondicion=lambda c: c.get("cuota_cloud_agotada", False),
            cuerpo_acciones=[
                "ACTIVAR_CIRCUIT_BREAKER_RECOMENDACIONES",
                "HABILITAR_RESPUESTA_ESTATICA_CDN",
            ],
        ),
        # --- Planes auxiliares coherentes con el resto de deseos ---
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
if __name__ == "__main__":
    print("=== EVALUACIÓN EJERCICIO 1: AGENTE BDI ===")

    agente = AgenteBDI_Estudiante()
    biblioteca_planes = construir_biblioteca_planes()

    # Escenario 1: CPU [40, 58, 75] -> debe inferirse fuga de memoria.
    # Escenario 2: cuota agotada -> debe conmutar al plan fallback.
    # Escenario 3: alerta SIEM -> debe priorizar seguridad (100).
    lecturas = [
        TelemetriaCluster(ciclo=1, cpu_pct=40.0, latencia_ms=120.0,
                          alerta_siem=False, cuota_agotada=False),
        TelemetriaCluster(ciclo=2, cpu_pct=58.0, latencia_ms=150.0,
                          alerta_siem=False, cuota_agotada=False),
        TelemetriaCluster(ciclo=3, cpu_pct=75.0, latencia_ms=180.0,
                          alerta_siem=False, cuota_agotada=False),
        TelemetriaCluster(ciclo=4, cpu_pct=82.0, latencia_ms=210.0,
                          alerta_siem=False, cuota_agotada=True),
        TelemetriaCluster(ciclo=5, cpu_pct=85.0, latencia_ms=230.0,
                          alerta_siem=True,  cuota_agotada=True),
    ]

    for tel in lecturas:
        print(f"\n--- Ciclo {tel.ciclo} "
              f"(CPU: {tel.cpu_pct}%, Latencia: {tel.latencia_ms}ms, "
              f"SIEM: {tel.alerta_siem}, Cuota agotada: {tel.cuota_agotada}) ---")

        acciones = agente.ejecutar_ciclo(tel, biblioteca_planes)

        print(f"  Creencias: fuga_memoria={agente.creencias['anomalia_fuga_memoria']}, "
              f"latencia_alta={agente.creencias['latencia_alta']}, "
              f"bajo_ataque={agente.creencias['bajo_ataque']}, "
              f"cuota_agotada={agente.creencias['cuota_cloud_agotada']}, "
              f"plan_fallido={agente.creencias['plan_fallido']}")
        print(f"  Intención : {agente.intencion_actual.nombre if agente.intencion_actual else None}")
        print(f"  Plan      : {agente.plan_activo.nombre if agente.plan_activo else None}")
        print(f"  Acciones  : {acciones}")