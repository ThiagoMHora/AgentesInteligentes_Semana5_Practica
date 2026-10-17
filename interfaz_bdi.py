
import copy
import importlib
from math import ceil
import tkinter as tk
from tkinter import ttk

MODULO_AGENTE = "agente_bdi" 

_m = importlib.import_module(MODULO_AGENTE)
AgenteBDI_Estudiante = _m.AgenteBDI_Estudiante
TelemetriaCluster = _m.TelemetriaCluster
construir_biblioteca_planes = _m.construir_biblioteca_planes


BG = "#F6F5FC"; SURF = "#FFFFFF"; FG = "#2A2850"; MUTED = "#5E5B80"; LINE = "#DDD9F0"
ACC = "#5B4FB8"; ACC_BG = "#E6E0FA"
Z1 = "#EAF3FC"; Z2 = "#F1EDFC"; Z3 = "#E8F6EF"
TONOS = {                       # (color fuerte, fondo pastel)
    "sec": ("#9C3248", "#FADCE2"),
    "cache": ("#854A07", "#FFE9C9"),
    "oom": ("#176148", "#D5F1E4"),
    "sla": ("#1F5693", "#D8E8FB"),
}
OK = ("#176148", "#D5F1E4"); WARN = ("#854A07", "#FFE9C9"); BAD = ("#9C3248", "#FADCE2")
F = "Segoe UI"; FM = "Consolas"


DESEOS = [  # nombre, prioridad, tono, etiqueta corta, creencia que lo activa
    ("MITIGAR_INTRUSION_SEGURIDAD", 100, "sec", "Seguridad", "bajo_ataque"),
    ("RESTAURAR_SERVICIO_CACHE", 80, "cache", "Caché", "latencia_alta"),
    ("PREVENIR_CAIDA_OOM", 75, "oom", "OOM", "anomalia_fuga_memoria"),
    ("ESTABILIZAR_RENDIMIENTO_SLA", 60, "sla", "SLA", None),
]
CORTO = {d[0]: d[3] for d in DESEOS}
TONO = {d[0]: d[2] for d in DESEOS}
PRECOND = {
    "Runbook_Autoescalado_Horizontal": ("cuota_cloud_agotada", False),
    "Runbook_Degradacion_Elegante_CircuitBreaker": ("cuota_cloud_agotada", True),
    "Runbook_Aislar_Nodo_Comprometido": ("bajo_ataque", True),
    "Runbook_Purga_Cache_Distribuida": ("latencia_alta", True),
    "Runbook_Mitigar_Fuga_Memoria": ("anomalia_fuga_memoria", True),
}
CREENCIAS = ["anomalia_fuga_memoria", "bajo_ataque", "cuota_cloud_agotada",
             "latencia_alta", "plan_fallido"]
ETAPAS = [
    ("Percibir", "B — Creencias · lo que el agente sabe", "brf(B, telemetría)", "Percibiendo la telemetría…"),
    ("Deliberar", "D — Deseos · lo que podría querer", "options(B) → D", "Generando deseos…"),
    ("Decidir", "I — Intención · lo que decide querer ahora", "filter(B, D) → I", "Seleccionando la intención…"),
    ("Planificar y actuar", "π — Plan · cómo lo va a lograr", "plan(B, I, Π) → π", "Eligiendo y ejecutando el plan…"),
]
ESCENARIOS = [
    ("codigo", "Tu simulación (5 ciclos)"),
    ("cuota", "Cuota agotada sostenida"),
    ("siem", "Alerta de seguridad que pasa"),
]
PRESETS = {  # cpu, latencia, alerta SIEM, cuota agotada
    "codigo": [(40, 120, 0, 0), (58, 150, 0, 0), (75, 180, 0, 0), (82, 210, 0, 1), (85, 230, 1, 1)],
    "cuota": [(35, 120, 0, 0), (36, 130, 0, 1), (34, 125, 0, 1), (33, 120, 0, 0)],
    "siem": [(30, 100, 0, 0), (31, 110, 1, 0), (32, 250, 0, 0), (30, 120, 0, 0)],
}
GLOSARIO = [
    ("Creencias (B)", "Lo que el agente sabe del mundo: CPU, latencia, alertas y cuota."),
    ("Deseos (D)", "Metas posibles, ordenadas por prioridad. Mayor número, más urgente."),
    ("Intención (I)", "El deseo que el agente decide perseguir ahora."),
    ("Plan (π)", "La lista de acciones con la que intenta lograr la intención."),
    ("Plan alternativo (fallback)", "El que se usa cuando el principal no es viable, sin abandonar la intención."),
    ("Plan fallido", "Un plan que el agente marcó como inviable y descarta al elegir."),
    ("Fuga de memoria", "Se deduce cuando la CPU sube sin pausa y suma al menos 15 puntos en tres lecturas."),
]


def crear(clave):
    return [{"cpu": c, "lat": l, "siem": bool(s), "cuota": bool(q)} for c, l, s, q in PRESETS[clave]]



def simular(lista):
    agente = AgenteBDI_Estudiante()
    biblioteca = construir_biblioteca_planes()
    trazas = []
    int_prev = None
    for i, t in enumerate(lista):
        antes = copy.deepcopy(agente.creencias)
        plan_prev = agente.plan_activo

        # Se "espía" options() solo para guardar el conjunto de deseos generado.
        capturado = {}
        original = agente.options

        def espiar(_orig=original, _cap=capturado):
            d = _orig()
            _cap["deseos"] = d
            return d

        agente.options = espiar
        tel = TelemetriaCluster(ciclo=i + 1, cpu_pct=float(t["cpu"]), latencia_ms=float(t["lat"]),
                                alerta_siem=bool(t["siem"]), cuota_agotada=bool(t["cuota"]))
        acciones = agente.ejecutar_ciclo(tel, biblioteca)     
        del agente.options

        despues = copy.deepcopy(agente.creencias)
        hist = despues["historial_cpu"]
        completa = len(hist) == 3
        mono, delta, cumple = False, None, False
        if completa:
            p1, p2, p3 = hist
            mono = p1 < p2 < p3
            delta = p3 - p1
            cumple = mono and delta >= agente.UMBRAL_INCREMENTO_FUGA
        marcado = plan_prev.nombre if (despues["cuota_cloud_agotada"] and plan_prev is not None) else None

        intencion = agente.intencion_actual
        plan = agente.plan_activo
        cand, visto = [], False
        for p in biblioteca:
            if p.intencion_meta != intencion.nombre:
                continue
            if plan is not None and p is plan:
                v = "ok"
                visto = True
            elif visto:
                v = "skip"
            elif despues["plan_fallido"] is not None and p.nombre == despues["plan_fallido"]:
                v = "failed"
            else:
                v = "pre"
            cand.append((p, v))

        trazas.append({
            "t": t, "antes": antes, "despues": despues, "deseos": capturado["deseos"],
            "intencion": intencion, "int_prev": int_prev, "plan": plan, "acciones": acciones,
            "cand": cand, "completa": completa, "mono": mono, "delta": delta, "cumple": cumple,
            "marcado": marcado,
        })
        int_prev = intencion
    return trazas


def si_no(v):
    return "True" if v else "False"


def narrar(r):
    c = r["despues"]
    if c["bajo_ataque"]:
        return "Alerta SIEM. La seguridad tiene prioridad 100 y desplaza todo lo demás."
    if r["plan"] is None:
        return "Ningún plan aplicable. El agente queda a la espera."
    if r["marcado"]:
        return "La cuota cloud se agotó con un plan en marcha: el agente lo marca como fallido y sigue con lo más urgente."
    if r["cumple"] and not r["antes"]["anomalia_fuga_memoria"]:
        return "Fuga inferida. El agente actúa antes de que la CPU llegue al límite."
    if c["anomalia_fuga_memoria"]:
        return "La fuga sigue inferida, pero ahora hay algo más urgente que atender."
    if r["completa"] and r["mono"]:
        return "La CPU sube, pero aún sin evidencia suficiente."
    return "Todo tranquilo. El agente vigila."


def resumen(r):
    c = r["despues"]
    if c["bajo_ataque"]:
        return "alerta SIEM, prioridad 100"
    if r["plan"] is None:
        return "sin plan aplicable"
    if r["cumple"] and not r["antes"]["anomalia_fuga_memoria"]:
        return "fuga inferida"
    if r["marcado"]:
        return "plan marcado como fallido"
    return "intención " + CORTO[r["intencion"].nombre]


def razonamiento(r):
    c, t = r["despues"], r["t"]
    hist = c["historial_cpu"]
    L = []
    if not r["completa"]:
        p = f"Ventana de CPU incompleta ({len(hist)} de 3 lecturas): todavía no hay evidencia de fuga."
    elif r["cumple"]:
        p = (f"Ventana de CPU {hist}: sube sin pausa y suma +{r['delta']:g} puntos "
             f"(el mínimo es 15). Se infiere fuga de memoria.")
    elif c["anomalia_fuga_memoria"]:
        p = f"Ventana de CPU {hist}: la regla ya no se cumple, pero la fuga sigue inferida porque esa creencia nunca se resetea."
    else:
        det = (f"sube solo +{r['delta']:g} puntos, menos de los 15 necesarios."
               if r["mono"] else "no sube en las tres lecturas seguidas.")
        p = f"Ventana de CPU {hist}: {det} No se infiere fuga."
    p += (f" Latencia de {t['lat']} ms ({'por encima' if c['latencia_alta'] else 'dentro'} del umbral de 200 ms), "
          f"alerta SIEM {'activa' if c['bajo_ataque'] else 'apagada'} y cuota cloud "
          f"{'agotada' if c['cuota_cloud_agotada'] else 'disponible'}.")
    L.append(("PERCIBIR", p, False))

    activos = [d for d in DESEOS if d[0] in r["deseos"]]
    txt = "Deseos activos: " + ", ".join(f"{d[3]} ({d[1]})" for d in activos) + "."
    if len(activos) < 4:
        txt += " Los demás no se generan porque su creencia es False."
    L.append(("DELIBERAR", txt, False))

    nombre = r["intencion"].nombre
    otros = [d for d in activos if d[0] != nombre]
    if otros:
        s = (f"Gana {CORTO[nombre]} porque su prioridad ({r['intencion'].prioridad}) es la más alta "
             f"entre los activos; la siguiente es {otros[0][1]}.")
    else:
        s = f"{CORTO[nombre]} es el único deseo activo, así que es la intención."
    if r["int_prev"] is not None and r["int_prev"].nombre != nombre:
        s += f" El agente cambia de foco: antes perseguía {CORTO[r['int_prev'].nombre]}."
    L.append(("DECIDIR", s, False))

    if r["plan"] is not None:
        a = f"Ejecuta {r['plan'].nombre}: {', '.join(r['acciones'])}."
        if r["cand"] and r["cand"][0][1] != "ok":
            a += " Es el plan alternativo: el principal no era viable."
    else:
        a = "Ningún plan aplicable. El agente queda a la espera."
        partes = []
        for p, v in r["cand"]:
            if v == "failed":
                partes.append(f"{p.nombre} está marcado como fallido")
            else:
                k, w = PRECOND.get(p.nombre, ("su precondición", True))
                partes.append(f"{p.nombre} exige {k} = {si_no(w)}")
        if partes:
            a += " " + "; ".join(partes) + "."
    L.append(("ACTUAR", a, False))

    if r["marcado"]:
        txt = (f"La cuota está agotada y el plan activo del ciclo anterior era {r['marcado']}, "
               f"así que brf lo marcó como fallido.")
        # ¿pertenece a otra meta distinta de la intención actual?
        pert = next((d for d in _m.construir_biblioteca_planes() if d.nombre == r["marcado"]), None)
        if pert is not None and pert.intencion_meta != nombre:
            txt += " Ese plan pertenece a otra meta, y seguirá descartado mientras plan_fallido no cambie."
        L.append(("ATENCIÓN", txt, True))
    elif c["plan_fallido"]:
        L.append(("ATENCIÓN", f"{c['plan_fallido']} sigue marcado como fallido de un ciclo anterior. "
                              f"El código nunca limpia plan_fallido.", True))
    return L


# ---------------------------------------------------------------------------
# Piezas de interfaz
# ---------------------------------------------------------------------------
def tarjeta(parent, bg=SURF):
    return tk.Frame(parent, bg=bg, highlightthickness=1, highlightbackground=LINE)


def boton(parent, texto, cmd, primario=False):
    borde = ACC if primario else LINE
    b = tk.Label(parent, text=texto, bg=ACC_BG if primario else SURF, fg=ACC if primario else FG,
                 font=(F, 10, "bold" if primario else "normal"), padx=12, pady=6, cursor="hand2",
                 highlightthickness=1, highlightbackground=borde)
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>", lambda e: b.configure(highlightbackground=ACC))
    b.bind("<Leave>", lambda e: b.configure(highlightbackground=borde))
    return b


def semaforo(parent, nivel, simbolo, texto, wrap=0):
    fg, fondo = {"ok": OK, "warn": WARN, "bad": BAD}[nivel]
    return tk.Label(parent, text=f"{simbolo} {texto}", fg=fg, bg=fondo, font=(FM, 9),
                    padx=8, pady=2, justify="left", wraplength=wrap)


class Grafico(tk.Canvas):
    """Gráfica de línea por ciclo. Se redibuja sola al cambiar de tamaño."""

    def __init__(self, master):
        super().__init__(master, height=140, bg=SURF, highlightthickness=0)
        self.args = None
        self.puntos = []
        self.alclic = None
        self.bind("<Configure>", lambda e: self.dibujar())
        self.bind("<Button-1>", self._clic)

    def poner(self, vals, sel, ymin, paso, umbral=None, unidad="", banda=False, alclic=None):
        self.args = (vals, sel, ymin, paso, umbral, unidad, banda)
        self.alclic = alclic
        self.dibujar()

    def _clic(self, e):
        if not self.alclic:
            return
        for x, i in self.puntos:
            if abs(e.x - x) <= 14:
                self.alclic(i)
                return

    def dibujar(self):
        self.delete("all")
        self.puntos = []
        if not self.args:
            return
        vals, sel, ymin, paso, umbral, unidad, banda = self.args
        W = self.winfo_width()
        H = self.winfo_height()
        W = W if W > 60 else 300
        H = H if H > 60 else 140
        L, R, T, B = 36, 14, 12, 24
        iw, ih, n = W - L - R, H - T - B, len(vals)
        ymax = ceil(max(vals + [ymin]) / paso) * paso

        def X(i):
            return L + iw / 2 if n == 1 else L + i * iw / (n - 1)

        def Y(v):
            return T + ih - (v / ymax) * ih

        if banda:
            a = max(0, sel - 2)
            self.create_rectangle(X(a) - 10, T, X(sel) + 10, T + ih, fill=ACC_BG, outline="")
        for g in (0, ymax):
            self.create_line(L, Y(g), W - R, Y(g), fill=LINE)
            self.create_text(L - 6, Y(g), text=str(g), anchor="e", fill=MUTED, font=(FM, 8))
        if umbral is not None:
            self.create_line(L, Y(umbral), W - R, Y(umbral), fill=WARN[0], dash=(4, 3))
            self.create_text(W - R, Y(umbral) - 7, text=f"{umbral} {unidad}", anchor="e",
                             fill=WARN[0], font=(FM, 8))
        if n > 1:
            self.create_line(*[c for i, v in enumerate(vals) for c in (X(i), Y(v))],
                             fill=LINE, width=3)
        if sel > 0:
            self.create_line(*[c for i, v in enumerate(vals[:sel + 1]) for c in (X(i), Y(v))],
                             fill=ACC, width=3)
        for i, v in enumerate(vals):
            r = 6 if i == sel else 4
            self.create_oval(X(i) - r, Y(v) - r, X(i) + r, Y(v) + r,
                             fill=ACC if i == sel else SURF,
                             outline=ACC if i <= sel else LINE, width=2)
            self.create_text(X(i), H - 8, text=str(i + 1), fill=MUTED, font=(FM, 8))
            self.puntos.append((X(i), i))


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador del Agente BDI")
        self.geometry("1300x900")
        self.minsize(1000, 700)
        self.configure(bg=BG)
        self.tels = crear("codigo")
        self.sel = 2
        self.ex = 0                # 0 = vista completa, 1..4 = modo explicación
        self.timer = None
        self.toast_id = None
        self.rr = None
        self.traza = []
        self.W = 1280
        self.show_regla = False
        self.show_desc = False
        self._relleno = False
        self._construir()
        self.seleccionar(2)

    # ---- helpers de texto -------------------------------------------------
    @property
    def wc(self):
        return max(170, int((self.W - 170) / 4) - 44)

    def t(self, parent, texto, fg=FG, bg=SURF, font=None, w=None):
        return tk.Label(parent, text=texto, fg=fg, bg=bg, font=font or (F, 10), justify="left",
                        anchor="w", wraplength=w or self.wc)

    # ---- construcción ---------------------------------------------------
    def _construir(self):
        self.lienzo = tk.Canvas(self, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.lienzo.yview)
        self.lienzo.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.lienzo.pack(side="left", fill="both", expand=True)
        self.cuerpo = tk.Frame(self.lienzo, bg=BG)
        self.ventana = self.lienzo.create_window((0, 0), window=self.cuerpo, anchor="nw")
        self.cuerpo.bind("<Configure>", lambda e: self.lienzo.configure(scrollregion=self.lienzo.bbox("all")))
        self.lienzo.bind("<Configure>", self._al_redimensionar)
        self.bind_all("<MouseWheel>", lambda e: self.lienzo.yview_scroll(-3 if e.delta > 0 else 3, "units"))
        self.bind_all("<Button-4>", lambda e: self.lienzo.yview_scroll(-3, "units"))
        self.bind_all("<Button-5>", lambda e: self.lienzo.yview_scroll(3, "units"))

        m = tk.Frame(self.cuerpo, bg=BG)
        m.pack(fill="both", expand=True, padx=24, pady=18)

        # Encabezado
        enc = tk.Frame(m, bg=BG)
        enc.pack(fill="x")
        izq = tk.Frame(enc, bg=BG)
        izq.pack(side="left")
        tk.Label(izq, text="Simulador del Agente BDI", font=(F, 22, "bold"), fg=FG, bg=BG).pack(anchor="w")
        tk.Label(izq, text="Un agente SRE que vigila microservicios de e-commerce. "
                           "Mira cómo percibe, delibera, decide y actúa en cada ciclo.",
                 font=(F, 10), fg=MUTED, bg=BG).pack(anchor="w")
        der = tk.Frame(enc, bg=BG)
        der.pack(side="right")
        self.cb = ttk.Combobox(der, values=[e[1] for e in ESCENARIOS], state="readonly", width=30)
        self.cb.current(0)
        self.cb.bind("<<ComboboxSelected>>", self._cambiar_escenario)
        self.cb.pack(side="left", padx=8)
        boton(der, "? Glosario", self._ir_glosario).pack(side="left")

        # Bienvenida
        self.hola = tk.Frame(m, bg=ACC_BG)
        self.hola.pack(fill="x", pady=(14, 0))
        tk.Label(self.hola, bg=ACC_BG, fg=FG, font=(F, 10), anchor="w", justify="left", wraplength=900,
                 text="Esta es la telemetría del clúster. Cambia los valores de la zona 1 o pulsa "
                      "«Reproducir historia» para ver, paso a paso, cómo delibera el agente."
                 ).pack(side="left", padx=14, pady=10)
        boton(self.hola, "Entendido", lambda: self.hola.pack_forget()).pack(side="right", padx=10, pady=6)

        # Controles
        barra = tk.Frame(m, bg=BG)
        barra.pack(fill="x", pady=(14, 6))
        self.b_hist = boton(barra, "Reproducir historia", self.alternar_historia, True)
        self.b_hist.pack(side="left", padx=(0, 6))
        self.b_exp = boton(barra, "Modo explicación", self.paso_explicacion)
        self.b_exp.pack(side="left", padx=4)
        self.b_salir = boton(barra, "Salir", self.salir_explicacion)
        boton(barra, "◀", lambda: self.elegir(self.sel - 1)).pack(side="left", padx=4)
        boton(barra, "▶", lambda: self.elegir(self.sel + 1)).pack(side="left", padx=4)
        self.fila_chips = tk.Frame(m, bg=BG)
        self.fila_chips.pack(fill="x", pady=(0, 6))

        # Zona 1: estado del mundo
        z1 = self.zona(m, 1, "Estado del mundo", "Lo que pasa en el clúster en este ciclo. Puedes cambiarlo.", Z1)
        z1.columnconfigure(1, weight=1)
        ed = tarjeta(z1)
        ed.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        tk.Label(ed, text="TELEMETRÍA DEL CICLO", bg=SURF, fg=MUTED, font=(F, 8, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        self.l_ed = tk.Label(ed, bg=SURF, fg=MUTED, font=(FM, 9))
        self.l_ed.pack(anchor="w", padx=12)
        self.v_cpu, self.v_lat = tk.StringVar(), tk.StringVar()
        self.v_siem, self.v_cuota = tk.BooleanVar(), tk.BooleanVar()
        for texto, var, hasta in (("CPU (%)", self.v_cpu, 100), ("Latencia (ms)", self.v_lat, 2000)):
            fila = tk.Frame(ed, bg=SURF)
            fila.pack(fill="x", padx=12, pady=4)
            tk.Label(fila, text=texto, bg=SURF, fg=FG, font=(F, 10), width=12, anchor="w").pack(side="left")
            sp = tk.Spinbox(fila, from_=0, to=hasta, increment=1 if hasta == 100 else 10, width=7,
                            textvariable=var, command=self._editado, font=(FM, 10), relief="flat",
                            highlightthickness=1, highlightbackground=LINE)
            sp.bind("<KeyRelease>", lambda e: self._editado())
            sp.pack(side="left")
        for texto, var in (("Alerta SIEM", self.v_siem), ("Cuota cloud agotada", self.v_cuota)):
            tk.Checkbutton(ed, text=texto, variable=var, command=self._editado, bg=SURF, fg=FG,
                           activebackground=SURF, font=(F, 10), anchor="w").pack(fill="x", padx=8, pady=2)
        bb = tk.Frame(ed, bg=SURF)
        bb.pack(fill="x", padx=12, pady=(6, 12))
        boton(bb, "+ Ciclo", self._agregar).pack(side="left", padx=(0, 6))
        boton(bb, "Quitar último", self._quitar).pack(side="left")

        dr = tk.Frame(z1, bg=Z1)
        dr.grid(row=0, column=1, sticky="nsew")
        dr.columnconfigure(0, weight=1, uniform="g")
        dr.columnconfigure(1, weight=1, uniform="g")
        self.f_tiles = tk.Frame(dr, bg=Z1)
        self.f_tiles.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for k in range(4):
            self.f_tiles.columnconfigure(k, weight=1, uniform="t")
        c1 = tarjeta(dr)
        c1.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(c1, text="CPU POR CICLO", bg=SURF, fg=MUTED, font=(F, 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.g_cpu = Grafico(c1)
        self.g_cpu.pack(fill="x", padx=6, pady=(0, 6))
        c2 = tarjeta(dr)
        c2.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(c2, text="LATENCIA POR CICLO", bg=SURF, fg=MUTED, font=(F, 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.g_lat = Grafico(c2)
        self.g_lat.pack(fill="x", padx=6, pady=(0, 6))

        # Zona 2: razonamiento
        z2 = self.zona(m, 2, "Razonamiento del agente", "El ciclo BDI es una tubería: cada etapa usa lo que dejó la anterior.", Z2)
        fn = tarjeta(z2)
        fn.pack(fill="x")
        self.t_narr = tk.Label(fn, bg=SURF, fg=FG, font=(F, 12, "bold"), anchor="w")
        self.t_narr.pack(fill="x", padx=14, pady=(10, 0))
        self.s_narr = tk.Label(fn, bg=SURF, fg=MUTED, font=(F, 10), anchor="w", justify="left")
        self.s_narr.pack(fill="x", padx=14, pady=(0, 10))
        self.f_pasos = tk.Frame(z2, bg=Z2)
        self.f_pasos.pack(fill="x", pady=8)
        self.f_flujo = tk.Frame(z2, bg=Z2)
        self.f_flujo.pack(fill="x")

        # Zona 3: evidencia
        z3 = self.zona(m, 3, "Evidencia: por qué decidió así", "Cada número del ciclo seleccionado, explicado en una frase.", Z3)
        self.f_porque = tarjeta(z3)
        self.f_porque.pack(fill="x")
        self.glos = tarjeta(z3)
        self.glos.pack(fill="x", pady=(10, 0))
        tk.Label(self.glos, text="GLOSARIO", bg=SURF, fg=MUTED, font=(F, 8, "bold")).pack(anchor="w", padx=14, pady=(10, 4))
        cuadro = tk.Frame(self.glos, bg=SURF)
        cuadro.pack(fill="x", padx=14, pady=(0, 12))
        cuadro.columnconfigure(0, weight=1, uniform="gl")
        cuadro.columnconfigure(1, weight=1, uniform="gl")
        for k, (term, defi) in enumerate(GLOSARIO):
            cel = tk.Frame(cuadro, bg=SURF)
            cel.grid(row=k // 2, column=k % 2, sticky="nw", padx=(0, 20), pady=4)
            tk.Label(cel, text=term, bg=SURF, fg=FG, font=(F, 10, "bold")).pack(anchor="w")
            tk.Label(cel, text=defi, bg=SURF, fg=MUTED, font=(F, 10), justify="left", wraplength=420).pack(anchor="w")

        self.toast = tk.Label(self, bg=FG, fg=BG, font=(F, 10, "bold"), padx=18, pady=8)

    def zona(self, parent, num, titulo, sub, color):
        z = tk.Frame(parent, bg=color)
        z.pack(fill="x", pady=8)
        top = tk.Frame(z, bg=color)
        top.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(top, text=str(num), bg=SURF, fg=ACC, font=(F, 12, "bold"), width=3).pack(side="left")
        tk.Label(top, text=titulo, bg=color, fg=FG, font=(F, 15, "bold")).pack(side="left", padx=10)
        tk.Label(top, text=sub, bg=color, fg=MUTED, font=(F, 10)).pack(side="left")
        inner = tk.Frame(z, bg=color)
        inner.pack(fill="x", padx=16, pady=(0, 16))
        return inner

    # ---- eventos ----------------------------------------------------------
    def _al_redimensionar(self, e):
        self.lienzo.itemconfigure(self.ventana, width=e.width)
        self.W = e.width
        if self.rr:
            self.after_cancel(self.rr)
        self.rr = self.after(150, self._rerender)

    def _rerender(self):
        self.rr = None
        if self.traza:
            self.render_narr()
            self.render_flujo()
            self.render_porque()

    def _ir_glosario(self):
        self.update_idletasks()
        y = self.glos.winfo_rooty() - self.cuerpo.winfo_rooty()
        total = max(1, self.cuerpo.winfo_height())
        self.lienzo.yview_moveto(max(0.0, min(1.0, (y - 80) / total)))

    def _cambiar_escenario(self, _e=None):
        clave = ESCENARIOS[self.cb.current()][0]
        self.detener()
        self.ex = 0
        self.tels = crear(clave)
        self.seleccionar(min(2, len(self.tels) - 1))

    def _editado(self):
        if self._relleno:
            return

        def num(v):
            try:
                return float(v.get())
            except ValueError:
                return 0.0

        t = self.tels[self.sel]
        t["cpu"], t["lat"] = num(self.v_cpu), num(self.v_lat)
        t["siem"], t["cuota"] = bool(self.v_siem.get()), bool(self.v_cuota.get())
        self.refresh()

    def _agregar(self):
        self.detener()
        self.ex = 0
        self.tels.append(dict(self.tels[-1]))
        self.seleccionar(len(self.tels) - 1)

    def _quitar(self):
        self.detener()
        self.ex = 0
        if len(self.tels) > 1:
            self.tels.pop()
            self.seleccionar(min(self.sel, len(self.tels) - 1))

    def elegir(self, i):
        self.detener()
        self.ex = 0
        self.seleccionar(i)

    # ---- narración / historia ----------------------------------------------
    def avanzar(self):
        if self.ex == 0:
            self.ex = 1
            self.refresh()
            return True
        if self.ex < 4:
            self.ex += 1
            self.refresh()
            return True
        self.mostrar_toast(f"Ciclo {self.sel + 1} completado: {resumen(self.traza[self.sel])}")
        if self.sel < len(self.tels) - 1:
            self.ex = 1
            self.seleccionar(self.sel + 1)
            return True
        self.ex = 0
        self.refresh()
        return False

    def paso_explicacion(self):
        self.detener()
        self.avanzar()

    def salir_explicacion(self):
        self.detener()
        self.ex = 0
        self.refresh()

    def alternar_historia(self):
        if self.timer:
            self.detener()
            return
        self.ex = 1
        self.seleccionar(0)
        self.b_hist.configure(text="Pausar historia")
        self.timer = self.after(1900, self._tick)

    def _tick(self):
        self.timer = None
        if self.avanzar():
            self.timer = self.after(1900, self._tick)
        else:
            self.b_hist.configure(text="Reproducir historia")

    def detener(self):
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        self.b_hist.configure(text="Reproducir historia")

    def mostrar_toast(self, msg):
        self.toast.configure(text=msg)
        self.toast.place(relx=0.5, rely=1.0, anchor="s", y=-18)
        self.toast.lift()
        if self.toast_id:
            self.after_cancel(self.toast_id)
        self.toast_id = self.after(2800, self.toast.place_forget)

    # ---- render ------------------------------------------------------------
    def seleccionar(self, i):
        self.sel = max(0, min(len(self.tels) - 1, i))
        self.rellenar_editor()
        self.refresh()

    def rellenar_editor(self):
        t = self.tels[self.sel]
        self._relleno = True
        self.l_ed.configure(text=f"ciclo {self.sel + 1} de {len(self.tels)}")
        self.v_cpu.set(f"{t['cpu']:g}")
        self.v_lat.set(f"{t['lat']:g}")
        self.v_siem.set(t["siem"])
        self.v_cuota.set(t["cuota"])
        self._relleno = False

    def refresh(self):
        self.traza = simular(self.tels)
        self.render_chips()
        self.render_tiles()
        self.render_graficos()
        self.render_narr()
        self.render_flujo()
        self.render_porque()

    def limpiar(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def render_chips(self):
        self.limpiar(self.fila_chips)
        for i, r in enumerate(self.traza):
            fuerte, pastel = TONOS[TONO[r["intencion"].nombre]]
            txt = f"Ciclo {i + 1}\n{CORTO[r['intencion'].nombre]}" + ("" if r["plan"] else "\n■ sin plan")
            sel = i == self.sel
            b = tk.Label(self.fila_chips, text=txt, bg=pastel, fg=fuerte, font=(F, 9, "bold"), padx=12, pady=5,
                         cursor="hand2", highlightthickness=3 if sel else 1,
                         highlightbackground=ACC if sel else LINE)
            b.bind("<Button-1>", lambda e, k=i: self.elegir(k))
            b.pack(side="left", padx=(0, 8))

    def render_tiles(self):
        self.limpiar(self.f_tiles)
        t = self.tels[self.sel]
        cpu = ("bad", "■", "CRÍTICO") if t["cpu"] > 80 else \
            ("warn", "▲", "ALTO") if t["cpu"] >= 60 else ("ok", "●", "OK")
        lat = ("bad", "■", "ALTA") if t["lat"] > 200 else \
            ("warn", "▲", "ATENCIÓN") if t["lat"] > 150 else ("ok", "●", "OK")
        siem = ("bad", "■", "ALERTA") if t["siem"] else ("ok", "●", "SIN ALERTA")
        cuota = ("bad", "■", "AGOTADA") if t["cuota"] else ("ok", "●", "DISPONIBLE")
        datos = [("CPU", f"{t['cpu']:g}%", cpu), ("LATENCIA", f"{t['lat']:g} ms", lat),
                 ("ALERTA SIEM", "Activa" if t["siem"] else "Apagada", siem),
                 ("CUOTA CLOUD", "Agotada" if t["cuota"] else "Con cuota", cuota)]
        for k, (cap, val, estado) in enumerate(datos):
            c = tarjeta(self.f_tiles)
            c.grid(row=0, column=k, sticky="nsew", padx=(0 if k == 0 else 5, 0 if k == 3 else 5))
            tk.Label(c, text=cap, bg=SURF, fg=MUTED, font=(F, 8, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
            tk.Label(c, text=val, bg=SURF, fg=FG, font=(F, 16, "bold")).pack(anchor="w", padx=10)
            semaforo(c, *estado).pack(anchor="w", padx=10, pady=(2, 10))

    def render_graficos(self):
        self.g_cpu.poner([float(t["cpu"]) for t in self.tels], self.sel, 100, 25, banda=True, alclic=self.elegir)
        self.g_lat.poner([float(t["lat"]) for t in self.tels], self.sel, 300, 100, umbral=200, unidad="ms", alclic=self.elegir)

    def render_narr(self):
        r = self.traza[self.sel]
        self.t_narr.configure(text=f"Ciclo {self.sel + 1} de {len(self.tels)}")
        extra = f"   Paso {self.ex} de 4 · {ETAPAS[self.ex - 1][3]}" if self.ex else ""
        self.s_narr.configure(text=narrar(r) + extra, wraplength=max(300, self.W - 160))
        self.limpiar(self.f_pasos)
        for i, e in enumerate(ETAPAS):
            act = self.ex == i + 1
            tk.Label(self.f_pasos, text=f" {i + 1}  {e[0]} ", bg=ACC_BG if act else SURF, fg=ACC if act else FG,
                     font=(F, 10, "bold"), padx=8, pady=4, highlightthickness=1,
                     highlightbackground=ACC if act else LINE).pack(side="left")
            if i < 3:
                tk.Label(self.f_pasos, text="  →  ", bg=Z2, fg=ACC, font=(F, 12, "bold")).pack(side="left")
        self.b_exp.configure(text="Siguiente paso ▶" if self.ex else "Modo explicación")
        if self.ex:
            self.b_salir.pack(side="left", padx=4, after=self.b_exp)
        else:
            self.b_salir.pack_forget()

    def _tarjeta_etapa(self, col, i):
        cur = self.ex == i + 1
        c = tk.Frame(self.f_flujo, bg=SURF, highlightthickness=3 if cur else 1,
                     highlightbackground=ACC if cur else LINE)
        c.grid(row=0, column=col, sticky="nsew")
        enc = tk.Frame(c, bg=SURF)
        enc.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(enc, text=str(i + 1), bg=ACC_BG, fg=ACC, font=(F, 11, "bold"), width=3).pack(side="left")
        tk.Label(enc, text=ETAPAS[i][0], bg=SURF, fg=FG, font=(F, 12, "bold")).pack(side="left", padx=8)
        self.t(c, ETAPAS[i][1], fg=MUTED).pack(anchor="w", padx=12, pady=(4, 0))
        tk.Label(c, text=ETAPAS[i][2], bg=ACC_BG, fg=ACC, font=(FM, 9), padx=6).pack(anchor="w", padx=12, pady=(4, 6))
        cuerpo = tk.Frame(c, bg=SURF)
        cuerpo.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return cuerpo

    def render_flujo(self):
        self.limpiar(self.f_flujo)
        r = self.traza[self.sel]
        for k in (0, 2, 4, 6):
            self.f_flujo.columnconfigure(k, weight=1, uniform="etapa")
        cuerpos = []
        for i in range(4):
            cuerpos.append(self._tarjeta_etapa(i * 2, i))
            if i < 3:
                tk.Label(self.f_flujo, text="→", bg=Z2, fg=ACC, font=(F, 18, "bold")).grid(row=0, column=i * 2 + 1, padx=6)
        builders = [self.cuerpo_percibir, self.cuerpo_deliberar, self.cuerpo_decidir, self.cuerpo_actuar]
        for i, (cuerpo, build) in enumerate(zip(cuerpos, builders)):
            if self.ex and self.ex < i + 1:
                self.t(cuerpo, "Pendiente…", fg=MUTED).pack(anchor="w")
            else:
                build(cuerpo, r)

    def cuerpo_percibir(self, f, r):
        c, a = r["despues"], r["antes"]
        if r["cumple"] and not a["anomalia_fuga_memoria"]:
            h = "Se dedujo una fuga de memoria."
        elif c["anomalia_fuga_memoria"]:
            h = "La fuga de memoria sigue inferida."
        elif not r["completa"]:
            h = "Reuniendo lecturas de CPU."
        else:
            h = "Sin fuga de memoria por ahora."
        self.t(f, h, font=(F, 10, "bold")).pack(anchor="w", pady=(0, 6))
        for clave in CREENCIAS:
            cambio = a[clave] != c[clave]
            bg = ACC_BG if cambio else SURF
            fila = tk.Frame(f, bg=bg, highlightthickness=1, highlightbackground=ACC if cambio else LINE)
            fila.pack(fill="x", pady=2)
            tk.Label(fila, text=clave, bg=bg, fg=MUTED, font=(FM, 8)).pack(anchor="w", padx=6, pady=(3, 0))
            v = c[clave]
            if clave == "plan_fallido":
                sm = semaforo(fila, "ok", "●", "None") if v is None else semaforo(fila, "bad", "■", v, wrap=self.wc - 30)
            elif not v:
                sm = semaforo(fila, "ok", "●", "False")
            else:
                sm = semaforo(fila, "bad" if clave == "bajo_ataque" else "warn",
                              "■" if clave == "bajo_ataque" else "▲", "True")
            sm.pack(anchor="w", padx=6, pady=(1, 4))
            if cambio:
                tk.Label(fila, text="CAMBIÓ EN ESTE CICLO", bg=bg, fg=ACC, font=(F, 7, "bold")).pack(anchor="w", padx=6, pady=(0, 3))
        self.enlace(f, "Ver la regla de fuga de memoria", "show_regla")
        if self.show_regla:
            hist = c["historial_cpu"]
            cv = tk.Canvas(f, width=min(220, self.wc), height=84, bg=SURF, highlightthickness=0)
            cv.pack(anchor="w", pady=(4, 0))
            ancho = min(220, self.wc) / 3
            for k in range(3):
                ix = len(hist) - 3 + k
                x0 = k * ancho + 8
                if ix >= 0:
                    alto = min(100, hist[ix]) * 0.5
                    cv.create_rectangle(x0, 62 - alto, x0 + ancho - 16, 62, fill="#C9C0F2", outline="")
                    cv.create_text(x0 + (ancho - 16) / 2, 70, text=f"{hist[ix]:g}%", fill=FG, font=(FM, 8))
                else:
                    cv.create_text(x0 + (ancho - 16) / 2, 70, text="—", fill=MUTED, font=(FM, 8))
                cv.create_text(x0 + (ancho - 16) / 2, 80, text=f"p{k + 1}", fill=MUTED, font=(FM, 8))
            if r["completa"]:
                h3 = hist
                ok1, ok2 = r["mono"], r["delta"] >= 15
                tk.Label(f, text=f"p1 < p2 < p3 → {h3[0]:g} < {h3[1]:g} < {h3[2]:g} {'✓' if ok1 else '✗'}",
                         bg=SURF, fg=OK[0] if ok1 else BAD[0], font=(FM, 9), anchor="w").pack(anchor="w")
                tk.Label(f, text=f"p3 − p1 ≥ 15 → {r['delta']:g} {'✓' if ok2 else '✗'}",
                         bg=SURF, fg=OK[0] if ok2 else BAD[0], font=(FM, 9), anchor="w").pack(anchor="w")

    def cuerpo_deliberar(self, f, r):
        c = r["despues"]
        n = len(r["deseos"])
        self.t(f, f"{n} de 4 deseos candidatos.", font=(F, 10, "bold")).pack(anchor="w", pady=(0, 6))
        for nombre, prio, tono, corto, clave in DESEOS:
            activo = nombre in r["deseos"]
            fuerte, pastel = TONOS[tono]
            bg = pastel if activo else SURF
            fila = tk.Frame(f, bg=bg, highlightthickness=1, highlightbackground=pastel if activo else LINE)
            fila.pack(fill="x", pady=2)
            self.t(fila, nombre, fg=fuerte if activo else MUTED, bg=bg, font=(FM, 8, "bold")).pack(anchor="w", padx=6, pady=(4, 0))
            self.t(fila, f"{'● Activo' if activo else '○ No generado'} · prioridad {prio}", fg=fuerte if activo else MUTED,
                   bg=bg, font=(F, 9, "bold")).pack(anchor="w", padx=6)
            cond = "Deseo de fondo, siempre presente." if clave is None else \
                f"Se genera si {clave} es True; ahora es {si_no(c[clave])}."
            self.t(fila, cond, fg=MUTED, bg=bg, font=(F, 9)).pack(anchor="w", padx=6, pady=(0, 4))

    def cuerpo_decidir(self, f, r):
        nombre = r["intencion"].nombre
        fuerte, pastel = TONOS[TONO[nombre]]
        caja = tk.Frame(f, bg=pastel, highlightthickness=2, highlightbackground=fuerte)
        caja.pack(fill="x")
        self.t(caja, nombre, fg=fuerte, bg=pastel, font=(FM, 9, "bold")).pack(anchor="w", padx=8, pady=(8, 0))
        self.t(caja, f"prioridad {r['intencion'].prioridad}", fg=MUTED, bg=pastel, font=(FM, 9)).pack(anchor="w", padx=8, pady=(0, 8))
        otros = [d for d in DESEOS if d[0] in r["deseos"] and d[0] != nombre]
        txt = "Gana por tener la mayor prioridad entre los activos." if otros else "Es el único deseo activo."
        if r["int_prev"] is not None and r["int_prev"].nombre != nombre:
            txt += f" El agente cambió de foco: antes perseguía {CORTO[r['int_prev'].nombre]}."
        elif r["int_prev"] is not None:
            txt += " Es la misma intención del ciclo anterior."
        self.t(f, txt).pack(anchor="w", pady=(8, 0))

    def cuerpo_actuar(self, f, r):
        c = r["despues"]
        if r["plan"] is not None:
            caja = tk.Frame(f, bg=OK[1], highlightthickness=2, highlightbackground=OK[0])
            caja.pack(fill="x")
            self.t(caja, r["plan"].nombre, fg=OK[0], bg=OK[1], font=(FM, 9, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
            if r["cand"] and r["cand"][0][1] != "ok":
                tk.Label(caja, text=" Plan alternativo (fallback) ", bg=SURF, fg=ACC, font=(F, 9, "bold")).pack(anchor="w", padx=8, pady=(0, 8))
            else:
                tk.Frame(caja, bg=OK[1], height=6).pack()
            for k, acc in enumerate(r["acciones"], 1):
                fila = tk.Frame(f, bg=SURF, highlightthickness=1, highlightbackground=LINE)
                fila.pack(fill="x", pady=(6, 0))
                tk.Label(fila, text=str(k), bg=ACC, fg=SURF, font=(F, 9, "bold"), width=2).pack(side="left", padx=6, pady=6)
                self.t(fila, acc, font=(FM, 8), w=self.wc - 50).pack(side="left")
        else:
            caja = tk.Frame(f, bg=BAD[1], highlightthickness=2, highlightbackground=BAD[0])
            caja.pack(fill="x")
            self.t(caja, "■ Ningún plan aplicable.", fg=BAD[0], bg=BAD[1], font=(F, 10, "bold")).pack(anchor="w", padx=8, pady=(8, 0))
            self.t(caja, "El agente queda a la espera. No hay acciones que ejecutar.", fg=BAD[0], bg=BAD[1]).pack(anchor="w", padx=8, pady=(0, 8))
        desc = [(p, v) for p, v in r["cand"] if v != "ok"]
        if desc:
            if r["plan"] is None:
                abierto = True
                self.t(f, f"Planes descartados ({len(desc)})", font=(F, 10, "bold")).pack(anchor="w", pady=(8, 0))
            else:
                self.enlace(f, f"Ver planes descartados ({len(desc)})", "show_desc")
                abierto = self.show_desc
            if abierto:
                for p, v in desc:
                    self.t(f, p.nombre, font=(FM, 8)).pack(anchor="w", pady=(4, 0))
                    if v == "failed":
                        msg = "Descartado: está marcado como fallido."
                    elif v == "skip":
                        msg = "No se evalúa: ya hay un plan elegido."
                    else:
                        k, w = PRECOND.get(p.nombre, ("su precondición", True))
                        ahora = si_no(c.get(k)) if k in c else "?"
                        msg = f"Descartado: exige {k} = {si_no(w)} y ahora es {ahora}."
                    self.t(f, msg, fg=MUTED if v == "skip" else BAD[0], font=(F, 9)).pack(anchor="w")

    def enlace(self, f, texto, attr):
        abierto = getattr(self, attr)
        lk = tk.Label(f, text=("▾ " if abierto else "▸ ") + texto, bg=SURF, fg=ACC, font=(F, 10, "bold"), cursor="hand2")
        lk.pack(anchor="w", pady=(8, 0))

        def alternar(_e, a=attr):
            setattr(self, a, not getattr(self, a))
            self.render_flujo()

        lk.bind("<Button-1>", alternar)

    def render_porque(self):
        self.limpiar(self.f_porque)
        for tag, txt, aviso in razonamiento(self.traza[self.sel]):
            fila = tk.Frame(self.f_porque, bg=SURF)
            fila.pack(fill="x", padx=14, pady=6)
            tk.Label(fila, text=f" {tag} ", bg=WARN[1] if aviso else ACC_BG, fg=WARN[0] if aviso else ACC,
                     font=(FM, 9, "bold"), width=12).pack(side="left", anchor="n")
            tk.Label(fila, text=txt, bg=SURF, fg=FG, font=(F, 10), justify="left", anchor="w",
                     wraplength=max(300, self.W - 260)).pack(side="left", padx=12)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()