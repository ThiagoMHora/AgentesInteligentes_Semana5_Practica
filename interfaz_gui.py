import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
from typing import Tuple, Dict, Any, List
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from practica_semana04_rl import (
    EntornoGridworldEstocastico,
    entrenar_q_learning,
    evaluar_politica_greedy,
    formatear_tabla_bloques,
)


class GridworldVisualApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluación Q-Learning & Reward Shaping - Gridworld 4x4")
        self.root.geometry("1180x760")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#1e1e2e")

        # Nombres y símbolos
        self.nombres_acciones = ["Izquierda (←)", "Abajo (↓)", "Derecha (→)", "Arriba (↑)"]
        self.flechas_acciones = ["←", "↓", "→", "↑"]

        # Variables RL
        self.reward_shaping_var = tk.BooleanVar(value=True)
        self.alpha_var = tk.DoubleVar(value=0.8)
        self.gamma_var = tk.DoubleVar(value=0.95)
        self.epsilon_var = tk.DoubleVar(value=1.0)
        self.decay_var = tk.DoubleVar(value=0.995)
        self.min_epsilon_var = tk.DoubleVar(value=0.01)
        self.velocidad_var = tk.IntVar(value=220)

        self.env = EntornoGridworldEstocastico(reward_shaping=self.reward_shaping_var.get())
        self.q_table = np.zeros((self.env.n_estados, self.env.n_acciones))

        # Estadísticas de simulación
        self.episodios_totales = 0
        self.victorias = 0
        self.caidas = 0
        self.estado_actual = self.env.reset()
        self.historial_trayectoria = [self.estado_actual]
        self.bucles_en_episodio = 0
        self.esta_animando = False
        self.animacion_job = None

        self._configurar_estilos()
        self._crear_notebook_pestanas()

    def _configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4", font=("Segoe UI", 10, "bold"), padding=[12, 6])
        style.map("TNotebook.Tab", background=[("selected", "#89b4fa")], foreground=[("selected", "#11111b")])
        style.configure("TLabel", background="#181825", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TFrame", background="#1e1e2e")

    def _crear_notebook_pestanas(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña 1: Simulación y Tablero en Vivo
        self.pestana_sim = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(self.pestana_sim, text="  🎮 Simulación en Vivo y Bellman  ")
        self._construir_pestana_simulacion()

        # Pestaña 2: Demostración Experimental de las 2 Evaluaciones
        self.pestana_exp = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(self.pestana_exp, text="  🔬 Demostración Experimental (Evaluaciones 1 y 2)  ")
        self._construir_pestana_experimentos()

    # ==========================================================================
    # PESTAÑA 1: SIMULACIÓN INTERACTIVA
    # ==========================================================================
    def _construir_pestana_simulacion(self):
        main_frame = tk.Frame(self.pestana_sim, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Panel Izquierdo: Tablero
        panel_izq = tk.Frame(main_frame, bg="#181825", bd=2, relief=tk.GROOVE)
        panel_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            panel_izq,
            text="Tablero Gridworld 4x4 (Valores Q y Política Óptima)",
            bg="#181825",
            fg="#89b4fa",
            font=("Segoe UI", 11, "bold")
        ).pack(pady=6)

        self.canvas_size = 440
        self.canvas = tk.Canvas(
            panel_izq,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#11111b",
            highlightthickness=0
        )
        self.canvas.pack(pady=4)

        # Leyenda
        leyenda = tk.Frame(panel_izq, bg="#181825")
        leyenda.pack(fill=tk.X, padx=10, pady=4)
        items = [
            ("Inicio (0)", "#89b4fa"),
            ("Hielo (F)", "#a6adc8"),
            ("Hoyo (H)", "#f38ba8"),
            ("Meta (15)", "#a6e3a1"),
            ("Agente 🤖", "#f9e2af")
        ]
        for txt, color in items:
            tk.Label(leyenda, text=f"■ {txt}", bg="#181825", fg=color, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, expand=True)

        # Panel Derecho: Controles e Inspección Detrás de Escenas
        panel_der = tk.Frame(main_frame, bg="#181825", bd=2, relief=tk.GROOVE)
        panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controles
        frame_ctrl = tk.LabelFrame(panel_der, text=" ⚙ Controles y Parámetros ", bg="#181825", fg="#cba6f7", font=("Segoe UI", 10, "bold"), padx=8, pady=6)
        frame_ctrl.pack(fill=tk.X, padx=10, pady=5)

        chk_shaping = tk.Checkbutton(
            frame_ctrl,
            text="Reward Shaping Activado (-1.0 por hoyo, -0.01 por paso)",
            variable=self.reward_shaping_var,
            bg="#181825",
            fg="#a6e3a1",
            activebackground="#181825",
            activeforeground="#a6e3a1",
            selectcolor="#313244",
            font=("Segoe UI", 9, "bold"),
            command=self.cambiar_shaping
        )
        chk_shaping.pack(anchor="w", pady=2)

        fila_botones = tk.Frame(frame_ctrl, bg="#181825")
        fila_botones.pack(fill=tk.X, pady=4)

        btn_step = tk.Button(fila_botones, text="⏭ Paso a Paso", bg="#89b4fa", fg="#11111b", font=("Segoe UI", 9, "bold"), command=self.dar_un_paso)
        btn_step.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_auto = tk.Button(fila_botones, text="▶ Auto-Play", bg="#a6e3a1", fg="#11111b", font=("Segoe UI", 9, "bold"), command=self.toggle_auto_episodio)
        self.btn_auto.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_reset_pos = tk.Button(fila_botones, text="🔄 Reset Pos", bg="#f9e2af", fg="#11111b", font=("Segoe UI", 9, "bold"), command=self.reiniciar_posicion)
        btn_reset_pos.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        fila_train = tk.Frame(frame_ctrl, bg="#181825")
        fila_train.pack(fill=tk.X, pady=4)

        tk.Button(fila_train, text="⚡ Entrenar +100 Ep.", bg="#cba6f7", fg="#11111b", font=("Segoe UI", 9, "bold"), command=lambda: self.entrenar_en_bloque(100)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(fila_train, text="🚀 Entrenar +1000 Ep.", bg="#fab387", fg="#11111b", font=("Segoe UI", 9, "bold"), command=lambda: self.entrenar_en_bloque(1000)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(fila_train, text="🗑 Reiniciar Q", bg="#f38ba8", fg="#11111b", font=("Segoe UI", 9, "bold"), command=self.reiniciar_q_table).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Métricas
        self.lbl_stats = tk.Label(
            panel_der,
            text="Episodios: 0 | Éxitos: 0 (0.0%) | Caídas: 0 | ε: 1.000",
            bg="#313244",
            fg="#cdd6f4",
            font=("Segoe UI", 9, "bold"),
            pady=4
        )
        self.lbl_stats.pack(fill=tk.X, padx=10, pady=4)

        # Log Detrás de Escenas (Bellman & Bucles)
        frame_log = tk.LabelFrame(
            panel_der,
            text=" 🧠 Inspección Bellman Model-Free & Detector de Bucles ",
            bg="#181825",
            fg="#89b4fa",
            font=("Segoe UI", 10, "bold"),
            padx=8, pady=6
        )
        frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.txt_log = tk.Text(frame_log, bg="#11111b", fg="#cdd6f4", font=("Consolas", 9), height=12, relief=tk.FLAT)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.txt_log.insert(tk.END, "▶ Evaluación 1: Ecuación Bellman Model-Free activa.\n")
        self.txt_log.insert(tk.END, "▶ Evaluación 2: Detector de bucles y cálculo de penalización activo.\n")
        self.txt_log.insert(tk.END, "Haz clic en 'Paso a Paso' o 'Auto-Play' para inspeccionar.\n")

        self.dibujar_tablero()

    # ==========================================================================
    # PESTAÑA 2: DEMOSTRACIÓN EXPERIMENTAL FORMAL
    # ==========================================================================
    def _construir_pestana_experimentos(self):
        panel_exp = tk.Frame(self.pestana_exp, bg="#181825", padx=10, pady=10)
        panel_exp.pack(fill=tk.BOTH, expand=True)

        top_bar = tk.Frame(panel_exp, bg="#181825")
        top_bar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            top_bar,
            text="Demostración Experimental Comparativa (Evaluación 2: Convergencia y Bucles)",
            bg="#181825",
            fg="#89b4fa",
            font=("Segoe UI", 12, "bold")
        ).pack(side=tk.LEFT)

        self.btn_ejecutar_benchmark = tk.Button(
            top_bar,
            text="▶ Ejecutar Benchmark Experimental (1500 Episodios)",
            bg="#a6e3a1",
            fg="#11111b",
            font=("Segoe UI", 10, "bold"),
            command=self.ejecutar_benchmark_gui
        )
        self.btn_ejecutar_benchmark.pack(side=tk.RIGHT)

        # Panel de Métricas y Diagnóstico Teórico/Empírico.
        # Se empaqueta ANTES que las gráficas (side=BOTTOM) para que Tk le reserve su altura
        # fija primero; las gráficas ocupan el espacio restante sin deformarse. La barra de
        # desplazamiento permite leer el reporte completo (resumen + tabla por bloques)
        # incluso con la ventana en su tamaño mínimo (1050x700).
        frame_reporte = tk.Frame(panel_exp, bg="#11111b", bd=1, relief=tk.SOLID)
        frame_reporte.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        scroll_reporte = tk.Scrollbar(frame_reporte, orient=tk.VERTICAL)
        scroll_reporte.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_reporte = tk.Text(
            frame_reporte, bg="#11111b", fg="#cdd6f4", font=("Consolas", 9), height=12,
            relief=tk.FLAT, wrap=tk.NONE, yscrollcommand=scroll_reporte.set
        )
        self.txt_reporte.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        scroll_reporte.config(command=self.txt_reporte.yview)
        self.txt_reporte.insert(
            tk.END,
            "Presiona 'Ejecutar Benchmark Experimental' para entrenar ambos entornos\n"
            "(Estándar vs Reward Shaping) con 1500 episodios y comparar:\n"
            "  1. La tasa de éxito a lo largo del entrenamiento (efecto de la penalización por hoyo, -1.0).\n"
            "  2. La longitud de las trayectorias y los episodios truncados (efecto de la penalización por paso, -0.01).\n"
            "     Valor de deambular indefinidamente con shaping: Q = -0.01 / (1 - 0.95) = -0.200.\n"
            "  3. La tasa de éxito media por bloques de 250 episodios."
        )

        # Área de Gráficas de Matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor='#181825')
        self.fig.tight_layout(pad=3.0)
        self._inicializar_ejes_grafica()

        self.canvas_matplotlib = FigureCanvasTkAgg(self.fig, master=panel_exp)
        self.canvas_matplotlib.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

    def _inicializar_ejes_grafica(self):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('#11111b')
            ax.tick_params(colors='#cdd6f4', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#313244')

        self.ax1.set_title('Evaluación 2A: Tasa de Éxito vs Episodios\n(Aceleración de Convergencia)', color='#89b4fa', fontsize=10, fontweight='bold')
        self.ax1.set_xlabel('Episodios', color='#cdd6f4', fontsize=9)
        self.ax1.set_ylabel('% Éxitos (Media móvil 100)', color='#cdd6f4', fontsize=9)
        self.ax1.grid(True, linestyle='--', alpha=0.3, color='#45475a')

        self.ax2.set_title('Evaluación 2B: Pasos por Episodio\n(Longitud de Trayectorias)', color='#89b4fa', fontsize=10, fontweight='bold')
        self.ax2.set_xlabel('Episodios', color='#cdd6f4', fontsize=9)
        self.ax2.set_ylabel('Pasos Promedio', color='#cdd6f4', fontsize=9)
        self.ax2.grid(True, linestyle='--', alpha=0.3, color='#45475a')

    def ejecutar_benchmark_gui(self):
        self.btn_ejecutar_benchmark.config(state=tk.DISABLED, text="⏳ Entrenando...")
        self.root.update()

        env_std = EntornoGridworldEstocastico(reward_shaping=False)
        env_shp = EntornoGridworldEstocastico(reward_shaping=True)

        q_std, m_std = entrenar_q_learning(env_std, episodios=1500)
        q_shp, m_shp = entrenar_q_learning(env_shp, episodios=1500)

        eval_std = evaluar_politica_greedy(env_std, q_std, n_evals=150)
        eval_shp = evaluar_politica_greedy(env_shp, q_shp, n_evals=150)

        def media_movil(datos, ventana=80):
            return np.convolve(datos, np.ones(ventana)/ventana, mode='valid')

        self.ax1.cla()
        self.ax2.cla()
        self._inicializar_ejes_grafica()

        # Gráfica 1
        self.ax1.plot(media_movil(m_std["exitos_por_episodio"]) * 100, label='Estándar (Sparse)', color='#f38ba8', linewidth=2)
        self.ax1.plot(media_movil(m_shp["exitos_por_episodio"]) * 100, label='Reward Shaping', color='#a6e3a1', linewidth=2)
        self.ax1.legend(loc='lower right', facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=8)

        # Gráfica 2
        self.ax2.plot(media_movil(m_std["pasos_por_episodio"]), label='Estándar (Sin penalización paso)', color='#f38ba8', linewidth=2)
        self.ax2.plot(media_movil(m_shp["pasos_por_episodio"]), label='Reward Shaping (-0.01 paso)', color='#89b4fa', linewidth=2)
        self.ax2.legend(loc='upper right', facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=8)

        self.fig.tight_layout(pad=2.5)
        self.canvas_matplotlib.draw()

        tasa_std_500 = np.mean(m_std["exitos_por_episodio"][:500]) * 100
        tasa_shp_500 = np.mean(m_shp["exitos_por_episodio"][:500]) * 100
        tasa_std_fin = np.mean(m_std["exitos_por_episodio"][-300:]) * 100
        tasa_shp_fin = np.mean(m_shp["exitos_por_episodio"][-300:]) * 100
        pasos_std_fin = np.mean(m_std["pasos_por_episodio"][-300:])
        pasos_shp_fin = np.mean(m_shp["pasos_por_episodio"][-300:])
        trunc_std_fin = np.mean(m_std["truncados_por_episodio"][-300:]) * 100
        trunc_shp_fin = np.mean(m_shp["truncados_por_episodio"][-300:]) * 100
        bucles_std = np.mean(m_std["bucles_por_episodio"]) * 100
        bucles_shp = np.mean(m_shp["bucles_por_episodio"]) * 100

        reporte = (
            f"RESULTADOS EXPERIMENTALES (1500 Episodios) - comparación Estándar vs Shaping:\n"
            f"• 1er Éxito (episodio, base 0): Estándar = Ep {m_std['primer_exito']} | Shaping = Ep {m_shp['primer_exito']}\n"
            f"• Éxito temprano (Ep 1-500): Estándar = {tasa_std_500:.1f}% | Shaping = {tasa_shp_500:.1f}%\n"
            f"• Éxito final (últimos 300 ep): Estándar = {tasa_std_fin:.1f}% | Shaping = {tasa_shp_fin:.1f}%\n"
            f"• Pasos promedio / truncados (últimos 300 ep): Estándar = {pasos_std_fin:.1f} / {trunc_std_fin:.1f}% | "
            f"Shaping = {pasos_shp_fin:.1f} / {trunc_shp_fin:.1f}%\n"
            f"• Política greedy (éxito / truncados): Estándar = {eval_std['tasa_exito']*100:.1f}% / {eval_std['tasa_truncados']*100:.1f}% | "
            f"Shaping = {eval_shp['tasa_exito']*100:.1f}% / {eval_shp['tasa_truncados']*100:.1f}%\n"
            f"• Estado repetido (poco discriminante): Estándar = {bucles_std:.1f}% | Shaping = {bucles_shp:.1f}%\n"
            f"• Referencia: deambular indefinidamente con shaping vale Q = -0.01 / (1 - 0.95) = -0.200.\n\n"
            f"{formatear_tabla_bloques(m_std, m_shp, tamano_bloque=250)}"
        )
        self.txt_reporte.delete("1.0", tk.END)
        self.txt_reporte.insert(tk.END, reporte)
        self.btn_ejecutar_benchmark.config(state=tk.NORMAL, text="▶ Re-ejecutar Benchmark")

    # ==========================================================================
    # LÓGICA DE SIMULACIÓN Y DIBUJO
    # ==========================================================================
    def cambiar_shaping(self):
        self.env.reward_shaping = self.reward_shaping_var.get()
        modo = "ACTIVADO (-1.0 hoyo, -0.01 paso)" if self.env.reward_shaping else "DESACTIVADO (0.0 hoyo, 0.0 paso)"
        self.txt_log.insert(tk.END, f"\n[Config] Reward Shaping cambiado a: {modo}\n")
        # Los valores Q aprendidos con un esquema de recompensas no son válidos para el otro
        # (p. ej. con shaping son negativos en casi todo el tablero); mezclarlos falsearía
        # el aprendizaje, así que se reinicia Q, los contadores, epsilon y el tablero.
        self.reiniciar_q_table(
            mensaje="[Config] Tabla Q, contadores y ε reiniciados para no mezclar valores de esquemas distintos."
        )

    def reiniciar_posicion(self):
        self.estado_actual = self.env.reset()
        self.historial_trayectoria = [self.estado_actual]
        self.bucles_en_episodio = 0
        self.dibujar_tablero()

    def reiniciar_q_table(self, mensaje: str = "[Q-Learning] Tabla Q reiniciada a ceros."):
        if self.animacion_job:
            self.root.after_cancel(self.animacion_job)
            self.esta_animando = False
            self.btn_auto.config(text="▶ Auto-Play", bg="#a6e3a1")

        self.q_table = np.zeros((self.env.n_estados, self.env.n_acciones))
        self.episodios_totales = 0
        self.victorias = 0
        self.caidas = 0
        self.epsilon_var.set(1.0)
        self.reiniciar_posicion()
        self.actualizar_labels()
        self.txt_log.insert(tk.END, f"\n{mensaje}\n")
        self.txt_log.see(tk.END)

    def actualizar_labels(self):
        pct = (self.victorias / self.episodios_totales * 100) if self.episodios_totales > 0 else 0
        self.lbl_stats.config(
            text=f"Episodios: {self.episodios_totales} | Éxitos: {self.victorias} ({pct:.1f}%) | Caídas: {self.caidas} | ε: {self.epsilon_var.get():.3f}"
        )

    def dibujar_tablero(self):
        self.canvas.delete("all")
        cell_size = self.canvas_size / 4

        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                x0, y0 = c * cell_size, r * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                xc, yc = (x0 + x1) / 2, (y0 + y1) / 2

                if idx in self.env.agujeros:
                    bg, border = "#3d1421", "#f38ba8"
                elif idx == self.env.meta:
                    bg, border = "#193c2c", "#a6e3a1"
                elif idx == 0:
                    bg, border = "#1e283d", "#89b4fa"
                else:
                    bg, border = "#181825", "#313244"

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=bg, outline=border, width=2)

                lbl = "S (0)" if idx == 0 else f"H ({idx})" if idx in self.env.agujeros else f"G ({idx})" if idx == self.env.meta else f"{idx}"
                fg = "#89b4fa" if idx == 0 else "#f38ba8" if idx in self.env.agujeros else "#a6e3a1" if idx == self.env.meta else "#6c7086"
                self.canvas.create_text(x0 + 18, y0 + 14, text=lbl, fill=fg, font=("Segoe UI", 9, "bold"))

                if idx not in self.env.agujeros and idx != self.env.meta:
                    q_vals = self.q_table[idx]
                    best_a = int(np.argmax(q_vals))
                    if not np.all(q_vals == 0):
                        self.canvas.create_text(xc, yc, text=self.flechas_acciones[best_a], fill="#f9e2af", font=("Segoe UI", 20, "bold"))

                    self.canvas.create_text(x0 + 18, yc, text=f"{q_vals[0]:.2f}", fill="#9399b2", font=("Consolas", 7))
                    self.canvas.create_text(xc, y1 - 10, text=f"{q_vals[1]:.2f}", fill="#9399b2", font=("Consolas", 7))
                    self.canvas.create_text(x1 - 18, yc, text=f"{q_vals[2]:.2f}", fill="#9399b2", font=("Consolas", 7))
                    self.canvas.create_text(xc, y0 + 26, text=f"{q_vals[3]:.2f}", fill="#9399b2", font=("Consolas", 7))

                if idx == self.estado_actual:
                    rad = 18
                    self.canvas.create_oval(xc - rad, yc - rad, xc + rad, yc + rad, fill="#f9e2af", outline="#fab387", width=3)
                    self.canvas.create_text(xc, yc, text="🤖", font=("Segoe UI", 12))

    def dar_un_paso(self) -> bool:
        if self.estado_actual in self.env.agujeros or self.estado_actual == self.env.meta:
            self.reiniciar_posicion()
            return False

        s = self.estado_actual
        eps = self.epsilon_var.get()
        alpha = self.alpha_var.get()
        gamma = self.gamma_var.get()

        # EVALUACIÓN 1: Epsilon-Greedy con desempate aleatorio
        if random.uniform(0, 1) < eps:
            action = random.randint(0, self.env.n_acciones - 1)
            tipo_sel = "🎲 Exploración"
        else:
            max_q = np.max(self.q_table[s, :])
            candidatos = np.where(self.q_table[s, :] == max_q)[0]
            action = int(random.choice(candidatos))
            tipo_sel = "🎯 Explotación"

        # Simular paso y viento estocástico
        next_state, reward, done, resbalo, action_real = self._paso_con_info(action)

        # EVALUACIÓN 2: Detector de bucle
        hubo_bucle = self.historial_trayectoria.count(next_state) >= 1
        if hubo_bucle:
            self.bucles_en_episodio += 1
        self.historial_trayectoria.append(next_state)

        # EVALUACIÓN 1: Ecuación de Bellman Model-Free
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(self.q_table[next_state, :])

        # Q-Learning actualiza Q(s, a) con la acción ELEGIDA por el agente: el viento forma
        # parte de la dinámica del entorno, y su efecto ya está reflejado en (r, s').
        # Así coincide con entrenar_q_learning (entrenamiento por lotes).
        td_error = td_target - self.q_table[s, action]
        q_prev = self.q_table[s, action]
        self.q_table[s, action] += alpha * td_error
        q_nuevo = self.q_table[s, action]

        self.estado_actual = next_state

        # Registro en log
        viento_info = f"⚠️ RESBALÓ -> {self.nombres_acciones[action_real]}" if resbalo else "Directo"
        bucle_alerta = " 🔄 [BUCLE DETECTADO: Estado repetido]" if hubo_bucle else ""

        log_msg = (
            f"─ Paso en Estado {s} [{tipo_sel}]{bucle_alerta}\n"
            f"  • Acción elegida: {self.nombres_acciones[action]} | Viento: {viento_info}\n"
            f"  • Estado {s} ➔ {next_state} | Recompensa (R): {reward:+.2f}\n"
            f"  • TD-Target = {reward:+.2f} + {gamma}*max_a' Q({next_state}) = {td_target:.3f}\n"
            f"  • TD-Error  = {td_target:.3f} - {q_prev:.3f} = {td_error:+.3f}\n"
            f"  • Bellman: Q({s}, {self.flechas_acciones[action]}) pasó de {q_prev:.3f} a {q_nuevo:.3f}\n"
        )
        self.txt_log.insert(tk.END, log_msg)
        self.txt_log.see(tk.END)

        if done:
            self.episodios_totales += 1
            if next_state == self.env.meta:
                self.victorias += 1
                self.txt_log.insert(tk.END, "  🎉 ¡LLEGADA A LA META (+1.0)!\n\n")
            else:
                self.caidas += 1
                self.txt_log.insert(tk.END, f"  💀 CAYÓ EN HOYO ({reward:+.2f})\n\n")

            # Decaimiento epsilon
            self.epsilon_var.set(max(self.min_epsilon_var.get(), self.epsilon_var.get() * self.decay_var.get()))
            self.actualizar_labels()

        self.dibujar_tablero()
        return done

    def _paso_con_info(self, accion: int):
        s = self.estado_actual
        accion_deseada = accion
        resbalo = False
        if random.uniform(0, 1) < 0.66:
            accion = (accion + random.choice([-1, 1])) % 4
            resbalo = (accion != accion_deseada)

        r, c = divmod(s, 4)
        if accion == 0: c = max(0, c-1)
        elif accion == 1: r = min(3, r+1)
        elif accion == 2: c = min(3, c+1)
        elif accion == 3: r = max(0, r-1)
        ns = r * 4 + c
        self.env.estado_actual = ns

        terminado = False
        if ns in self.env.agujeros:
            terminado = True
            recompensa = -1.0 if self.env.reward_shaping else 0.0
        elif ns == self.env.meta:
            terminado = True
            recompensa = 1.0
        else:
            recompensa = -0.01 if self.env.reward_shaping else 0.0

        return ns, recompensa, terminado, resbalo, accion

    def toggle_auto_episodio(self):
        if self.esta_animando:
            self.esta_animando = False
            if self.animacion_job:
                self.root.after_cancel(self.animacion_job)
            self.btn_auto.config(text="▶ Auto-Play", bg="#a6e3a1")
        else:
            self.esta_animando = True
            self.btn_auto.config(text="⏸ Pausar", bg="#f38ba8")
            self._bucle_auto()

    def _bucle_auto(self):
        if not self.esta_animando:
            return
        done = self.dar_un_paso()
        espera = self.velocidad_var.get()
        if done:
            self.animacion_job = self.root.after(espera * 2, self._reiniciar_y_seguir)
        else:
            self.animacion_job = self.root.after(espera, self._bucle_auto)

    def _reiniciar_y_seguir(self):
        if not self.esta_animando:
            return
        self.reiniciar_posicion()
        self.animacion_job = self.root.after(self.velocidad_var.get(), self._bucle_auto)

    def entrenar_en_bloque(self, episodios: int):
        # Entrenamiento acumulativo: se continúa desde la Q y el ε actuales en lugar de
        # empezar de cero en cada lote.
        q, m = entrenar_q_learning(
            self.env,
            episodios=episodios,
            alpha=self.alpha_var.get(),
            gamma=self.gamma_var.get(),
            epsilon_inicial=self.epsilon_var.get(),
            epsilon_min=self.min_epsilon_var.get(),
            decay_rate=self.decay_var.get(),
            q_table_inicial=self.q_table
        )
        self.q_table = q
        exitos_lote = sum(m["exitos_por_episodio"])
        truncados_lote = sum(m["truncados_por_episodio"])
        self.episodios_totales += episodios
        self.victorias += exitos_lote
        # Los episodios truncados por límite de pasos no son caídas en un hoyo.
        self.caidas += (episodios - exitos_lote - truncados_lote)
        self.epsilon_var.set(m["epsilons"][-1])
        self.reiniciar_posicion()
        self.actualizar_labels()
        self.dibujar_tablero()
        self.txt_log.insert(
            tk.END,
            f"\n[Lote] +{episodios} episodios completados (truncados: {truncados_lote}). "
            f"ε = {self.epsilon_var.get():.3f} | Q(0) = {np.round(self.q_table[0, :], 3)}\n"
        )
        self.txt_log.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = GridworldVisualApp(root)
    root.mainloop()
