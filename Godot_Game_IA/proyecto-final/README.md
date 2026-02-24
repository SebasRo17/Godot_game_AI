# 🚀 AplicacionDeSpeedrunConAI

<img width="950" height="537" alt="Screenshot 2025-07-26 194201" src="https://github.com/user-attachments/assets/d5b357c9-9f42-4135-92ad-3ee9729bc896" />

[![Demo del proyecto](https://img.youtube.com/vi/BMqF3_rKCw8/0.jpg)](https://www.youtube.com/watch?v=BMqF3_rKCw8)


Proyecto de speedrun automatizado con IA integrada en Godot Engine usando aprendizaje por refuerzo (RL).

Aplicación de aprendizaje por refuerzo en un entorno de videojuego desarrollado para la simulación de estrategias de speedrun con un agente inteligente
Es un investigación que propone el desarrollo de un entorno de videojuego plataformero en Godot Engine 4.3, diseñado para entrenar un agente de inteligencia artificial (IA) mediante aprendizaje por refuerzo (RL) con el fin de optimizar estrategias de speedrun (completar el nivel en el menor tiempo posible sin recibir danio). El proyecto busca demostrar la eficacia de algoritmos como PPO (Proximal Policy Optimization) en la automatización de rutas eficientes, adaptación a obstáculos dinámicos, evitar danio y mejora iterativa del rendimiento. La necesidad de fomentar el aprendizaje práctico de la inteligencia artificial y el desarrollo de videojuegos como herramientas educativas. Al diseñar un entorno accesible donde una IA aprenda a completar un juego del genero plataformero, se busca demostrar el potencial de estas tecnologías para motivar la enseñanza de la programación. Desarrollar un entorno interactivo de videojuego plataformero en Godot Engine y entrenar un agente de inteligencia artificial mediante aprendizaje por refuerzo (RL) utilizando algoritmos como PPO para optimizar estrategias de speedrun, demostrando su capacidad para: Automatizar rutas eficientes en tiempo real. Adaptarse dinámicamente a obstáculos y mecánicas complejas (ej: saltos precisos, dashes). Mejorar iterativamente su rendimiento mediante recompensas basadas en tiempo y eficiencia de movimientos. Objetivos Específicos Diseñar y desarrollar un videojuego plataformero 2D en Godot Engine Implementar mecánicas clave para speedruns (saltos, dashes, plataformas móviles). Programar físicas personalizadas (gravedad, colisiones) y niveles con obstáculos dinámicos. Integrar un sistema de comunicación entre Godot y el agente de RL Exponer variables estructuradas (posición, velocidad, estado del personaje) mediante una API. Establecer un protocolo para enviar acciones del agente al juego (ej: movimientos, saltos). Implementar y entrenar un agente de RL, PPO Diseñar una función de recompensa que optimice el tiempo de completado (reward = - tiempo). Comparar el rendimiento del agente contra estrategias baselines (jugador humano, aleatorio). Evaluar la adaptabilidad del agente en escenarios no vistos Probar su desempeño en niveles con disposiciones de plataformas distintas a las de entrenamiento. Analizar su respuesta ante obstáculos dinámicos (ej: enemigos con patrones cambiantes). Cuantificar métricas de éxito (tiempo promedio, tasa de victoria, eficiencia de movimientos).

Objetivos Específicos
Diseñar y desarrollar un videojuego plataformero 2D en Godot Engine
Implementar mecánicas clave para speedruns (saltos, dashes, plataformas móviles).

Programar físicas personalizadas (gravedad, colisiones) y niveles con obstáculos dinámicos.
Integrar un sistema de comunicación entre Godot y el agente de RL
Exponer variables estructuradas (posición, velocidad, estado del personaje) mediante una
API.
Establecer un protocolo para enviar acciones del agente al juego (ej: movimientos, saltos).
Implementar y entrenar un agente de RL (PPO)
Diseñar una función de recompensa que optimice el tiempo de completado (reward = -
tiempo).
Comparar el rendimiento del agente contra estrategias baselines (jugador humano,
aleatorio).
Evaluar la adaptabilidad del agente en escenarios no vistos
Probar su desempeño en niveles con disposiciones de plataformas distintas a las de
entrenamiento.
Analizar su respuesta ante obstáculos dinámicos (ej: enemigos con patrones cambiantes).
Documentar el proceso y resultados
Cuantificar métricas de éxito (tiempo promedio, tasa de victoria, eficiencia de
movimientos).
Generar material educativo que relacione desarrollo de videojuegos con RL para fines
académicos.

OBJETIVOS
Objetivo General
Desarrollar un sistema autónomo que, mediante la integración de un entorno de videojuego en Godot Engine y un agente basado en aprendizaje por refuerzo (PPO), optimice estrategias de speedrun y demuestre capacidades de generalización ante entornos no vistos.
Objetivos Específicos
Diseñar un entorno plataformero 2D con mecánicas reproducibles para el entrenamiento del agente.
Implementar una comunicación bidireccional entre Godot y Python mediante sockets TCP/IP.
Entrenar un agente PPO con Stable Baselines3 para optimizar la velocidad de completado de los niveles.
Evaluar el rendimiento del agente frente a jugadores humanos y agentes aleatorios.
Documentar el proceso técnico y metodológico para su replicación en entornos educativos.

METODOLOGIA
La metodología combina investigación teórica y desarrollo técnico experimental, dividida en fases.
Fase teórica
Revisión bibliográfica de RL aplicado a videojuegos (DQN en Atari, PPO en Unity ML-Agents).
Estudio de técnicas de speedrun en juegos clásicos (Super Mario Bros, Celeste) para identificar mecánicas clave (saltos en pared, cadenas de dash).
Análisis de funciones de recompensa utilizadas en optimización temporal (reward shaping, discount factors).
Fase técnica
Selección de herramientas:
Motor de juego: Godot Engine 4.3 por su sistema modular, físicas personalizables y soporte para GDScript.
Framework de RL: Stable Baselines3 (PPO), equilibrando facilidad de integración y rendimiento.
Protocolo de comunicación: Socket TCP/IP, con intercambio de variables (posición, velocidad, estado del dash, datos de plataformas móviles).
Métricas de evaluación:
Tiempo promedio por nivel.
Eficiencia de movimientos (acciones redundantes penalizadas).
Tasa de generalización (% de éxito en niveles no vistos).

## 🔧 Instalación y Configuración

### Prerrequisitos
| Herramienta | Versión mínima | Enlace de descarga |
|-------------|----------------|-------------------|
| Godot Engine | 4.3 | [Descargar](https://godotengine.org/download) |
| Python | 3.10 | [Instalador](https://www.python.org/downloads/) |
| Git | 2.30+ | [Instalador](https://git-scm.com/downloads) |
| VSCode (Opcional) | 1.75+ | [Descargar](https://code.visualstudio.com/) |

### 🔄 Configuración del entorno virtual de Python
```bash
# Clonar repositorio
git clone https://github.com/MathSantill/AplicacionDeSpeedrunConAI.git
cd AplicacionDeSpeedrunConAI

# Crear y activar entorno virtual (Windows)
python -m venv .venv
.venv\Scripts\activate

# Crear y activar entorno virtual (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

Se recomienda estructurar los datos en formato JSON para facilidad de parsing y flexibilidad.

GitHub + GitHub Actions: para control de versiones, integración continua y automatización del despliegue.
```

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\rl\Scripts\Activate.ps1
```

```bash
py stable_baselines3_example.py --speedup=4 --timesteps=300000
```

### Arquitectura Implementada
Visión General
Estamos implementando una arquitectura cliente-servidor con separación estricta de responsabilidades, diseñada para:

Aislar el motor del juego (Godot) de la lógica de IA (Python)

Permitir entrenamiento offline de modelos RL

Facilitar la integración continua y despliegue

Mantener alta performance en tiempo real

Capas de la Arquitectura (en orden de implementación)
1. Capa de Presentación (Godot Engine)
Aspecto	Detalle
Responsabilidad	Renderizado gráfico, interfaz de usuario y física del juego
Tecnologías	Godot Engine 4.3, GDScript (81.3%), C# (9.9%)
Ubicación	Sprites/, Levels/, Scripts/Player/
Estado:	 Completado (100%)

2. Capa de Control de Juego
Aspecto	Detalle
Responsabilidad	Gestión de estados del juego, mecánicas y reglas
Tecnologías	GDScript, sistema de nodos de Godot
Ubicación	Scripts/Game/
Componentes clave	game_manager.gd, level_loader.gd
Estado: Completado (100%)

3. Capa de Comunicación
Aspecto	Detalle
Responsabilidad	Intercambio de datos entre juego y servidor de IA
Tecnologías	API REST (FastAPI), JSON over HTTP
Implementación	Godot: Scripts/AIController/agent.gd, Python: api.py
Protocolo	HTTP POST con estado del juego → Respuesta JSON con acción
Estado: En desarrollo (85%)

4. Capa de Lógica de IA
Aspecto	Detalle
Responsabilidad	Procesamiento de estados y generación de acciones óptimas
Tecnologías	Python 3.10+, Stable-Baselines3 (PPO/DQN), PyTorch
Ubicación	api.py, stable_baselines3_example.py
Estado: En desarrollo (70%)

5. Capa de Persistencia
Aspecto	Detalle
Responsabilidad	Almacenamiento de modelos entrenados y datos de sesiones
Tecnologías	Sistema de archivos local, formato .zip para modelos
Implementación	Directorio models/, training_logs/
Estado: Pendiente (0%)

6. Capa de Entrenamiento (Offline)
Aspecto	Detalle
Responsabilidad	Entrenamiento y optimización de modelos RL
Tecnologías	Python scripts, GitHub Actions (CI/CD)
Ubicación	.github/workflows/train.yml
Estado: Pendiente (30%)
Flujo Completo de Datos

Ciclo de Vida de una Acción
Captura: Godot recolecta estado del juego (60 FPS)
Preparación: Datos se estructuran en JSON
Transmisión: HTTP POST a localhost:5000/action

Procesamiento:
Servidor recibe estado
Modelo RL calcula mejor acción

Respuesta:
Acción serializada en JSON
Enviada de vuelta a Godot
Ejecución: Godot aplica acción en próximo frame
Retroalimentación: Resultado usado para próximo ciclo
Evolución de la Implementación
Fase Inicial (Completada)
Configuración de Godot Engine
Diseño básico de niveles
Movimiento básico del personaje
Sistema de colisiones
Fase Actual (Implementando)
Integración API REST
Comunicación Godot-Python
Modelo RL básico (PPO)
Sistema de acciones parametrizadas
Gestión de estados del juego
Próxima Fase
Entrenamiento avanzado con recompensas
Optimización de comunicación
Sistema de persistencia para modelos
Integración CI/CD con GitHub Actions
Sistema de logging y métricas
Desarrollo paralelo de componentes
Actualizaciones independientes
Escalabilidad para nuevos algoritmos RL
Portabilidad entre proyectos
Monitoreo granular del rendimiento
