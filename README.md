# Proyecto Mateo - RL + Godot (Speedrun)

## Resumen
Este repositorio integra un videojuego 2D en Godot con un pipeline en Python para entrenamiento y evaluación de agentes de aprendizaje por refuerzo. El objetivo es entrenar un agente (PPO con LSTM) capaz de completar niveles de forma eficiente y permitir su uso en juego real mediante un modo “Fantasma” (IA vs Humano) con inferencia ONNX local (sin servidor Python).

## Objetivo del proyecto
- Entrenar un agente que aprenda a completar niveles de un juego 2D tipo speedrun.
- Integrar el entorno Godot con un agente en Python mediante comunicación TCP.
- Evaluar el desempeño con métricas reproducibles y exportables a CSV.
- Permitir jugar con un fantasma IA en paralelo al jugador humano.

## Alcance
- Entrenamiento con `RecurrentPPO` (PPO con memoria LSTM).
- Evaluación de episodios con métricas de retorno, pasos y distribución de acciones.
- Utilidades de inferencia y depuración del protocolo.
- Modo Fantasma con inferencia ONNX local y UI integrada.

## Tecnologías
- Godot Engine (juego 2D)
- Godot 4.6.1 .NET (para ONNX en C#)
- Python 3.x
- `stable-baselines3`
- `sb3-contrib` (RecurrentPPO)
- `gymnasium`
- `numpy`
- `onnx` (exportación del modelo)

## Arquitectura general
1. Godot ejecuta el juego y expone observaciones y recompensas.
2. Python actúa como servidor TCP, recibe observaciones y envía acciones.
3. El agente PPO aprende con episodios y actualiza la política.
4. Se guardan modelos y métricas para análisis posterior.
5. Para juego real, ONNX se ejecuta localmente (sin Python).

## Estructura del repositorio
- `train_ppo_mateo.py`: entrenamiento principal PPO (RecurrentPPO LSTM).
- `godot_gym_env.py`: entorno Gym que implementa el protocolo TCP con Godot.
- `export_onnx_speedrun.py`: exporta `ppo_speedrun_latest.zip` a ONNX (incluye normalización).
- `eval_metrics_godot.py`: evaluación de episodios y guardado de métricas al final.
- `eval_metrics_godot_livecsv.py`: evaluación con guardado por episodio.
- `eval_multiagent_livecsv.py`: evaluación multi-agente con CSV por episodio.
- `inference_server.py`: servidor de inferencia usando modelo JSON simple.
- `godot_server.py`: servidor de prueba con acciones aleatorias (debug).
- `debug_ports.py`: prueba de puertos múltiples para instancias de Godot.
- `speedrun_agent.py`: modelo simple (stdlib) entrenado con CSV o datos sintéticos.

Carpetas:
- `Godot_Game_IA\proyecto-final`: proyecto Godot (escena principal en `Levels/DemoLevel.tscn`).
- `checkpoints`: modelos PPO y estadísticas de normalización.
- `models`: modelos JSON para inferencia simple.
- `logs`: registros locales (si se usan).
- `venv`: entorno virtual local (no versionar).
- `AI-VideoGames`: copias de scripts (no requerido en el flujo principal).

## Instalación
1. Instalar Python 3.x.
2. Crear y activar un entorno virtual.
3. Instalar dependencias:

```bash
pip install stable-baselines3 sb3-contrib gymnasium numpy onnx
```

Nota: no hay `requirements.txt` en la raíz.

## Procedimiento de ejecución

### 1) Entrenamiento PPO (principal)
El entrenamiento usa 8 entornos en paralelo y espera 8 instancias de Godot en los puertos `11008..11015`.

```bash
python train_ppo_mateo.py
```

Arranque recomendado (para evitar espera):
1) Ejecutar Python
2) Luego abrir las instancias Godot con `--port`

Ejemplo (4 entornos):
```
speedRun.exe --port=11008
speedRun.exe --port=11009
speedRun.exe --port=11010
speedRun.exe --port=11011
```

### 2) Evaluación con métricas (single-agent)
```bash
python eval_metrics_godot.py
```

### 3) Evaluación con CSV en vivo
```bash
python eval_metrics_godot_livecsv.py
```

### 4) Evaluación multi-agente
```bash
python eval_multiagent_livecsv.py
```

### 5) Inferencia con modelo JSON simple
```bash
python inference_server.py --host 127.0.0.1 --port 11009 --model models/demo_speedrun_model.json
```

### 6) Modo Fantasma (IA local sin Python)
Requiere Godot .NET y el ONNX exportado en `models/ppo_speedrun_latest.onnx`.

1) Exportar ONNX:
```bash
python export_onnx_speedrun.py --model ppo_speedrun_latest --out /c/Users/roble/OneDrive/Documentos/proyecto-final/models/ppo_speedrun_latest.onnx
```

2) En Godot, presionar el botón “Jugar con IA”.

## Datos y salidas
- Modelos PPO: `checkpoints/ppo_speedrun_latest.zip`
- Normalización: `checkpoints/vecnormalize.pkl`
- Métricas CSV: `eval_metrics.csv`, `eval_multiagent.csv`
- Modelos simples: `models/*.json`
- ONNX: `models/ppo_speedrun_latest.onnx`

## Consideraciones para el informe
- Describir el entorno y el objetivo del agente.
- Justificar el uso de PPO + LSTM (observaciones parciales y estabilidad).
- Explicar la comunicación TCP y el framing (u32 + JSON).
- Reportar métricas: retorno promedio, pasos, distribución de acciones.
- Incluir configuración de entrenamiento (n_steps, batch_size, gamma, clip_range).

## Troubleshooting
- Si el entrenamiento se queda esperando: iniciar primero las instancias de Godot.
- Si hay error de puerto: ajustar rango o cerrar procesos.
- Si el desempeño no mejora: revisar el diseño de recompensas en Godot.
- Si el EXE no encuentra ONNX: en Export Preset, agregar `models/*.onnx` en non-resource files.

## Licencia
No especificada.

## Configuración Godot .NET (ONNX)
1. Instalar Godot 4.6.1 .NET.
2. Abrir el proyecto en Godot .NET.
3. Crear solución C# (si no existe): `Project → Tools → C# → Create C# solution`.
4. Compilar: `Build → Build Solution`.

### Dependencia ONNX Runtime
En el `.csproj` se requiere:
```xml
<ItemGroup>
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.18.0" />
</ItemGroup>
```

## Configuración de escena para el modo Fantasma
En `Levels/DemoLevel.tscn`:
1. Agregar un nodo `OnnxService` (Node) con script:
   `res://Scripts/OnnxService.cs`
2. Agregar un nodo `CSharpBootstrapNode` (Node) con script:
   `res://Scripts/CSharpBootstrap.cs`
3. En el nodo `Sync`, asignar:
   - `onnx_service_path = ../OnnxService`

## Exportación a .exe (con ONNX)
1. En el preset de exportación, **incluir non-resource files**:
   - `models/*.onnx`
2. Exportar desde Godot.
3. Verificar que en consola del .exe aparezca:
   - `[OnnxService] Init: res://models/ppo_speedrun_latest.onnx`

Si el .exe no encuentra el ONNX, revisar el filtro de exportación.
