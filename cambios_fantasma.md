# Cambios funcionales para implementar el “Fantasma” (IA vs Humano)

Este documento registra **solo los cambios funcionales** realizados para habilitar el modo “Fantasma” (IA controlando un jugador paralelo) en el proyecto. No incluye errores ni intentos fallidos.

## Objetivo funcional
- El juego **inicia siempre en modo humano**.
- Al presionar el botón de IA, **no se toma el control del jugador humano**.
- Se **crea un fantasma** que juega en paralelo como desafío.
- El fantasma **no interfiere** con el jugador humano (no colisiones directas), pero **sí colisiona con el mundo**.
- La IA del fantasma funciona **sin servidor Python**, usando **ONNX local**.
- Al activar/desactivar IA se **resetea el contador**.

---

## Herramientas y requisitos
- **Godot 4.6.1 .NET** (necesario para ONNX con C#).
- **Microsoft.ML.OnnxRuntime** como dependencia NuGet.
- **Python 3.11 + PyTorch + sb3_contrib** para exportar el modelo a ONNX.
- Modelo entrenado `ppo_speedrun_latest.zip` y normalización `vecnormalize.pkl`.

---

## Cambios en Godot (GDScript)

### 1) `Scripts/Player.gd`
**Propósito:** permitir que un mismo Player pueda funcionar como humano o fantasma.

**Cambios clave:**
- Se agregaron flags de control:
  - `allow_human_input` (bool)
  - `allow_ai_input` (bool)
  - `is_ghost` (bool)
  - `ghost_tint` (Color)
- Se guardó `base_modulate` para mantener el tinte del fantasma después del dash.
- `apply_ai_action()` ahora **ignora** IA si `allow_ai_input` es falso.
- `get_input()` ahora **ignora** teclado si `allow_human_input` es falso.
- El color del sprite vuelve a `base_modulate` al salir del dash.

**Efecto:** el fantasma se mueve solo con IA y tiene visual distinto.

---

### 2) `Levels/demo_level.gd`
**Propósito:** crear, activar y sincronizar el fantasma dentro del nivel.

**Cambios clave:**
- Se agregó soporte para:
  - `_ghost`, `_player`, `_ai_controller`
- Función `set_ghost_enabled(enable)`:
  - Al activar: resetea jugador y fantasma al spawn, activa IA.
  - Al desactivar: destruye fantasma.
- Se reinicia `run_time` **al activar** y **al desactivar** IA.
- Se añadió `_create_ghost()`:
  - Instancia `Objects/Player.tscn`.
  - Desactiva input humano.
  - Marca `is_ghost`.
  - Colisiones: `collision_layer=2`, `collision_mask=1`.
  - Excepción de colisión con jugador humano.
- Se añadió `_reset_ghost_to_player()` para resetear fantasma cuando el humano muere.
- Se añadió `_reset_player_to_spawn()` para reiniciar el humano al activar IA.

**Efecto:** el fantasma aparece, respawnea y se mantiene independiente del humano.

---

### 3) `Levels/ai_controller_2d.gd`
**Propósito:** permitir que el controlador IA apunte al fantasma y no al humano.

**Cambios clave:**
- `set_player_target(target)` permite cambiar el jugador controlado por la IA.
- Protección en `_do_reset()` para no acceder a `_player` si es null.

**Efecto:** la IA puede controlar únicamente al fantasma.

---

### 4) `script_templates/AIController/ui_controller.gd`
**Propósito:** botón de UI activa/desactiva fantasma y usa ONNX local.

**Cambios clave:**
- Texto del botón:
  - “Jugar con IA” / “Jugar sin IA”.
- Botón sin focus para evitar que Space active el toggle.
- Al activarse:
  - llama `set_ghost_enabled(true)`.
  - llama `sync_node.start_onnx_mode()` con ruta ONNX.
- Al desactivarse:
  - llama `sync_node.stop_onnx_mode()`.

---

### 5) `addons/godot_rl_agents/sync.gd`
**Propósito:** ejecutar inferencia ONNX local sin servidor Python.

**Cambios clave:**
- Nuevo export `onnx_service_path` para usar inferencia vía C#.
- Implementación de `start_onnx_mode()` y `stop_onnx_mode()`.
- Soporte para inferencia usando `OnnxService` (C#) **antes** de `ONNXModel`.
- Decodificación correcta de acciones discretas desde logits:
  - Se usa `argmax` por cada rama (`move`, `jump`, `dash`).
- Validaciones robustas de tamaño y outputs.

**Efecto:** IA funciona localmente sin conexión a Python.

---

## Cambios en Godot (C#)

### 6) `addons/godot_rl_agents/onnx/csharp/ONNXInference.cs`
**Propósito:** inferencia ONNX directa en C#.

**Cambios clave:**
- Adapta inputs según el modelo (no requiere `state_ins` si no existe).
- Selecciona outputs disponibles (`output`, `state_outs` o primero disponible).
- Logs de inputs/outputs para depuración.

---

### 7) `Scripts/OnnxService.cs`
**Propósito:** puente C# para usar `ONNXInference` desde GDScript sin depender de ClassDB.

**Funciones:**
- `Init(modelPath, batchSize)`
- `RunInference(obs, stateIns)`
- `Stop()`

---

### 8) `Scripts/CSharpBootstrap.cs`
**Propósito:** asegurar carga del assembly C# al ejecutar la escena.

Se coloca como script en un **nodo hijo** del nivel para no romper el nodo raíz.

---

## Dependencias .NET

### 9) `PlatformerTemplate(v2).csproj`
**Propósito:** agregar ONNX Runtime.

Se añadió:
```xml
<ItemGroup>
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.18.0" />
</ItemGroup>
```

---

## Exportación del modelo a ONNX

### 10) Script `export_onnx_speedrun.py`
Ruta: `C:\Users\roble\Downloads\Mateo\Mateo\export_onnx_speedrun.py`

**Función:** exporta `ppo_speedrun_latest.zip` a ONNX y aplica **VecNormalize**.

Uso:
```bash
python ../export_onnx_speedrun.py --model ppo_speedrun_latest --out /c/Users/roble/OneDrive/Documentos/proyecto-final/models/ppo_speedrun_latest.onnx
```

Requiere:
- `vecnormalize.pkl` junto al `.zip`.
- `onnx` instalado:
```bash
python -m pip install onnx
```

---

## Configuración de escena (pasos en editor)

1. **Agregar nodo OnnxService** al nivel:
   - Crear Node hijo llamado `OnnxService`.
   - Asignar script: `res://Scripts/OnnxService.cs`.

2. **Asignar ruta en Sync**:
   - En nodo `Sync`, setear `onnx_service_path` a `../OnnxService`.

3. **Agregar CSharpBootstrap**:
   - Crear Node hijo llamado `CSharpBootstrapNode`.
   - Asignar script: `res://Scripts/CSharpBootstrap.cs`.

4. **Build**:
   - `Build → Build Solution` en Godot .NET.

---

## Resultado final

- El fantasma se activa/desactiva con botón.
- IA corre con ONNX local (sin Python).
- Contador se reinicia al activar y desactivar IA.
- Fantasma no colisiona con el jugador humano.
- Fantasma sí colisiona con el mapa.
- Juego inicia siempre en modo humano.

---

## Archivos tocados (resumen)

**GDScript**
- `Scripts/Player.gd`
- `Levels/demo_level.gd`
- `Levels/ai_controller_2d.gd`
- `script_templates/AIController/ui_controller.gd`
- `addons/godot_rl_agents/sync.gd`
- `addons/godot_rl_agents/onnx/wrapper/ONNX_wrapper.gd`

**C#**
- `addons/godot_rl_agents/onnx/csharp/ONNXInference.cs`
- `Scripts/OnnxService.cs`
- `Scripts/CSharpBootstrap.cs`

**.NET Project**
- `PlatformerTemplate(v2).csproj`

**Python**
- `C:\Users\roble\Downloads\Mateo\Mateo\export_onnx_speedrun.py`

---

## Notas técnicas
- ONNX usa inputs/outputs detectados dinámicamente.
- Decodificación de acciones discretas se hace con `argmax`.
- La normalización de observaciones se incorpora al modelo ONNX.
- La IA no requiere servidor Python en modo fantasma.

---
