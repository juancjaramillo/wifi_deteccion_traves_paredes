# WiFi-DensePose- Detección A través de paredes

Convierte estrictamente señales Wi‑Fi en una “cámara” capaz de reconstruir y visualizar la pose humana a través de paredes, usando la técnica DensePose de Detectron2. Basado en la investigación de la Universidad Carnegie Mellon.

---

## 📝 Descripción

Este proyecto ilustra el pipeline completo:

1. **Captura de CSI** (Channel State Information): amplitud y fase de subportadoras Wi‑Fi.
2. **Preprocesamiento**: filtra ruido y normaliza los datos.
3. **Reconstrucción 3D aproximada**: genera un volumen de movimiento simulado.
4. **DensePose**: mapea cada píxel de la superficie corporal usando un modelo Detectron2.
5. **Visualización**: muestra resultados y advierte sobre privacidad.

Aunque el ejemplo usa datos sintéticos, refleja cómo señales Wi‑Fi pueden reemplazar cámaras para monitoreo no invasivo.  
⚠️ **Importante**: las aplicaciones reales requieren hardware CSI y cumplimiento de regulaciones éticas.

---

## 📁 Estructura del proyecto

```bash
deteccion_wifi_densepose/
├── .gitignore
├── requirements.txt
├── main.py
└── venv/                      # entorno virtual (ignorarlo en Git)
```

---

## 📋 Prerrequisitos

- Python 3.8 o superior  
- Adaptador Wi‑Fi con capacidad CSI (Intel 5300, Nexmon, etc.) — sólo para versiones avanzadas  
- (Opcional) GPU con CUDA para acelarar Detectron2

---

## ⚙️ Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/juancjaramillo/deteccion_wifi_densepose.git
cd deteccion_wifi_densepose

# 2. Crea y activa entorno virtual
# Windows (PowerShell)
py -3 -m venv venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
# macOS/Linux
# python3 -m venv venv
# source venv/bin/activate

# 3. Instala dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota**: Detectron2 a veces requiere instalación específica según tu sistema. Consulta su [documentación oficial](https://github.com/facebookresearch/detectron2).

---

## ▶️ Uso

```bash
python main.py
```

Salida esperada:
```
=== WiFi-DensePose Pipeline ===
[1] Capturando CSI...
[2] Preprocesando señales...
[3] Reconstrucción 3D...
[4] Cargando DensePose...
[5] Mapear DensePose...
[6] Visualizar resultados...
[Visualización] XXX píxeles mapeados de superficie corporal

⚠️ Atención: tecnología no invasiva, atraviesa paredes...
```

— `main.py` usa datos sintéticos para demo. Para un entorno real, implementa `capture_csi()` con tu herramienta CSI.

---

## 📦 Dependencias en `requirements.txt`

```text
numpy
opencv-contrib-python
imutils
scipy
pillow
detectron2
matplotlib
```

---

## 🔒 Privacidad y Ética

- **Sin cámaras**: monitorización a través de paredes.  
- **Aplicaciones**: seguridad doméstica, cuidado de ancianos, seguimiento de movimiento.  
- **Riesgos**: vigilancia no autorizada, invasión de privacidad.  

Se recomienda el diseño de **políticas claras**, **transparencia** y **consentimiento** de las personas monitoreadas.

---

## 🤝 Contribuciones

¡Tu feedback es bienvenido! Puedes abrir _issues_ o _pull requests_ en GitHub.

---

## 📄 Licencia

MIT License. Consulta el archivo `LICENSE` para más detalles.

