# Proyecto de Visión por Ordenador: Calculadora

Este proyecto implementa una calculadora utilizando técnicas de visión por ordenador. Está diseñado para ejecutarse en una Raspberry Pi y emplea diversas herramientas para capturar, procesar y analizar imágenes.

## Estructura del Proyecto
El proyecto está organizado en varios archivos y carpetas para una mejor gestión y modularidad:

### Archivos principales:
- **`sistemaseguridad.py`**: Contiene las funciones relacionadas con la seguridad del sistema y el manejo de accesos.
- **`proyectofinal.py`**: Implementa la lógica principal de la calculadora y las operaciones de procesamiento de imagen.
- **`testcompleto.py`**: Archivo principal que ejecuta todo el proyecto. Incluye llamadas a los demás módulos y coordina la funcionalidad completa.

### Carpetas auxiliares:
- **`utils/`**: Contiene funciones auxiliares y herramientas necesarias para diversas operaciones del proyecto.
- **`output/`**: Carpeta donde se almacenan las imágenes generadas y procesadas por el proyecto.
- **`calibracion/`**: Archivo dedicado a la calibración de la cámara de la Raspberry Pi para garantizar la precisión en la captura de imágenes.

## Requisitos del Sistema
- Raspberry Pi con cámara compatible
- Python 3.x
- Bibliotecas necesarias: OpenCV, NumPy, pytesseract.


## Ejecución del Proyecto
Para ejecutar el proyecto completo, utiliza el archivo `testcompleto.py`:
```bash
python testcompleto.py
```

Este archivo coordina la funcionalidad de todos los módulos y asegura la correcta interacción entre ellos.

## Calibración
El archivo para calibrar la cámara de la raspberry. 
```bash
python calibracion.py
```

## Salida del Proyecto
Las imágenes procesadas y los resultados generados se almacenarán automáticamente en la carpeta `output/`.
