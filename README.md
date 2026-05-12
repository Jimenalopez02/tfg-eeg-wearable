# Sistema wearable EEG para monitorización cognitiva

Este repositorio contiene el código desarrollado para un Trabajo Fin de Grado sobre monitorización de estados cognitivos mediante EEG frontal wearable.

## Módulos principales

- `SEED_VIG`: módulo de somnolencia entrenado con un modelo MLP.
- `EEGMAT`: módulo de concentración entrenado con Random Forest.
- `Puente_HW_MLP`: interfaz global y puente hardware-software para replay e inferencia pseudo-online.

## Datos

Los datasets originales no se incluyen en el repositorio por tamaño y condiciones de uso. Para reproducir los experimentos, deben colocarse manualmente en las carpetas correspondientes.

## Instalación

```bash
pip install -r requirements.txt

## Autora

Jimena López Maldonado