#!/usr/bin/env sh
# SCRIPT PARA LA EJECUCION DE LA INGESTA DE DATOS DEL BOE Y BOCYL A TRABVES DE FAISS
# Cargamos variables de entorno
export PROJECT_ROOT=/content/tfm-oepia
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

# Ejecutamos la secuencia principal del chatoboot
python -m $PROJECT_ROOT/ObtencionDatos/main.py

