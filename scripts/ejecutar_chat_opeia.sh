#!/usr/bin/env sh
# SCRIPT PARA LA EJECUCION DEL INTERFACE DE OPEIA
# Cargamos variables de entorno
export PROJECT_ROOT=/content/tfm-oepia
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

# Ejecutamos la secuencia principal del chatoboot
python -m $PROJECT_ROOT/OEPIA/llm_chat.py

