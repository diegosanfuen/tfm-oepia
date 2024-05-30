#!/usr/bin/env sh
# SCRIPT PARA INSTALAR LOS REQUERIMENTS DEL PROYECTO

# Cargamos variables de entorno
export PROJECT_ROOT=/content/tfm-oepia
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

pip install -r $PROJECT_ROOT/scripts/requirements.txt



