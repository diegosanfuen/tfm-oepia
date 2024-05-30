#!/usr/bin/env sh
# Script para la descarga del modelo de OLLAMA e instalacion de la libreria.

# Cargamos variables de entorno
export PROJECT_ROOT=/content/tfm-oepia
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

# Descargamos el software de ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve &
seleep 15
ollama pull llama3

# Descargamos el modelo NLTK de resuemn de noticias
python -m $PROJECT_ROOT/scripts/descarga_resumidor_nltk.py
