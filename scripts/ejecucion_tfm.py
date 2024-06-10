# -*- coding: utf-8 -*-
"""
Script python a modo de guía de ejecucion del proyecto TFM
"""


!rm -Rf /content/tfm-oepia
!git clone https://github.com/diegosanfuen/tfm-oepia.git
!mkdir /content/tfm-oepia/Logs

!chmod 775 /content/tfm-oepia/scripts/*

!/content/tfm-oepia/scripts/instalar_requirements.sh

!/content/tfm-oepia/scripts/instalar_ollama.sh
!sleep 5
!/content/tfm-oepia/scripts/instalar_ollama.sh

# Descargamos del repo
!rm -Rf /content/tfm-oepia
!git clone https://github.com/diegosanfuen/tfm-oepia.git
!mkdir /content/tfm-oepia/Logs

!chmod 775 /content/tfm-oepia/scripts/*
!nohup ollama serve &

# Ejecutar la ingesta
!/content/tfm-oepia/scripts/ejecutar_ingesta.sh

# Descargamos del repo
!rm -Rf /content/tfm-oepia
!git clone https://github.com/diegosanfuen/tfm-oepia.git
!mkdir /content/tfm-oepia/Logs

!chmod 775 /content/tfm-oepia/scripts/*

# Ejecutar el chatbot
!nohup ollama serve &
!/content/tfm-oepia/scripts/ejecutar_chat_opeia.sh

