#!/usr/bin/env sh
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve &
seleep 15
ollama pull llama3
