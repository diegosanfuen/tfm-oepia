# OEPIA ![Logo](https://diegosanfuen.github.io/staticsTFM/logo/logo.png)
Proyecto fin de master OEPIA

![Arquitecrtura de OEPIA](https://diegosanfuen.github.io/staticsTFM/sources/Arquitectura2.png)

El proyecto consta de los siguientes módulos:

- **ObtencionDatos**:
  En este módulo se implementan las clases necesarias para realizar WebScrapping sobre las fuentes de empleo público (BOE, BOCyL, ...).

- **Sesiones**:
  Clase personalizada, que utiliza SQLlite para persistir la base de datos de sesiones, y que sirve como memoria para el LLM.

- **AgentePDF**:
  Clase para el manejo de agente LangChain, que se encarga de parsear un PDF concreto directamente de la fuente.

- **LLM-CHAT**:
  En este módulo se implementa el Chatbot con ayuda del LLM de Meta LLaMA 3 y la biblioteca Ollama de uso libre, también se instancia la base de datos de sesiones propias, con persistencia en SQLlite y la base de datos vectorial implementada con ayuda de FAISS, todo ello coordinado con ayuda de un agente de Inteligencia Artificial (LLA).
  
  Por otro lado, presenta un interfaz basado en Gradio y lo expone al usuario, para que pueda interactuar con el Chat de manera visual y sencilla.

- **Script: principal.py**:
  Este script implementa el código necesario para lanzar toda la lógica del proyecto.

- **Fichero requirements**:
  Donde se listan todas las librerías necesarias para la ejecución del proyecto.

- **Ruta configuraciones**:
  Fichero de configuración auto explicado con los parámetros necesarios para la ejecución del proyecto.

- **Readme.md**:
  Guía de ayuda para la ejecución del proyecto y ejemplos de cómo utilizarlo.

En el siguiente diagrama se muestra el despliegue de los distintos módulos que componen el proyecto:

![Esquema de despliegue](https://diegosanfuen.github.io/staticsTFM/sources/Despliegue%20Proyecto%20OEPIA.png)