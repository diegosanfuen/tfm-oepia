# OEPIA
# Script principal IA

# Importamos librerias
import sys
from pathlib import Path
import os, yaml
import datetime
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
import logging
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re
from dotenv import load_dotenv  # Esta librería nos permite cargar las variables de ambiente en memoria
from langchain.agents import Tool
from typing import Sequence, Any
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.tools.base import BaseTool
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import AgentExecutor

load_dotenv()  # Realizamos la carga de las variables de ambiente
# Introducir esta variable de entorno en el lanzador
os.environ['PROJECT_ROOT'] = r'/content/tfm-oepia'

sys.path.insert(0, os.environ['PROJECT_ROOT'])
from Sesiones.sesiones import ManejadorSesiones as ses
from FaissOPEIA import carga as fcg
from OEPIA.Utiles import Prompts as prompts
from OEPIA.Utiles import Utiles as utls

# Herramienta del Agente PDF
obtener_boe_texto = utls.obtener_boe_texto

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['llm_oepia']['ruta'])
date_today = datetime.datetime.today().strftime("%Y_%m_%d")

# Configuración básica del logger
log_level = None
match config['logs_config']['level']:
    case 'DEBUG':
        log_level = logging.DEBUG
    case 'WARN':
        log_level = logging.WARNING
    case 'WARNING':
        log_level = logging.WARNING
    case 'ERROR':
        log_level = logging.ERROR
    case _:
        log_level = logging.INFO

logging.basicConfig(filename=PATH_BASE / config['logs_config']['ruta_salida_logs'] / f'logs_{date_today}.log',
                    level=log_level,
                    format=config['logs_config']['format'])

# Creamos el logger
logger = logging.getLogger()

# CSS para Gradio
CSS = utls.obtenerCSSOEPIAInterfaz()
try:
    modelo = config['llm_oepia']['parameters_modelo']['llm_model']
    temperature = config['llm_oepia']['parameters_modelo']['temperature']
    assistant_name = config['llm_oepia']['parameters_modelo']['nombre_asistente']
    llm = Ollama(model=modelo,
                 temperature=temperature)
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar el modelo {modelo} : {e}')
    exit()
try:
    sesiones = ses()
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar la base de datos de sesiones: {e}')
    exit()

# Generamos el token de sesion
token = ses.generate_token()
prompt_template = ChatPromptTemplate.from_template(prompts.obtenerPROMPTTemplatePrincipalOEPIA())
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever_inst = fcg()
retriever_faiss = retriever_inst.inialize_retriever()
retrieval_chain = create_retrieval_chain(retriever_faiss, document_chain)

# Definimos las herramientas de langchain
HERRAMIENTAS = [
    Tool(
        name="ObtenerTextBOE",
        func=obtener_boe_texto,
        description="Trae el texto del BOE dado una URL al PDF que contiene la oferta de empleo",
    )
]

# El sufijo y los ejemplos para la activacion de las herramientas
AGENTE_FEW_SHOT_EJEMPLOS = [
    """
    Question: ¿Obtén el BOE del enlace que te paso?
    Thought: Necesito localizar la url proporcionada por el usuario, la localizamos y es 
    https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf
    Action: ObtenerTextBOE["https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf"]
    Observation: "texto del BOE": Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ....    
    Action: Finish["El BOE contiene: Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ...."]
    """
]

AGENTE_FEW_SHOT_EJEMPLOS.extend([
    """
    Question: Descarga el BOE del texto anterior
    Thought: Necesito localizar del contexto la url proporcionada por el usuario, la localizamos y es 
    https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf
    Action: ObtenerTextBOE["https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf"]
    Observation: "texto del BOE": Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ....    
    Action: Finish["El BOE contiene: Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ...."]
""",
    """
    Question: Descarga el enlace al documento proporcionado 
    Thought: Necesito localizar la url proporcionada por el usuario, la localizamos y es 
    https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf
    Action: ObtenerTextBOE["https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf"]
    Observation: "texto del BOE": Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ....    
    Action: Finish["El BOE contiene: Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ...."]
    """,
])

SUFIJO = """
    \nEres un sistema inteligente realizando una serie de pensamientos y ejecutando acciones para poder responder la pregunta del usuario.
    Pero es importante que sólo debes de usar este agente, cuando te solicitan o detectas que hay que descargar un BOE o un BOCYL.
    Cada acción es una llamada a una función: ObtenerTextBOE(url: str): str
    Por favor, entrega la respuesta sin usar caracteres que puedan causar problemas de parsing como comillas dobles o comillas simples o comas.
    Puedes usar la función cuando consideres necesario. Cada acción se realiza por separado. Contesta siempre en castellano. 
    Después sigue procesando la petición del usuario con las demás ordenes
    
    Vamos a empezar
    
    Question: {input}
    {agent_scratchpad}
"""

PROMPT_AGENTE = PromptTemplate.from_examples(
    examples=AGENTE_FEW_SHOT_EJEMPLOS,
    suffix=SUFIJO,
    input_variables=["input", "agent_scratchpad"],
)


class ReActAgent(Agent):
    """
    Clase que define un agente especializado para la implementación de la estrategia ReAct.
    Esta clase extiende de `Agent` y proporciona personalizaciones específicas necesarias
    para adaptar el agente a las necesidades del flujo de trabajo ReAct.

    Métodos:
        _get_default_output_parser: Retorna el parser de salida personalizado para este agente.
        create_prompt: Genera y retorna una plantilla de prompt específica para este agente.
        _validate_tools: Valida que el conjunto de herramientas utilizado cumpla con los requisitos del agente.
    """

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """
        Retorna una instancia de ReActOutputParser, que es un parser de salida personalizado
        para manejar y analizar las respuestas generadas por el LLM específico de ReAct.

        Args:
            kwargs: Argumentos adicionales para la configuración del parser.

        Returns:
            Una instancia de ReActOutputParser.
        """
        return ReActOutputParser()

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """
        Crea y retorna una plantilla de prompt personalizada que es usada por el agente ReAct.

        Args:
            tools: Una secuencia de herramientas disponibles para ser usadas en el prompt.

        Returns:
            Una instancia de BasePromptTemplate que contiene el prompt personalizado para el agente.
        """
        return PROMPT_AGENTE

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """
        Valida que el conjunto de herramientas proporcionado cumpla con los requisitos específicos del agente.
        Para este agente, solo debe haber una herramienta en el conjunto.

        Args:
            tools: Una secuencia de herramientas a validar.

        Raises:
            ValueError: Si el número de herramientas no es exactamente uno.
        """
        if len(tools) != 1:
            raise ValueError("The number of tools is invalid.")

    @property
    def _agent_type(self) -> str:
        """
        Define y retorna el tipo de agente, que en este caso es 'react'.

        Returns:
            El string 'react' que identifica este tipo de agente.
        """
        return "react"

    @property
    def finish_tool_name(self) -> str:
        """
        Proporciona el nombre de la herramienta que se usa para finalizar las sesiones con este agente.

        Returns:
            El nombre de la herramienta, que es 'Finish'.
        """
        return "Finish"

    @property
    def observation_prefix(self) -> str:
        """
        Retorna el prefijo utilizado para las observaciones generadas por este agente.

        Returns:
            Un string que representa el prefijo de observación.
        """
        return f"Observation: "

    @property
    def llm_prefix(self) -> str:
        """
        Retorna el prefijo utilizado para los pensamientos generados por el LLM cuando este agente está activo.

        Returns:
            Un string que representa el prefijo de pensamiento.
        """
        return f"Thought: "


# Creamos una instancia de nuestro agente
agent = ReActAgent.from_llm_and_tools(
    llm,
    HERRAMIENTAS,
)
# Definimos el agente ejecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=HERRAMIENTAS,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=3,
    return_messages=True,
)

# Creamos el chain final
llmApp = retrieval_chain | agent_executor
# llmApp = agent_executor | retrieval_chain


def chat(pregunta):
    """
    Procesa una pregunta de entrada y devuelve una respuesta basada en diferentes comandos y el contexto de la sesión.

    Esta función maneja diferentes tipos de comandos dentro de las preguntas, como resetear la sesión, ver el historial,
    o utilizar un agente específico para procesar la pregunta. Utiliza un token de sesión global para mantener el estado
    y el contexto del diálogo durante las interacciones con el usuario.

    Args:
        pregunta (str): La pregunta o comando ingresado por el usuario.

    Returns:
        str: La respuesta generada en base a la pregunta del usuario.

    Raises:
        Exception: Captura y registra excepciones específicas si la invocación del LLM o el procesamiento falla.
    """

    global token
    answer = "<h1>SE PRODUJO UN ERROR</h1>"
    if ("@resetear_sesion" in pregunta.lower()):
        token = ses.generate_token()
        answer = "Sesión reseteada"

    elif ("@ver_historial" in pregunta.lower()):
        answer = sesiones.obtener_mensajes_por_sesion(token)

    elif ("usa el agente para" in pregunta.lower()):
        try:
            response = agent_executor.run(pregunta + " " + str(sesiones.obtener_mensajes_por_sesion(token)))
            answer = str(response['answer'])
            sesiones.add_mensajes_por_sesion(token, str(pregunta))
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(str(str))
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar invocar el LLM: {e}')


    else:
        try:
            response = llmApp.invoke({"input": pregunta,
                                      "context": str(sesiones.obtener_mensajes_por_sesion(token))})
            answer = str(response['answer'])
            sesiones.add_mensajes_por_sesion(token, str(pregunta))
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(str(str))
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar invocar el LLM: {e}')


    return answer


history = ""


def format_links(text):
    """
    Convierte todas las URLs encontradas en el texto dado en etiquetas HTML <a> clicables.

    Esta función identifica URLs utilizando una expresión regular y las transforma en etiquetas <a>,
    que son enlaces clicables que se abren en nuevas pestañas del navegador. Esto mejora la interactividad
    y accesibilidad del texto en interfaces web.

    Args:
        text (str): El texto que puede contener URLs para formatear.

    Returns:
        str: El texto con URLs transformadas en etiquetas <a> HTML.

    Ejemplo:
        >>> example_text = "Visita https://www.example.com para más información."
        >>> format_links(example_text)
        'Visita <a href="https://www.example.com" target="_blank" style="color: blue;">https://www.example.com</a> para más información.'
    """

    # Esta función busca URLs en el texto y las reemplaza por etiquetas HTML <a>
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    formatted_text = re.sub(url_pattern, lambda
        url: f'<a href="{url.group(0)}" target="_blank" style="color: blue;">{url.group(0)}</a>', text)
    return formatted_text


# Ejecutamos gradio
with gr.Blocks() as iface:
    with gr.Row():
        texto_entrada = gr.Textbox(label="Ingresa tu mensaje", placeholder="Escribe aquí...", lines=10)
        historial_previo = gr.Textbox(label="Historial", value="",
                                      visible=False)  # Campo oculto para mantener el historial

    texto_entrada.change(fn=format_links, inputs=texto_entrada, outputs=historial_previo)


def interactuar_con_llm(texto, historial_previo):
    """
    Procesa una interacción con un modelo de lenguaje, gestionando y formateando el historial de conversación.

    Esta función toma un texto de entrada del usuario, lo limpia, y luego genera una respuesta utilizando
    una función chat simulada. Además, gestiona y actualiza un registro del historial de la conversación,
    mostrando este historial en formato HTML para una fácil visualización.

    Args:
        texto (str): El texto ingresado por el usuario para ser procesado por el modelo.
        historial_previo (str): El historial acumulado de interacciones anteriores.

    Returns:
        str: El historial actualizado de la conversación con la nueva entrada y respuesta incluida,
             formateado como un string HTML.

    Notas:
        - `history` es una variable global que mantiene el estado del historial de la conversación.
        - Esta función depende de la función `chat()` para simular respuestas del modelo LLM.
    """

    global history
    historial_previo = historial_previo + str(history)
    # Limpia el texto de entrada
    texto_limpio = texto.strip()

    # Simula la respuesta del modelo LLM
    respuesta = chat(texto_limpio)
    html_wrapper = f"""
    <div class="container">
        <details>
            <summary>Historial {datetime.datetime.today().strftime('%H:%M:%S')}</summary>
            <div class="content">
                <p>{history}</p>
            </div>
        </details>
    </div>
    """

    # Si es la primera interacción, no añade una línea en blanco al inicio
    if historial_previo:
        nuevo_historial = f"\n<h3><u>USUARIO:</h3></u><pre> {texto_limpio}</pre>\n\n<h3><u>OEPIA:</u></h3> <div><p>{respuesta}</p></div><br><br>{html_wrapper}\n\n"
    else:
        nuevo_historial = f"\n<h3><u>USUARIO:</u></h3><pre> {texto_limpio}</pre>\n\n<h3><u>OEPIA:</u></h3> <div><p>{respuesta}</p></div>\n\n"

    # Retorna el historial actualizado para mostrarlo en la salida
    history = nuevo_historial
    return nuevo_historial


# Esta función podría contener la lógica de postprocesamiento
def procesar_respuesta(respuesta):
    """
     Procesa la respuesta recibida, realizando ajustes o transformaciones antes de su uso posterior.

     Esta función está destinada a ser un lugar para realizar cualquier manipulación necesaria
     de la respuesta antes de mostrarla o procesarla más en el programa. Actualmente, la función
     limpia el valor de un campo de entrada después de recibir la respuesta, pero podría
     extenderse para incluir más lógica según sea necesario.

     Args:
         respuesta (str): La respuesta obtenida que necesita ser procesada.

     Returns:
         str: La respuesta procesada, que en este caso es la misma que la entrada.

     Ejemplo:
         >>> procesar_respuesta("Hola, mundo!")
         'Hola, mundo!'

     Nota:
         `texto_entrada.value = ""` parece indicar que hay un campo de entrada (posiblemente en una interfaz gráfica)
         que se limpia cada vez que se procesa una respuesta. Esto sugiere que la función está vinculada
         a la lógica de la interfaz de usuario donde `texto_entrada` es un widget o componente interactivo.
     """
    return respuesta


def procesar_flag(texto_entrada: str,
                  flag_option: str,
                  flag_index: int):
    """
    Procesa y registra la marcación de un dato específico con un flag correspondiente a una opción seleccionada.

    Esta función es útil en interfaces donde los usuarios interactúan con datos y tienen la capacidad de marcar
    o etiquetar estos datos para operaciones futuras como clasificación, revisión o cualquier otro procesamiento.
    La función imprime detalles sobre la entrada marcada, incluyendo su contenido, la opción de flag seleccionada
    y el índice de la entrada dentro de una colección o lista.

    Args:
        texto_entrada (objeto): Un objeto que contiene el texto de la entrada marcada. Se espera que tenga un atributo `value`.
        flag_option (str): La opción de flag seleccionada por el usuario para aplicar a la entrada de texto.
        flag_index (int): El índice numérico que identifica la posición de la entrada marcada dentro de una lista o colección.

    Ejemplo:
        >>> procesar_flag(mi_entrada, "Importante", 3)
        Dato marcado: Contenido de mi_entrada
        Opción seleccionada para marcar: Importante
        Índice del dato marcado: 3

    Notas:
        - Asume que `texto_entrada` es un objeto que tiene un atributo `value` que puede ser accedido directamente.
        - Los prints son para propósitos de depuración y visualización en desarrollo; considerar otro método de
          logging o manejo de eventos para entornos de producción.
    """
    print(f"Dato marcado: {texto_entrada.value}")
    print(f"Opción seleccionada para marcar: {flag_option}")
    print(f"Índice del dato marcado: {flag_index}")


# Crea la interfaz de Gradio
iface = gr.Interface(
    fn=interactuar_con_llm,
    inputs=[texto_entrada, historial_previo],
    outputs=gr.Markdown(label="Historial de la conversación"),
    title="<img src ='https://diegosanfuen.github.io/staticsTFM/logo/logo.png' />OEPIA: La IA especializada en ofertas de Empleo Público",
    description="Escribe un mensaje y presiona 'Submit' para interactuar con el modelo de lenguaje.",
    live=False,  # Desactiva la actualización en tiempo real
    css=CSS,
    article=utls.obtener_ayuda_oepia(),
    thumbnail=True,
    allow_flagging="manual",  # Permite marcar manualmente las entradas
    flagging_options=["Incorrecto", "Irrelevante", "Ofensivo"],  # Opciones para el usuario al marcar
    flagging_dir="flagged_data",  # Directorio donde se guardarán los datos marcados
)

# Lanza la interfaz
iface.launch(share=True)
