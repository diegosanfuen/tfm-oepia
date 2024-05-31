# Importamos las librerias
import sqlite3
import string, secrets
import logging
import os, sys, yaml
from pathlib import Path
import datetime

# Introducir esta variable de entorno en el lanzador
# os.environ['PROJECT_ROOT'] = r'/content/tfm-oepia'

sys.path.insert(0, os.environ['PROJECT_ROOT'])

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['sesiones']['ruta'])
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


class ManejadorSesiones():
    """
    Clase para gestionar las sesiones de un modelo de aprendizaje automático (LLM).

    Esta clase proporciona métodos para inicializar, guardar y cargar sesiones
    de un modelo de aprendizaje automático, facilitando el manejo de estados
    entre sesiones de entrenamiento o inferencia. Utiliza internamente mecanismos
    de serialización para mantener la consistencia de los estados del modelo.

    Atributos:


    Métodos:
        generate_token (tamaño): Genera un token de sesion.
        obtener_db_conexion (): Obtirene la conexion con la BBDD de sesiones SQLite.
        probar_connection (). prueba la conexion con la BBDD sesiones
        obtener_mensajes_por_sesion (id_session): Obtiene los mensajes de sesion por id_sesion
        add_mensajes_por_sesion (id_session, prompt): Añade un mensaje a la sesion

    Ejemplo:
        >>> ManejadorSesiones.generate_token()
        >>> manager = ManejadorSesiones()
        >>> manager.obtener_db_conexion()
        >>> # realizar operaciones del modelo
        >>> manager.add_mensajes_por_sesion(id_sesion, prompt)
        >>> manager.obtener_mensajes_por_sesion(id_session)
    """

    def __init__(self):
        self.id_session = 0
        self.directorio_proyecto = Path(PATH_BASE) / config['sesiones']['ruta']
        self.path_db = self.directorio_proyecto / config['sesiones']['ruta_db']
        self.tabla_sesiones = config['sesiones']['tabla']
        self.conexion = self.obtener_db_conexion()
        self.probar_connection()
        self.conexion.close()

    @staticmethod
    def generate_token(length=32):
        """
        Genera un token aleatorio de sesión por defecto 32 caracteres alfanumericos
        :param length: Tamaño del token de sesion
        :return: El token de sesion
        """

        characters = string.ascii_letters + string.digits

        token = ''.join(secrets.choice(characters) for _ in range(length))
        return token


    def obtener_db_conexion(self):
        """
        Obtiene la conexion con la bbdd de sqlite de sesiones
        :param self:
        :return: la conexion con la bbdd de sesiones
        """
        try:
            conn = sqlite3.connect(self.path_db / config['sesiones']['nombre_db'])
            logging.info("Conexion establecida")
        except Exception as e:
            logging.exception("Ocurrió un error al intentar conectar con las base de datos")

        conn.row_factory = sqlite3.Row
        return conn


    def probar_connection(self):
        """
        Prueba la conexión con la bbdd sqlite de sesiones
        :param self:
        :return: ErrorCode (0: OK, 1: NOOK)
        """
        try:
            conn = self.conexion
            cursor = conn.execute(f'SELECT * FROM {self.tabla_sesiones} limit 1').fetchall()
            logging.info(f"Consulta probada OK en {cursor}")
            return 1
        except Exception as e:
            logging.exception("Ocurrió un error al intentar probar la consulta")
            return 0


    def obtener_mensajes_por_sesion(self, id_session) -> list:
        """
        Obitene los mensajes por sesion dado el id_sesion
        :param self:
        :param id_session: Identificador alfanumerico de la sesion
        :return: Los mensajes en formato lista de la sesion
        """
        prompts = []
        try:
            conn = self.obtener_db_conexion()
            res = conn.execute(f"SELECT * FROM {self.tabla_sesiones} where id_session='{id_session}';").fetchall()
            logging.info("Consulta ejecutada OK")

            for item in res:
                prompts.append(item[2])

            conn.close()
        except Exception as e:
            logging.exception(f"Ocurrió un error al obtener datos para la sesion {id_session}")

        return prompts


    def add_mensajes_por_sesion(self, id_session, prompt: str) -> list:
        """
        Añade mensajes para la sesion dado el id_sesion
        :param self:
        :param id_session: identificador de sesión o token alfanumerico
        :param prompt:
        :return: devuelve el listado con los mensajes por id_sesion
        """
        prompts = []
        try:
            conn = self.obtener_db_conexion()
            cursor = conn.cursor()
            consulta = f"INSERT INTO {self.tabla_sesiones} (id_session, prompt) VALUES(?, ?);"
            cursor.execute(consulta, (id_session, prompt))
            logging.info("Consulta ejecutada OK")
            conn.commit()
            conn.close()

        except Exception as e:
            logging.exception(f"Ocurrió un error al insertar los datos para la sesion {id_session}")

        try:
            conn = self.obtener_db_conexion()
            res = conn.execute(f"SELECT * FROM {self.tabla_sesiones} where id_session='{id_session}';").fetchall()
            logging.info("Consulta ejecutada OK")

            for item in res:
                prompts.append(item[2])

            conn.close()
        except Exception as e:
            logging.exception(f"Ocurrió un error al obtener datos para la sesion {id_session}")

        return prompts
