# Importamos librerias
import pickle as pkl
import warnings
import yaml
from pathlib import Path
import logging, os, datetime

# Definimos variables de entorno necesarias
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PROJECT_ROOT'] = r'/content/content/tfm-oepia'

# Ignorar warnings específicos de huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

# Abrir y leer el archivo YAML de configuración
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Leemos variables de entorno
PATH_BASE = Path(config['ruta_base'])
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


class carga():
    """
       Clase que gestiona la carga de una base de datos vectorial de FAISS.

       Esta clase está diseñada para inicializar y poblar una base de datos FAISS
       utilizando datos vectoriales que se extraen de archivos CSV. Estos archivos
       CSV son descargados y preparados por un módulo de obtención de datos externo.

       Métodos:
           cargar_db_Vectorial(self,): Carga un archivo pkl desde la ruta especificada.
           getRetriver(self): Inicializa el Retriver de FAISS.
           inialize_retriever(self,): Realiza toda la logica de la carga
       """

    def __init__(self):
        logger.debug(f'Volcamos toda la informacion del fichero de configuracion: {config}')
        # Parametros externos configuracion
        self.ruta_db = Path(config['ruta_base']) / Path(config['vectorial_database']['ruta']) / Path(
            config['vectorial_database']['serialized_database'])
        logger.debug(f'Leemos la configuracion Ruta de la Base de datos: {self.ruta_db}')
        self.cargar_db_Vectorial()

    def cargar_db_Vectorial(self):
        """
        Carga la base de datos vectorial partiendo de los parámetros obtenidos del fichero de configuración
        :return:
        """
        try:
            with open(self.ruta_db / Path(config['vectorial_database']['file_vector_index']), 'rb') as archivo:
                self.vector_index = pkl.load(archivo)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar leer la base de datos de embbedings vector Index: {e}')

        try:
            with open(self.ruta_db / Path(config['vectorial_database']['file_retriever']), 'rb') as archivo:
                self.retriever = pkl.load(archivo)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar guardar la base de datos de embbedings tipo retriever: {e}')

    def getRetriver(self):
        """
        Devuelve el Retreiver que se incluirá en la cadena de blockchain del proyecto OEPIA
        :return: retreiver FAISS
        """
        return self.retriever

    def inialize_retriever(self):
        """
        Ejecuta en el orden adevcuado todos los métodos para obtener el retreiver FAISS
        :return: retreiver FAISS
        """
        self.cargar_db_Vectorial()
        return self.getRetriver()


if __name__ == '__main__':
    """
    Método principal para probar la clase aisladamente.
    """
    BDVect = carga()
    retriever = BDVect.getRetriver()
