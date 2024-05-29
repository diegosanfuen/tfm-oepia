"""
Script principal que descarga via configuración el contenido del BOE y del BOCYL y lo deja accesible por el ingestador de FAISS
"""

# Importamos librerias
from DescargaBOE import DescargaBOE
from DescargaBOCyL import DescargaBOCyL
import os, datetime
import time, logging, yaml
from pathlib import Path

# Clase principal
os.environ['PROJECT_ROOT'] = r'/content/content/tfm-oepia'

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['scrapping']['ruta'])
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

BOCyL = DescargaBOCyL()
BOCyL.initialize_download()

BOE = DescargaBOE()
BOE.initialize_download()
