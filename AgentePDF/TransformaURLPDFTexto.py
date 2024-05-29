# Importamos las librerias
import requests
import pdfplumber
import io, datetime
import os, logging, yaml
from pathlib import Path


class TransformaURLPDFTexto:
    """

    """
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

    @staticmethod
    def extraerPDFURL(url):
        # Obtener el PDF desde la web
        response = requests.get(url)
        text = ""

        try:
            if response.status_code == 200:
                # Abrir el PDF desde un stream de bytes
                with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                    text = ''
                    # Extraer texto de cada página
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
            else:
                logging.INFO("Error al descargar el archivo")
        except Exception as e:
            logging.ERROR(f"Error al decargar y parsear el fichero PDF puede que no sea un PDF realemente? {e}")
        return text
