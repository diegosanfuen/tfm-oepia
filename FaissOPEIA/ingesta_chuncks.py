import numpy as np
import faiss
from collections import namedtuple
import pandas as pd
from langchain_core.documents.base import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pickle as pkl
import warnings
import yaml
from pathlib import Path
import logging, glob, os, datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Variables de entorno necesarias
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PROJECT_ROOT'] = r'/content/tfm-oepia'

# Ignorar warnings específicos de huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

# Definición de la estructura del documento
Document = namedtuple('Document', ['page_content', 'metadata'])


# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

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

class DocumentProcessor:
    def __init__(self):
        self.documentos = []
        logger.debug(f'Volcamos toda la informacion del fichero de configuracion: {config}')
        # Parametros externos configuracion
        self.embedding_llm = OllamaEmbeddings(
            model=config['vectorial_database']['parameters_tokenizador']['name_model_tokenizador'])
        self.ruta_db = Path(config['ruta_base']) / Path(config['vectorial_database']['ruta']) / Path(
            config['vectorial_database']['serialized_database'])
        logger.debug(f'Leemos la configuracion Ruta de la Base de datos: {self.ruta_db}')

    def convertir_csv_a_documentos(self, dataframe, col_text, cols_metadata):
        """
        Convierte en documentos todos los csvs extraidos a través del webscrapping de las distintas fuentes.
        :param dataframe: Dataframe (csvs leidos)
        :param col_text: Campo a indexar
        :param cols_metadata: Campos a la METADATA
        :return: Lista de documentos
        """
        documentos = []
        # Iterar sobre cada fila del DataFrame
        for index, row in dataframe.iterrows():
            # Crear un objeto Document
            doc = Document(
                page_content=row[col_text],  # El contenido principal del documento
                metadata={campo: row[campo] for campo in cols_metadata}
            )
            # Añadir el Documento a la lista
            documentos.append(doc)
        self.documentos = documentos
        return documentos

    def dividir_en_chunks_con_overlap(self, documentos, chunk_size, overlap_size):
        """
        Divide la lista de documentos en chunks con solapamiento.
        :param documentos: Lista de documentos
        :param chunk_size: Tamaño del chunk
        :param overlap_size: Tamaño del solapamiento
        :return: Lista de chunks
        """
        chunks = []
        for i in range(0, len(documentos), chunk_size - overlap_size):
            end = min(i + chunk_size, len(documentos))
            chunk = documentos[i:end]
            chunks.append(chunk)
        return chunks

    def indexar_chunks(self, chunks, dimension):
        """
        Indexa cada chunk utilizando Faiss.
        :param chunks: Lista de chunks
        :param dimension: Dimensión de los vectores
        :return: Lista de índices
        """
        indices = []
        for chunk in chunks:
            # Supongamos que transformamos los documentos en vectores de alguna manera
            vectors = np.array([self.transformar_documento_a_vector(doc) for doc in chunk]).astype('float32')
            index = faiss.IndexFlatL2(dimension)  # Indexación usando L2
            index.add(vectors)
            indices.append(index)
        return indices

    def transformar_documento_a_vector(self, doc):
        """
        Transforma un documento en un vector.
        Aquí se debe definir cómo transformar el texto en vectores (e.g., usando embeddings).
        :param doc: Documento a transformar
        :return: Vector del documento
        """
        # Ejemplo simple: convertir el contenido a un vector de longitud fija
        vector = np.random.rand(64)  # Placeholder, usar un método real de conversión a vector
        return vector

    def generar_vector_store(self):
        """
        Gener los vectores usando el LLM configurado en el fichero config.xml
        :return:
        """
        text_splitter = RecursiveCharacterTextSplitter()
        documents_embd = text_splitter.split_documents(self.documentos)
        self.vector_index = FAISS.from_documents(documents_embd, self.embedding_llm)
        self.retriever = self.vector_index.as_retriever()

    def persistir_bbdd_vectorial(self):
        """
        Persiste la bbdd vectorial a través de un pickle no es la mejor opción pero la usé por portabilidad
        :return:
        """
        if os.path.exists(self.ruta_db):
            # Si existe, borra su contenido
            for filename in os.listdir(self.ruta_db):
                file_path = os.path.join(self.ruta_db, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Error al eliminar {file_path}. Razón: {e}')
        else:
            # Si no existe, crea el directorio
            os.makedirs(self.ruta_db)

        try:
            with open(self.ruta_db / Path(config['vectorial_database']['file_vector_index']), 'wb') as archivo:
                pkl.dump(self.vector_index, archivo)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar guardar la base de datos de embbedings vector Index: {e}')

        try:
            with open(self.ruta_db / Path(config['vectorial_database']['file_retriever']), 'wb') as archivo:
                pkl.dump(self.retriever, archivo)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar guardar la base de datos de embbedings tipo retriever: {e}')

    def inicialize_db_vect(self):
        """
        Inicializa la base de datos vectotrial a partir de los csv extraidos de la web
        :return:
        """
        if config['vectorial_database']['enabled_ingest'] == 0:
            logger.info("La ingesta de datos en la base de datos vectorial fue deshabilitada")
            exit(0)

        path_csv = Path(config['ruta_base']) / config['scrapping']['ruta'] / config['scrapping'][
            'descarga_datos'] / '*.csv'
        path_csv_str = str(path_csv)
        archivos_csv = glob.glob(path_csv_str)
        dataframes = []
        for archivo in archivos_csv:
            # Lee el archivo CSV y añádelo a la lista de DataFrames
            df = pd.read_csv(archivo, sep='|')
            dataframes.append(df)

        self.dataset = pd.concat(dataframes, ignore_index=True)

        self.convertir_pandas_lista_documentos(self.dataset,
                                               config['scrapping']['dataset_index']['campo_texto'][0],
                                               config['scrapping']['dataset_index']['campos_metadata'])
        self.generar_vector_store()
        self.persistir_bbdd_vectorial()

    def getRetriver(self):
        """
        Obtiene el retreiver para probarlo.
        :return:
        """
        return self.retriever

if __name__ == '__main__':
    """
    Método principal para probar la clase.
    """
    #BDVect = ingesta()
    #BDVect.inicialize_db_vect()
    #retriever = BDVect.getRetriver()

    # Ejemplo de uso:
    df = pd.read_csv('/content/tfm-oepia/ObtencionDatos/datos/csv_boes_oferta_publica.csv', sep='|')  # Leer un CSV como DataFrame
    col_text = 'texto'  # Columna del texto principal
    cols_metadata = ['url', 'titulo']  # Columnas de metadata

    processor = DocumentProcessor()
    documentos = processor.convertir_csv_a_documentos(df, col_text, cols_metadata)
    chunks = processor.dividir_en_chunks_con_overlap(documentos, chunk_size=100, overlap_size=10)
    indices = processor.indexar_chunks(chunks, dimension=64)

    # Ahora tienes tus documentos divididos en chunks con solapamiento y cada chunk indexado con Faiss
