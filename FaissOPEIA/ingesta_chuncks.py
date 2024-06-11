class ingesta():
    """
       Clase que gestiona la ingesta de datos hacia la base de datos vectorial de FAISS.

       Esta clase está diseñada para crear y poblar una base de datos FAISS
       utilizando datos vectoriales que se extraen de archivos CSV. Estos archivos
       CSV son descargados y preparados por un módulo de obtención de datos externo.

       Métodos:
           convertir_pandas_lista_documentos(self, dataframe, col_text, cols_metadata):
           Convierte los pandas dataframes obtenidos a partir de los CSVs en documentos
           generar_vector_store(self): Genera los vectores con los documentos
           persistir_bbdd_vectorial(self,): persiste la base de datos vectorial en disco
           inicialize_db_vect(self,): Inicia todos los procesos en su orden
           getRetriver(self,): Obtiene el Retreiver para probarlo

    """

    def __init__(self):
        logger.debug(f'Volcamos toda la informacion del fichero de configuracion: {config}')
        # Parametros externos configuracion
        self.embedding_llm = OllamaEmbeddings(
            model=config['vectorial_database']['parameters_tokenizador']['name_model_tokenizador'])
        self.ruta_db = Path(config['ruta_base']) / Path(config['vectorial_database']['ruta']) / Path(
            config['vectorial_database']['serialized_database'])
        logger.debug(f'Leemos la configuracion Ruta de la Base de datos: {self.ruta_db}')

    def convertir_pandas_lista_documentos(self, dataframe: pd,
                                          col_text: str,
                                          cols_metadata: list):
        """
        Convierte en documentos todos los csvs extraidos a través del webscrapping de las distintas fuentes.
        :param dataframe: Dataframe (csvs leidos)
        :param col_text: Campo a indexar
        :param cols_metadata: Campos a la METADATA
        :return:
        """
        documentos = []
        # Iterar sobre cada fila del DataFrame
        for index, row in dataframe.iterrows():
            # Crear un objeto Document
            doc = Document(
                page_content=row[col_text],  # El contenido principal del documento
                metadata={
                    campo: row[campo] for campo in cols_metadata
                }
            )
            # Añadir el Documento a la lista
            documentos.append(doc)
        self.documentos = documentos

    def generar_vector_store(self):
        """
        Genera los vectores usando el LLM configurado en el fichero config.yml
        :return:
        """
        # Configurar el splitter con chunk_size y overlap_size
        chunk_size = 1000  # Tamaño de cada chunk
        overlap_size = 200  # Tamaño del solapamiento

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size
        )

        # Dividir documentos en chunks
        documents_chunks = text_splitter.split_documents(self.documentos)

        # Generar la base de datos vectorial con FAISS
        self.vector_index = FAISS.from_documents(documents_chunks, self.embedding_llm)
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
    BDVect = ingesta()
    BDVect.inicialize_db_vect()
    retriever = BDVect.getRetriver()
