import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib3 as url
import datetime
import string
from urllib.parse import urlparse, urlunparse


class DescargaBOE:
    """
    Clase que permite la descarga del BOE en lo referente a las Resoluciones relacionadas con las convocatorias de Oposiciones
    Para instanciar la clase:
    MiClase = DescsargaBOE()
    Para fijar el Offset
    MiClase.establecer_offset(offset)
    """

    def __init__(self):
        """
        Generador de la clase no recibe parámetros
        establece las variables internas
        fecha_actual, url_patron, dominio u dataset con los boes
        """
        # Obtiene la fecha y hora actual
        self.fecha_actual = datetime.datetime.now()
        self.url_patron = string.Template("https://www.boe.es/boe/dias/$anio/$mes/$dia/index.php?s=2B")
        self.dominio = "https://www.boe.es"
        self.dataset_boes = pd.DataFrame({'url': [],
                                          'titulo': [],
                                          'texto': []})

    def quitar_etiquetas_html(self, cadena_html: str) -> str:
        """
        Método Helper para la eliminación de etiquetas HTML de los textos parseados
        uso:
        Entrada: Texto con etiquetas HTML
        Salida: Mismo Texto sin etiquetas HTML
        self.quitar_etiquetas_html(Texto)
        """
        # Parsear la cadena HTML
        soup = BeautifulSoup(cadena_html, 'html.parser')
        # Obtener solo el texto sin etiquetas HTML
        texto = soup.get_text(separator='')
        texto = texto.replace('[', '')
        texto = texto.replace(']', '')
        return texto

    def establecer_offset(self, offset: int):
        """
        Método que estalece el OFFSET definido como el número de días a partir de la fecha
        actual desde la que se quiere descargar los BOES
        Si instanciamos
        MiClase.establecer_offset(5)
        Inspeccionaremos los BOES de hace 5 días
        Entrada: Offset Es un etero
        Salida: Variables internas de la clase (URLS de los BOES)
        """
        fecha_calculada = self.fecha_actual - datetime.timedelta(days=offset)
        anio = fecha_calculada.year
        mes = str(fecha_calculada.month).zfill(2)
        dia = str(fecha_calculada.day).zfill(2)
        fecha = {'anio': anio,
                 'mes': mes,
                 'dia': dia}
        self.url_busqueda = self.url_patron.substitute(anio=fecha['anio'],
                                                       mes=fecha['mes'],
                                                       dia=fecha['dia'])

    def buscar_urls_xmls(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos las URLS
        de las disposiciones relativas a las ofertas de empelo público es decir
        Sección II B del BOE
        Uso
        self.buscar_urls_xmls()
        """

        url = self.url_busqueda
        parsed_url = urlparse(url)

        dominio = parsed_url.netloc

        response = requests.get(url)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')

        titulo_buscado = "Otros formatos"

        enlaces_con_titulo = soup.find_all('a', string=titulo_buscado)

        lista_urls = []
        for enlace in enlaces_con_titulo:
            url_obtenida = f'https://{dominio}{enlace["href"]}'

            parsed_url = urlparse(url_obtenida)
            parsed_url_lista = list(parsed_url)
            parsed_url_lista[2] = 'diario_boe/xml.php'

            # Convertir la lista de nuevo a un objeto ParseResult
            parsed_url_modificada = urlparse(urlunparse(parsed_url_lista))
            lista_urls.append(urlunparse(parsed_url_modificada))

        self.lista_urls = lista_urls

    def obtener_lista_xmls(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos los XMLs
        de las disposiciones relativas a las ofertas de empelo público es decir
        Sección II B del BOE
        Uso
        self.obtener_lista_xmls()
        """
        lista_respuestas = []
        for url in self.lista_urls:
            # url = 'https://www.boe.es/diario_boe/xml.php?id=BOE-A-2021-10344'
            headers = {'accept': 'application/xml;q=0.9, */*;q=0.8'}
            response = requests.get(url, headers=headers)
            lista_respuestas.append(response.text)
        self.lista_xmls = lista_respuestas

    def obtener_lista_titulos(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos los titulos
        de las disposiciones relativas a las ofertas de empelo público es decir
        Sección II B del BOE
        Uso
        self.obtener_lista_titulos()
        """
        lista_titulos = []
        for XML in self.lista_xmls:
            soup = BeautifulSoup(XML, "xml")
            titulo = soup.find("titulo")
            lista_titulos.append(titulo.get_text())
        self.lista_titulos = lista_titulos

    def obtener_lista_textos(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos los textos
        de las disposiciones relativas a las ofertas de empelo público es decir
        Sección II B del BOE
        Uso
        self.obtener_lista_textos()
        """
        lista_textos = []
        for XML in self.lista_xmls:
            textos = ""
            soup = BeautifulSoup(XML, "xml")
            text = soup.find_all("texto")
            lista_textos.append(str(text))
        self.lista_textos = lista_textos

    def obtener_lista_urls_pdf(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos las urls pdfs
        de las disposiciones relativas a las ofertas de empelo público es decir
        Sección II B del BOE
        Uso
        self.obtener_lista_urls_pdf()
        """
        lista_urls_pdf = []
        for XML in self.lista_xmls:
            textos = ""
            soup = BeautifulSoup(XML, "xml")
            url_pdf = soup.find_all("url_pdf")
            lista_urls_pdf.append(f'{self.dominio}{str(self.quitar_etiquetas_html(str(url_pdf)))}')
        self.lista_urls_pdf = lista_urls_pdf

    def generar_dataset(self) -> int:
        """
        Con los parámetros obtenidos de establecer_offset, generamos el dataset pandas
        de las disposiciones relativas a las ofertas de empelo público es decir
        Sección II B del BOE
        Uso
        self.generar_dataset()
        Salida: Conteo de filas del dataset
        """
        self.buscar_urls_xmls()
        self.obtener_lista_xmls()
        self.obtener_lista_titulos()
        self.obtener_lista_textos()
        self.obtener_lista_urls_pdf()
        dataset_capturado = pd.DataFrame({'url': self.lista_urls_pdf,
                                          'titulo': self.lista_titulos,
                                          'texto': self.lista_textos})

        self.dataset_boes = pd.concat([self.dataset_boes, dataset_capturado], ignore_index=True)
        return self.dataset_boes.shape[0]

    def obtener_dataset_final(self):
        """
        Finalmente devolvemos a la rutina principal el contenido del dataset completo
        MiClase.obtener_dataset_final()
        Salida: Dataset Completo
        """
        return self.dataset_boes
