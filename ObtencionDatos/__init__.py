# Definimos una variable
nombre_paquete = "ObtenerDatosCollab"
import os
os.environ['PROJECT_ROOT'] = r'/content/recuperacion_informacion_modelos_lenguaje/tfm'

# Importamos los módulos que queremos que estén disponibles
from ObtenerDatosCollab.DescargaBOE import DescargaBOE
from ObtenerDatosCollab.DescargaBOCyL import DescargaBOCyL
