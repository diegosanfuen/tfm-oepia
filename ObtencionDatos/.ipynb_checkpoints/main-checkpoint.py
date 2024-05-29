# Rutina principal
from DescargaBOE import DescargaBOE
import sys
import os

# Obtener la ruta del script actual
ruta_script = os.path.abspath("__file__")
directorio_proyecto = os.path.dirname(ruta_script)


BOE = DescargaBOE()
i = 0
while True:
    BOE.establecer_offset(i)
    if(BOE.generar_dataset() > 100):
        break
    i += 1
BOE.obtener_dataset_final()

df = BOE.obtener_dataset_final()
df.to_csv(f'{directorio_proyecto}/datos/csv_boes_oferta_publica.csv', sep='|')