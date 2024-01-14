import pandas as pd
from IPython.display import display

data = {'name': ["john", "smith"], 'location': ["LA", "new york"], 'age': [25, 26]}
dataPandas = pd.DataFrame(data)

display(dataPandas)
display(dataPandas[dataPandas.age > 25])