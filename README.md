ANÁLISIS DE RIESGO CREDITICIO CON MODELOS ANFIS GRID Y FUZZY C MEANS.

1. anfis_credito_reducido:
   Usa training_data_reducido (53k filas), que corresponde al 30% de la data de training original (150k filas) y respeta el
   desbalance de clases (88% para Default de clase 0 y 11% para Default de clase 1)
2. comparar_modelos:
   Compara los resultados de 3 modelos: Anfis grid [2 2 2 3], [3 3 3 3] y fuzzy c means con 7 clústers. Asimismo, se analiza el
   umbral que maximiza f1-score en cada caso.
   Se seleccionaron el modelo anfis grid [2 2 2 3] y fuzzy c means. Utiliza data balanceada por medio de undersampling (TRAINING.csv)
   y se redujo el test a 10k (TESTING_DATA), respetando el desbalance original. 
4. Modelo_elegido:
   Analiza con mayor profundidad los modelos elegidos para detectar sesgos en las variables demográficas (Edad, Estado Civil, Educación). 
