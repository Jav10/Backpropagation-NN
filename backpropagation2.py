#Importamos la libreria numpy
import numpy as np

#Función de activación ReLu
def Relu(input):
	'''Funcion de activacion - ReLu'''
	output = max(input, 0)
	return(output)

def predict_with_network(input_data_row, weights):
	#Calcular el valor del nodo_0
	node_0_input = (input_data_row * weights['node_0']).sum()
	#Aplicando ReLu
	node_0_output = Relu(node_0_input)
	
	#Calcular el valor del nodo_1
	node_1_input = (input_data_row * weights['node_1']).sum()
	#Aplicando ReLu
	node_1_output = Relu(node_1_input)
	
	#Colocamos los valores en un array: hidden_layer_outputs
	hidden_layer_outputs = np.array([node_0_output, node_1_output])
	
	#Calculamos la salida del modelo
	input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
	model_output = Relu(input_to_final_layer)
	
	#Retornamos la salida del modelo
	return(model_output)
	
#Datos de entrada
input_data = [np.array([3, 5]), np.array([ 1, -1]), np.array([0, 0]), np.array([8, 4])]
weights = {'node_0': np.array([2,4]), 'node_1': np.array([4,-5]), 'output': np.array([2,7])}

#Creamos una lista para guardar las predicciones
results = []
for input_data_row in input_data:
	#Agregamos las predicciones a la lista
	results.append(predict_with_network(input_data_row, weights))
	
print(results)	

	