#Backpropagation algorithm
import numpy as np

#Función de activación ReLu
def Relu(input):
	'''Funcion de activacion - ReLu'''
	output = max(input, 0)
	return(output)

#Datos de entrada
input_data = np.array([3,5])
weights = {'node_0': np.array([2,4]), 'node_1': np.array([4,-5]), 'output': np.array([2,7])}

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()
#Aplicando ReLu
node_0_output = Relu(node_0_value)

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()
#Aplicando ReLu
node_1_output = Relu(node_1_value)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output -39
print(output)