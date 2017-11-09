#Perceptrón
import numpy as np

def threshold(x):
	if(x>=1):
		return 1
	elif(x<1):
		return 0

def update(error, j):
	global weights
	global input_data
	a = 0.2
	for i in range(2):
		weights['node_0'][i] = weights['node_0'][i] + (a*error*input_data[j][i])
	
#Datos de entrada
input_data = np.array([np.array([0,0]),np.array([0,1]),np.array([1,0]),np.array([1,1])])
weights = {'node_0': np.array([-0.2,0.4])}
output = np.array([0,0,0,1])
real = np.array([0,0,0,0])
epoch = 15


#epoch
for j in range(epoch):
	print("Epoch ","(X1,X2) ","SalidaE ","SalidaR ","Error ","W1   ","W2\t")
	for i in range(4):
		valor = (input_data[i] * weights['node_0']).sum()
		#print('Imprimir valor',valor) #BORRAR ESTO
		salida = threshold(valor)
		real[i] = salida
		if(real[i] == output[i]):
			error = 0
		elif(real[i] != output[i]):
			if(valor > 1):
				error = -1
			elif(valor < 1):
				error = 1
		if(error != 0):
			update(error, i)
		print("{}	{}	 {}	  {}	 {}	 {:.2f}	{:.2f}\t".format(j+1,input_data[i,],output[i],real[i] ,error,weights['node_0'][0],weights['node_0'][1]))	
	#Terminar
	finaliza = (output == real)
	if(finaliza.all()):
		print('\nTermino')
		break
		

