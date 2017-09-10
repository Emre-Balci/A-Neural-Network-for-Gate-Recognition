#!/usr/bin/python
# -*- coding= utf-8 -*-
from __future__ import print_function
import numpy
import random

#A neural network example that learns to behave like a logic gate.
#One of the four logic gates "or", "and", "xor" and "nand" can be chosen for this example.
#Lists are used to hold nodes and weights.
#For example; layer_list[i][j] is the jth node in the ith layer.
#Nodes and weights are instances of their own classes.
class Network:
	
	def __init__(self, input_list, output_list, test_input_list, test_output_list):
		#constructor of the neural network.
		#"input_list" and "output_list" are the data sets which will be used in training.
		#"test_input_list" and "test_output_list" will be used to test the network after training.
		#"gamma_wight" and "gamma_bias" are the learning rates which will be used in backpropagation phase.
		self.layer_list = list()
		self.weight_list = list()
		self.input_list = input_list
		self.output_list = output_list
		self.test_input_list = test_input_list
		self.test_output_list = test_output_list
		self.layer_no = 0
		self.total_error = 0.0
		self.gamma_weight = 1.50
		self.gamma_bias = 1.50
	

	def addlayer(self, number_of_nodes):
		#Adds another layer to the network every time it is called.
		#"number_of_nodes" defines that how many nodes will be in the layer.
		#All bias values are set to 1 initially but they will be uptaded later in backpropagation phase.
		layer = list()
		for i in range(number_of_nodes):
			layer.append(addNode(1.0, self.layer_no, i))
		self.layer_list.append(layer)
		self.layer_no += 1
	
	def initializeWeights(self):
		#All weights are set to random values between 1 and 0.
		number_of_layers = len(self.layer_list)
		for i in range(number_of_layers):
			if( i < (number_of_layers - 1) ):
				number_of_nodes = len(self.layer_list[i])
				number_of_nodes_nextlayer = len(self.layer_list[i+1])
				for j in range(number_of_nodes):
					for k in range(number_of_nodes_nextlayer):
						self.weight_list.append(addWeight(numpy.random.randn()%1, 
						self.layer_list[i][j], self.layer_list[i+1][k]))
	
	def setInput(self, input_list):
		#Sets the inputs vales as the outputs values of the first layer nodes.
		for i in range(len(input_list)):
			self.layer_list[0][i].node_output(input_list[i])
	
	def forward_pass(self):
		#Forward-pass phase.
		z = 0.0
		for i in range(1, len(self.layer_list)):
			for j in range(len(self.layer_list[i])):
				for k in range(len(self.weight_list)):
					if (self.weight_list[k].goes_to == self.layer_list[i][j]):
						z = z + calculate_z(self.weight_list[k].value, float(self.weight_list[k].comes_from.out))
				self.layer_list[i][j].node_output(sigmoid(z+self.layer_list[i][j].bias))
				z = 0
	
	def backpropagation(self, output_list):
		#Backpropagation phase. Every weight is updated individually.
		updated_values = list()
		for w in self.weight_list:
			if(w.goes_to.layer_no == (len(self.layer_list) - 1)):
				#checks if the weight is connected to output layer directly
				#if so weight's error will only contribute to one output node
				#and it is not needed to involve other output nodes in delta
				delta = self.calculate_delta(output_list, w.goes_to.index)
				new_value = w.value - (self.gamma_weight*(delta*w.comes_from.out))
				updated_values.append(new_value)
			else:
				#if weight is not directly connected to the output layer
				#it will contribute to the error of the all output nodes since it is a fully-connected network
				#that's why all output nodes are needed to be involved in delta
				delta_output_layer = 0.0
				for n in self.layer_list[len(self.layer_list)-1]:
					delta_output_layer = ( delta_output_layer +
					((self.calculate_delta(output_list, n.index))*
					self.weight_list[(len(self.weight_list)-
					((len(self.layer_list[len(self.layer_list)-2]))*
					(len(self.layer_list[len(self.layer_list)-1])))+
					((len(self.layer_list[len(self.layer_list)-2]))*n.index)+n.index)].value))
				delta = (delta_output_layer)*((w.goes_to.out)*(1.0 - w.goes_to.out))
				new_value = w.value - (self.gamma_weight*(delta*w.comes_from.out))
				updated_values.append(new_value)
		for i in range(len(self.weight_list)):
			self.weight_list[i].value = updated_values[i]
		#Lastly updates biases
		self.update_bias(output_list)
	
	def calculate_delta(self, output_list, node_index):
		#Calculates delta for the nodes in the hidden layers
		last_layer = (len(self.layer_list)-1)
		delta = ((-1.0)*
		((output_list[node_index])  - (self.layer_list[last_layer][node_index].out))*
		((self.layer_list[last_layer][node_index].out)*(1.0 - (self.layer_list[last_layer][node_index].out))))
		return delta
	
	def update_bias(self,  output_list):
		updated_bias = list()
		for i in range (len(self.layer_list)):
			for n in self.layer_list[i]:
				if(i == len(self.layer_list)-1):
					delta = self.calculate_delta(output_list,n.index)
					updated_bias.append(self.gamma_bias*delta)
				elif(i>0 and i<(len(self.layer_list)-1)):
					delta_output_layer = 0.0
					for k in self.layer_list[len(self.layer_list)-1]:
						delta_output_layer = ( delta_output_layer +
						((self.calculate_delta(output_list, k.index))*
						self.weight_list[(len(self.weight_list)-
						((len(self.layer_list[len(self.layer_list)-2]))*
						(len(self.layer_list[len(self.layer_list)-1])))+
						((len(self.layer_list[len(self.layer_list)-2]))*k.index)+k.index)].value))
					delta = (delta_output_layer)*((n.out)*(1.0-n.out))
					updated_bias.append(self.gamma_bias*delta)
				else:
					pass
		counter = 0
		for i in range (len(self.layer_list)):
			for n in self.layer_list[i]:
				if(i > 0):
					n.update_bias(updated_bias[counter])
					counter += 1
				else:
					pass

	def train(self):
		#Trains the network with the given data set.
		print ("\nTraining The Network...")
		for i in range(100):
			print ('.', end='')
		for i in range(len(self.input_list)):
			self.setInput(self.input_list[i])
			self.forward_pass()
			self.backpropagation(self.output_list[i])
			self.forward_pass()
			self.print_progress(i+1)
		print ("\nTraining Completed.")
	
	def test(self):
		#Tests the network without backpropagation.
		self.total_error = 0.0
		for i in range(len(self.test_input_list)):
			self.setInput(self.test_input_list[i])
			self.forward_pass()
			for j in range (len(self.layer_list[len(self.layer_list)-1])):
				self.total_error += abs(self.test_output_list[i] - self.layer_list[len(self.layer_list)-1][j].out)
			self.evaluate(self.test_input_list[i], self.test_output_list[i], i)
	
	def evaluate(self, list_input, list_output, test_number):
		#Outputs the test result.
		print ("\nTest: "+ str(test_number+1))
		for i in range(len(list_input)):
			print ("Input "+ str(i+1)+ ": "+ str(list_input[i]))
		print ("Actual Output: "+ str(list_output[0]))
		print ("Network's Output: "+ str(self.layer_list[len(self.layer_list)-1][0].out))
	
	def avg_error(self):
		#Calculates and prints the average error.
		average_error = (self.total_error/float(len(self.test_output_list)))
		print ("\nAverage Error: "+ str(average_error)+ "(%"+ str(int(average_error*100.0))+ ")" )
	
	def print_progress(self, train_counter):
		#Prints the progress of the training phase.
		for i in range(100):
			print ("\b",end="")
		for i in range(train_counter/(len(self.input_list)/100)):
			print ("#",end="")
		for i in range((len(self.input_list)-train_counter)/(len(self.input_list)/100)):
			print (".",end="")
	
class Weight:

	def __init__(self, value, comes_from, goes_to):
		#"comes_from" and "goes_to are" the nodes the weight connects.
		self.value = value
		self.comes_from = comes_from
		self.goes_to = goes_to

class Node:

	def __init__(self, bias, layer_no, index):
		#"layer_no" is the index of the node's layer. Indexes are starts from the input layer.
		#"index" is index of the node's in the layer it belongs to.
		self.bias = bias
		self.layer_no = layer_no
		self.index = index
		self.net = 0.0
		self.out = 0.0

	def node_input(self, net):
		self.net = net

	def node_output(self, out):
		self.out = out

	def update_bias(self, bias):
		self.bias -= bias

def addNode(bias, layer_no, index):
	nodeInstance = Node(bias, layer_no, index)
	return nodeInstance


def addWeight(value, comes_from, goes_to):
	weightInstance = Weight(value, comes_from, goes_to)
	return weightInstance

def calculate_z(w, i):
	return (w*i)

def sigmoid(z):
	return (1.0 / ( 1.0 + numpy.exp(-z) ))

def main(gate):
	train = 10000 #number of train data sets. XOR gate might need to be trained 100000 times to work correctly.
	test = 50 #number of test data sets
	train_list_input = list()
	train_list_output = list()
	train_input = list()
	train_output = list()
	test_list_input = list()
	test_list_output = list()

	for i in range(train):
		#Initializing the train data sets
		x = random.randint(0,1)
		y = random.randint(0,1)
		train_input.append(float(x))
		train_input.append(float(y))
		train_list_input.append(train_input[:])
		del train_input[:]
		if (gate == "XOR") or (gate == "xor"):
			if (x == 0 and y == 0) or (x == 1 and y == 1):
				train_output.append(0.0)
			else:
				train_output.append(1.0)
		elif (gate == "AND") or (gate == "and"):
			if (x == 1 and y == 1):
				train_output.append(1.0)
			else:
				train_output.append(0.0)
		elif (gate == "OR") or (gate == "or"):
			if (x == 0 and y == 0):
				train_output.append(0.0)
			else:
				train_output.append(1.0)
		elif (gate == "NAND") or (gate == "nand"):
			if (x == 1 and y == 1):
				train_output.append(0.0)
			else:
				train_output.append(1.0)
		else:
			print ("Invalid gate name!\n")
			return
		train_list_output.append(train_output[:])
		del train_output[:]
	
	for i in range(test):
		#Randomly choosing test data sets from train data sets.
		x = random.randint(0, train-1)
		test_list_input.append(train_list_input[x][:])
		test_list_output.append(train_list_output[x][:])

	#creating network.
	myNetwork = Network(train_list_input, train_list_output, test_list_input, test_list_output)
	myNetwork.addlayer(2)
	myNetwork.addlayer(5)
	myNetwork.addlayer(1)
	myNetwork.initializeWeights()
	#testing before and after training to see the difference.
	myNetwork.test() 
	myNetwork.avg_error()
	myNetwork.train()
	myNetwork.test()
	myNetwork.avg_error()


gate = raw_input("Enter a gate name or type \"exit\" to terminate the program\n")
while not (gate == "exit"):
	main(gate)
	gate = raw_input("\nEnter another gate name or type \"exit\" to terminate the program\n")
