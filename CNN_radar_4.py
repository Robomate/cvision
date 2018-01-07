#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#=======================================================================
#Purpose: Classification with a CNN
#
#Model:   4 Layer Convolutional Neural Network
#
#Inputs:  2D UWB radar data (SAR images)
#
#Output:  8 Classes (bed positions)
#Version: 11/2017 Roboball (MattK.)
#=======================================================================
'''
import numpy as np
import tensorflow as tf
import random
import os, re
import datetime 
import matplotlib.pyplot as plt
import matplotlib.image as pltim 
import scipy.io
import pandas as pd
# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

# init paths:
#path = "/home/root_alien/Desktop/AML_Lab/" #on alien2
path = "/home/roboball/Desktop/AML_Lab/" #on laptop
filepath = path + '04_nn_training/data/' #on laptop

########################################################################
# init globals
########################################################################

# init unit testing:
unit_test = False         # boolean, default is False

# model parameter
modtype = "CNN" 
modlayer = 4
train_split = 0.9 # train/test split ratio
randomint = 1 # fix initial randomness:
norm = 'no_norm'
optimizing = 'adam'
classifier = 'softmax'
save_thres = 0.9 # save only good models	

epochs = 20
learnrate= 0.01
batchsize_train = 50
batchsize_test = 10	

# init activation function:
actdict = ["tanh", "relu", "selu","sigmoid"]
acttype = actdict[1]

########################################################################
# init functions
########################################################################

def get_filelist(filepath):
	filelist = []
	for index, filename in enumerate(sorted(os.listdir(filepath))):
		filelist.append(filename)
		#print '{0:02d}. {1}'.format(index + 1, filename)
	print('=============================================')
	print("load filenames")
	print('=============================================')
	print(len(filelist),'data_files loaded\n')
	return filelist

def load_mat_data(files):
	'''load .mat files'''
	label_data = np.empty(shape=[0],dtype=int)
	label_data_extend = np.empty(shape=[0],dtype=int)
	dict_list = []
	num_im_list = [] # length of image data
	mat = scipy.io.loadmat('data/' + files[0])
	label_dict = mat['label_dict'] # bed positions
	for label in label_dict:
		dict_list.append(label.rstrip())
	classes = len(dict_list)
	print(str(classes)+' Label Categories:')
	for entry in dict_list:
		print(entry)
	print()
	image_data  = np.array(mat['image_tensor'])
	num_im = image_data.shape[2]
	num_im_list.append(num_im)
	label_array = np.array(mat['label'])
	for pos in range(1,len(files)):
		mat = scipy.io.loadmat('data/' + files[pos])
		# load images
		images = np.array(mat['image_tensor'])
		num_im = image_data.shape[2]
		num_im_list.append(num_im)
		#print(images.shape)
		image_data = np.append(image_data, images, axis=2)
		# load labels
		label = np.array(mat['label'])
		label_array = np.append(label_array, label, axis=0)
	# convert labels to skalar values
	for label in label_array:
		val = [i for i, x in enumerate(dict_list) if x==label]
		if val != []:
			#print(val[0])
			label_data = np.append(label_data,val)
	# create correct number of labels
	for pos in range(len(num_im_list)):
		lab_array = np.ones(num_im_list[pos]) * label_data[pos]
		label_data_extend = np.append(label_data_extend, lab_array)
	# images: swap axis, convert to 4D tensor
	image_data = np.swapaxes(image_data,0,2)
	image_data = image_data[:,:,:,np.newaxis]
	#number of classes
	classes = len(label_dict)
	# one hot encoding and classes
	label_onehot = one_hot_encoding(label_data_extend,label_dict)		
	return image_data, label_onehot, label_dict, classes	
	
def one_hot_encoding(label_data,label_dict):
	'''convert labels to one hot encoding'''
	classes = len(label_dict)
	num_lab = len(label_data)
	lab_one = np.zeros((num_lab, classes))
	for one in range(num_lab):
		lab_one[one][int(label_data[one])] = 1
	return lab_one

def create_random_vec(randomint, datalength):
	'''create random vector'''
	np.random.seed(randomint)
	randomvec = np.arange(datalength)
	np.random.shuffle(randomvec)
	return randomvec

def random_shuffle_list(randomint, dataset_name):
	'''shuffle list for data randomly'''
	datasetlength = len(dataset_name)
	# init random list
	dataset_rand = []
	# create random vector
	randomvec = createRandomvec(randomint, datasetlength)
	# fill random list
	for pos in range(datasetlength):
		dataset_rand.append(dataset_name[randomvec[pos]])
	return dataset_rand

def split_dataset(train_split, label_onehot, randomint):
	''' split data into training and test set '''
	# opt: create random list and split e.g. 90:10 (train/test)
	randomvec = create_random_vec(randomint, label_onehot.shape[0])
	num_train = int(randomvec.shape[0] * train_split)
	num_test = randomvec.shape[0] - num_train
	# generate lists for training/testing
	train_array = randomvec[0:num_train]
	test_array = randomvec[num_train:num_train + num_test]
	return train_array, test_array
	
def batch_generator(batchsize, rand_mb, image_data, label_onehot):
	''' create mini-batches from data and labels'''
	
	# init mbs with first array
	image_mbx = image_data[rand_mb[0],:,:,:]
	image_mb = image_mbx[np.newaxis,:,:,:]
	lab_mbx = label_onehot[rand_mb[0]]
	lab_mb = lab_mbx[np.newaxis,:]
	#print(image_mb.shape)
	for pos2 in range(1,len(rand_mb)):
		# images
		image = image_data[rand_mb[pos2],:,:,:]
		image_mbx2 = image[np.newaxis,:,:,:]
		image_mb = np.append(image_mb, image_mbx2, axis=0)
		# labels
		label = label_onehot[rand_mb[pos2]]
		lab_mbx2 = label[np.newaxis,:]
		lab_mb = np.append(lab_mb, lab_mbx2, axis=0)
	return image_mb , lab_mb

def train_loop(epochs, train_list,test_list, batchsize, 
               image_data, label_onehot, unit_test):
	''' train the neural model '''
	print('=============================================')
	print('start '+str(modtype)+' training')
	print('=============================================')
	t1_1 = datetime.datetime.now()	
	# init cost, accuracy:
	cost_history = np.empty(shape=[0],dtype=float)
	train_acc_history = np.empty(shape=[0],dtype=float)
	crossval_history = np.empty(shape=[0],dtype=float)
	# calculate num of batches
	num_samples = len(train_list)
	batches = int(num_samples/batchsize)
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 5
		batchsize = 20
	# epoch loop:
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		np.random.shuffle(train_list) # for testing
		np.random.shuffle(test_list) # for validation
		for pos in range(batches):
			# iterate over rand list for mb
			rand_mb = train_list[pos * batchsize: (pos * batchsize) + batchsize]
			#print(rand_mb)
			# generate batches
			train_im_mb , train_lab_mb = batch_generator(batchsize_train, 
			                                             rand_mb, image_data, 
			                                             label_onehot)
			#print(image_mb.shape)
			#print(lab_mb.shape)
			#print(lab_mb)
			# start feeding data into model
			feedtrain = {x_input: train_im_mb, y_target: train_lab_mb}
			optimizer.run(feed_dict = feedtrain)
			# record histories
			ce_loss = sess.run(cross_entropy,feed_dict=feedtrain)
			cost_history = np.append(cost_history, ce_loss)
			train_acc = sess.run(accuracy,feed_dict=feedtrain)
			train_acc_history = np.append(train_acc_history,train_acc)	
			#check progress: training accuracy
			if  pos%4 == 0:
				# start feeding data into the model:
				feedval = {x_input: train_im_mb, y_target: train_lab_mb}
				train_accuracy = accuracy.eval(feed_dict=feedval)
				crossvalidationloss = sess.run(cross_entropy,feed_dict=feedval)
				crossval_history = np.append(crossval_history,crossvalidationloss)
				# print out info
				t1_2 = datetime.datetime.now()
				print('epoch: '+ str(epoch)+'/'+str(epochs)+
				' -- training utterance: '+ str(pos * batchsize)+'/'+str(num_samples)+
				" -- loss: " + str(ce_loss)[0:-2])
				print('training accuracy: %.2f'% train_accuracy + 
				" -- training time: " + str(t1_2-t1_1)[0:-7])
	print('=============================================')
	#Total Training Stats:
	total_trainacc = np.mean(train_acc_history, axis=0)
	print("overall training accuracy %.3f"%total_trainacc)       
	t1_3 = datetime.datetime.now()
	train_time = t1_3-t1_1
	print("total training time: " + str(train_time)[0:-7]+'\n')
	# create return list
	train_list_ex = [train_acc_history, cost_history, crossval_history, 
	                 train_time, total_trainacc]
	return train_list_ex
		
def test_loop(epochs, test_list, batchsize,image_data, 
              label_onehot, unit_test):
	''' inference: test the neural model '''
	print('=============================================')
	print('start '+str(modtype)+' testing')
	print('=============================================')
	t2_1 = datetime.datetime.now()
	# init random numbers
	randomint_test = 1
	# init histories
	test_acc_history = np.empty(shape=[0],dtype=float)
	test_filenames_history = np.empty(shape=[0],dtype=str)
	
	# calculate num of batches
	num_samples = len(test_list)
	batches = int(num_samples/batchsize_test)
	
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 4
		batchsize = 10
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		np.random.shuffle(test_list)
		for pos in range(batches):
			# iterate over rand list for mb
			rand_mb = test_list[pos * batchsize: (pos * batchsize) + batchsize]
			# generate batches
			test_im_mb , test_lab_mb = batch_generator(batchsize, rand_mb, 
			                                           image_data, label_onehot)
			# start feeding data into model
			feedtest = {x_input: test_im_mb, y_target: test_lab_mb}
			predictions = accuracy.eval(feed_dict = feedtest)
			test_acc_history = np.append(test_acc_history,predictions)
			#check progress: test accuracy
			if  pos%40 == 0:
				test_accuracy = accuracy.eval(feed_dict=feedtest)			
				t2_2 = datetime.datetime.now()
				print('test utterance: '+ str(pos*batchsize)+'/'+str(num_samples))
				print('test accuracy: %.2f'% test_accuracy + 
				" -- test time: " + str(t2_2-t2_1)[0:-7])
	print('=============================================')
	# total test stats:
	print('test utterance: '+ str(num_samples)+'/'+str(num_samples))
	total_testacc = np.mean(test_acc_history, axis=0)
	print("overall test accuracy %.3f"%total_testacc)       
	t2_3 = datetime.datetime.now()
	test_time = t2_3-t2_1
	print("total test time: " + str(test_time)[0:-7]+'\n')		
	
	# create return list
	test_list_ex  = [test_acc_history, total_testacc, test_time]
	return test_list_ex

########################################################################
# import data and pre-process data
########################################################################

# get .mat-file names
filelist = get_filelist(filepath)
# load data from .mat files
image_data, label_onehot, label_dict, classes = load_mat_data(filelist)


########################################################################
# init and define model:
########################################################################

# init model:
def weight_variable(shape):
	'''init weights'''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name="W")
	
def bias_variable(shape):
	'''init biases'''
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name="b")

def matmul(x, W, b, name="fc"):
	'''matrix vector multiplication'''
	with tf.name_scope(name):
	  return tf.add(tf.matmul(x, W), b)
		
def dense(x, W, b, name="dense"):
	'''matrix vector multiplication'''
	with tf.name_scope(name):
		return tf.add(tf.matmul(x, W), b)

def selu(x):
	'''Selu activation function'''
	with ops.name_scope('elu') as scope:
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
		 
def activfunction(x, acttype):
	'''activation functions'''
	if acttype == "tanh":
		activation_array = tf.tanh(x)
	elif acttype == "relu":
		activation_array = tf.nn.relu(x)
	elif acttype == "selu":
		activation_array = selu(x)	
	else:
		activation_array = tf.sigmoid(x)
	return activation_array

def conv2d(x, W):
	'''convolution operation (cross correlation)'''
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	'''max pooling'''
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
# set conv hyper params:
f1 = 3
f1_d = 32
f2_d = 32 * 2
f3_fc = 512 * 2
                       
# init weights:                   
W1 = weight_variable([f1, f1, 1, f1_d])
b1 = bias_variable([f1_d])
W2 = weight_variable([f1, f1, f1_d, f2_d])
b2 = bias_variable([f2_d])
W3 = weight_variable([1216 * 1 * f1_d, f3_fc])
b3 = bias_variable([f3_fc])
W4= weight_variable([f3_fc, classes])
b4 = bias_variable([classes])

# init placeholder:
x_input = tf.placeholder(tf.float32, shape=[None,61,151, 1],name="input")
y_target = tf.placeholder(tf.float32, shape=[None, classes],name="labels")

####### init model ########
# 1.conv layer    
h_conv1 = tf.nn.relu(conv2d(x_input, W1) + b1)
h_pool1 = max_pool_2x2(h_conv1)
# 2.conv layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool_2x2(h_conv2)
# 1. FC Layer
h_flat = tf.reshape(h_pool2, [-1, 1216 * 1 * f1_d])
#print(h_flat)
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W3) + b3)
# readout Layer
h_out = tf.matmul(h_fc1, W4) + b4
# define classifier, cost function:
softmax = tf.nn.softmax(h_out)

# define loss, optimizer, accuracy:
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(logits= h_out,labels = y_target))
optimizer = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_out,1), tf.argmax(y_target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init tf session :
sess = tf.InteractiveSession()
# save and restore all the variables:
saver = tf.train.Saver()
# start session:
sess.run(tf.global_variables_initializer()) 

# print out model info:
print("**********************************")
print("model: "+str(modlayer)+ " layer "+str(modtype))
print("**********************************")
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))
print()

########################################################################
#start main: 
########################################################################
# split dataset (e.g. 90:10 train/test)
train_array, test_array = split_dataset(train_split,label_onehot, 
                                        randomint)
# start training
train_list = train_loop(epochs, train_array,test_array, batchsize_train, 
                        image_data, label_onehot,unit_test)
# start testing
test_list = test_loop(1, test_array, batchsize_test,image_data, 
                      label_onehot, unit_test)

########################################################################
#plot settings:
########################################################################

# plot model summary:
plot_model_sum = "model_summary: "+str(modtype)+"_"+str(modlayer)+" hid_layer, "+\
                          ", classes: " +str(classes)+", epochs: "+str(epochs)+"\n"+\
                          norm+ ", "+"training accuracy %.3f"%train_list[4]+\
                          " , test accuracy %.3f"%test_list[1]+" , "+\
                          str(timemarker)
                          
# define storage path for model:
model_path = path + "04_nn_training/pretrained_models/"
model_name = str(timemarker)+"_"+modtype + "_"+ str(modlayer)+ "lay_"+\
             str(classes)+ "class_" +str(epochs)+"eps_"+\
             acttype+"_"+str(test_list[1])[0:-9]+"testacc"
              
# init moving average:
wintrain = 300
wintest = 300
wtrain_len = len(train_list[1])
if wtrain_len < 100000:
	wintrain = 5
wtest_len = len(test_list[0])
if wtest_len < 10000:
	wintest = 5
	
#=======================================================================
#plot training
#=======================================================================

#plot training loss function
fig1 = plt.figure(1, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(train_list[1])),train_list[1], color='#1E90FF')
#plt.plot(np.convolve(train_list[1], np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
plt.axis([0,len(train_list[1]),0,int(10)+1])
#plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batchsize_train))
plt.ylabel('loss')

#plot training accuracy function
plt.subplot(212)
plt.plot(range(len(train_list[0])),train_list[0], color='#20B2AA')
#plt.plot(np.convolve(train_list[0], np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
plt.axis([0,len(train_list[1]),0,1])
plt.title('Training Accuracy')
plt.xlabel('number of batches, batch size:'+ str(batchsize_train))
plt.ylabel('accuracy percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
im1 = pltim.imread('imagetemp/'+model_name+"_fig1"+'.jpeg')
pltim.imsave("imagetemp/out.png", im1)

#=======================================================================
#plot testing
#=======================================================================

#plot validation loss function
plt.figure(2, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(train_list[2])),train_list[2], color='#1E90FF')
#plt.plot(np.convolve(train_list[2], np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
plt.axis([0,len(train_list[2]),0,int(10)+1])
#plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
plt.title('Cross Validation Loss')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

#plot test accuracy function
plt.subplot(212)
plt.plot(range(len(test_list[0])),test_list[0], color='m')
#plt.plot(np.convolve(test_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
plt.axis([0,len(test_list[0]),0,1])
plt.title('Test Accuracy')
plt.xlabel('number of batches, batch size: '+ str(batchsize_test))
plt.ylabel('accuracy percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig2"+'.jpeg', bbox_inches='tight')
im2 = pltim.imread('imagetemp/'+model_name+"_fig2"+'.jpeg')


########################################################################
#start export:
########################################################################

#.mat file method:
model = {"model_name" : model_name} # create dict model
# modelweights
model["W1"] = np.array(sess.run(W1))
model["W2"] = np.array(sess.run(W2))
model["W3"] = np.array(sess.run(W3))
model["W4"] = np.array(sess.run(W4))
# biases
model["b1"] = np.array(sess.run(b1))
model["b2"] = np.array(sess.run(b2))
model["b3"] = np.array(sess.run(b3))
model["b4"] = np.array(sess.run(b4))
# activation type
model["acttype"] = acttype
#train_list = [train_acc_history, cost_history, crossval_history, train_time, total_trainacc]
#test_list  = [test_acc_history, total_testacc, test_time]
# modellayout
model["layer"] = modlayer
model["classes"] = classes
model["randomint"] = randomint
model["classification"] = classifier 
model["optimizer"] = optimizing
model["norm"] = norm
# trainingparams
model["epochs"] = epochs
model["learnrate"] = learnrate
model["batsize_train"] = batchsize_train
model["total_trainacc"] = train_list[4]
# testparams
model["batsize_test"] = batchsize_test
model["total_testacc"] = test_list[1]
# history = [cost_history, train_acc_history, test_acc_history]
model["cost_history"] = train_list[1]
model["train_acc_history"] = train_list[0]
model["test_acc_history"] = test_list[0]

# save plot figures
model["fig_trainplot"] = im1
model["fig_testplot"] = im2

# save radar specific information
model["label_dict"] = label_dict
#model["radar_arena"] = 

# export to .csv file
model_csv = [timemarker,train_list[4],test_list[1],
             modtype,modlayer,classes,
             classifier,epochs,acttype,learnrate,optimizing,
             norm,batchsize_train,batchsize_test]
df = pd.DataFrame(columns=model_csv)
# save only good models
if train_list[4] > save_thres:
	print('=============================================')
	print("start export")
	print('=============================================')
	scipy.io.savemat(model_path + model_name,model)
	df.to_csv('model_statistics/nn_model_statistics.csv', mode='a')
	print("Model saved to: ")
	print(model_path + model_name)
	print('=============================================')
	print("export finished")
	print('=============================================')
	print(" "+"\n")
	
# print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: "+ str(modtype)+"_"+ str(modlayer)+" layer")
print("*******************************************")
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))
print("-------------------------------------------")
print(str(modtype)+' training:')
print("total training time: " + str(train_list[3])[0:-7])
print("overall training accuracy %.3f"%train_list[4]) 
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(test_list[2])[0:-7])	
print("overall test accuracy %.3f"%test_list[1]) 
print("*******************************************")

#plot show options:-----------------------------------------------------
#plt.show()
#plt.close()








