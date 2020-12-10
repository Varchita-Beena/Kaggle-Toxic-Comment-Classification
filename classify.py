import pandas as pd
import numpy as np
import re
import nltk
import random
import operator
import math
import mpmath
from keras import backend as K
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.preprocessing import sequence
from skfuzzy.cluster import cmeans, cmeans_predict
import matplotlib.pyplot as plt


train_file = pd.read_csv("train_1.csv")

train_data = train_file.iloc[:,1]
train_labels = train_file.iloc[:,-6:]

X_train, X_remain, y_train, y_remain = train_test_split(train_data, train_labels, test_size=0.4, random_state = 42)                                                   
X_validate, X_test, y_validate, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state = 42)     
length_list = 0
def preprocess(data):
	list = []
	lent = []
	for each in data:
		temp = re.sub('[^A-Za-z0-9]+', ' ', each)
		lent.append(len(temp))
		temp = temp.lower()
		tokens = nltk.word_tokenize(temp)
		list.append(tokens)
		lent.append(len(tokens))
	print ("preprocess")
	maxlength = (max(lent))
	return list, maxlength
preprocessed_train, maxlength_train = preprocess(X_train)
preprocessed_validate, maxlength_val = preprocess(X_validate)
preprocessed_test, maxlength_test = preprocess(X_test)

def generatinglabels(data):
	list_1 = []
	temp = data.values
	for each in temp:
		value = each.tolist()
		values = "".join(str(x) for x in value)
		values = "%06d" % (int(values))
		list_1.append(values)
	length_list = len(list(set(list_1)))
	labels = list(set(list_1))
	new_label = []
	for each in list_1:
		#if each in labels:
		new_label.append(labels.index(each))
	print ("generatinglabels")
	return new_label, length_list
labels_train, train_length_labels = generatinglabels(y_train)
labels_validate, val_length_labels = generatinglabels(y_validate)
labels_test, test_length_labels = generatinglabels(y_test)

def labelsreshape(data):
	shapped = data.reshape((data.shape[0], 1))
	return shapped
labels_train = np.array(labels_train)
labels_validate = np.array(labels_validate)
labels_test = np.array(labels_test)

labels_train_shaped = labelsreshape(labels_train)
labels_validate_shaped = labelsreshape(labels_validate)
labels_test_shaped = labelsreshape(labels_test)


def generatingwordvectors(data):
	vectors = []
	print("generatingwordvectors")
	model = Word2Vec(data, min_count=1, size=300)
	for each in data:
		for every in each:
			vectors.append(model.wv[every])
	#vectors = model.wv.syn0	
	return vectors
vectors_train = generatingwordvectors(preprocessed_train)
vectors_validate = generatingwordvectors(preprocessed_validate)
vectors_test = generatingwordvectors(preprocessed_test)

max_sent_len = 5000
def partioningvectors(data, vectors):
	length_1 = []
	for each in data:
		length_1.append(len(each))
	list = []
	j=0
	for each in length_1:
		temp = []
		temp.extend(vectors[j:j+ each])
		
		list.append(temp)
		j = j + each	
	print ("partioningvectors")
	list = sequence.pad_sequences(list, maxlen=2000,dtype='float')
	
	return list
preprocessed_train_vectors = partioningvectors(preprocessed_train,vectors_train)
preprocessed_validate_vectors = partioningvectors(preprocessed_validate,vectors_validate)
preprocessed_test_vectors = partioningvectors(preprocessed_test, vectors_test)

def shapingvectors(data_1):
	templist = []
	data=np.array(data_1)
	shapped = data.reshape((data.shape[0], data.shape[1], data.shape[2]))
	return shapped

	
shaped_train_vectors = shapingvectors(preprocessed_train_vectors)
shaped_validate_vectors = shapingvectors(preprocessed_validate_vectors)
shaped_test_vectors = shapingvectors(preprocessed_test_vectors)

def shappingzip(data_1):
	templist = []
	data=np.array(data_1)
	shapped = data.reshape(data.shape[0], data.shape[1])
	
	return shapped
'''
def generatingsentvectavg(data):
	
	sent_vect = []
	f_vect = []
	
	print ("generatingsentvect")
	for each in data:
		temp = list(map(sum, zip(*each)))
		
		f_vect = []
		for single in temp:
			f_vect.append(single/ 2000)
		sent_vect.append(f_vect)
	print ("generatingsentvect")
	
	return sent_vect

train_sent_vect = generatingsentvectavg(preprocessed_train_vectors)

cntr, u_orig, _, _, _, _, _ = cmeans(shappingzip(train_sent_vect),c=6, m =2, error=0.005, maxiter=10)
u, u0, d, jm, p, fpc = cmeans_predict(shappingzip(train_sent_vect), cntr ,m =2, error=0.005, maxiter=10)
print (u)


'''


model = Sequential()
model.add(SimpleRNN(300, input_shape = (2000,300)))

model.add(Dense(train_length_labels, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(shaped_train_vectors, labels_train_shaped)

#print (shaped_validate_vectors.shape)

y = model.predict(shaped_validate_vectors)
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
layer_output = get_3rd_layer_output([shaped_validate_vectors])[0]



#for each in layer_output:
#cntr, u_orig, _, _, _, _, _ = cmeans(a,c=6, m =2, error=0.005, maxiter=10)
#u, u0, d, jm, p, fpc = cmeans_predict(a, cntr ,m =2, error=0.005, maxiter=10)
#print (u)

#cntr, u_orig, _, _, _, _, _ = cmeans(layer_output,c=6, m =2, error=0.005, maxiter=10)
#u, u0, d, jm, p, fpc = cmeans_predict(layer_output, cntr ,m =2, error=0.005, maxiter=10)


df = pd.DataFrame(layer_output)
# Number of Clusters
k = 6

# Maximum number of iterations
MAX_ITER = 100

# Number of data points
n = len(df)

# Fuzzy parameter
m = 2.00

def initializeMembershipMatrix():
	membership_mat = []
	for i in range(n):
		random_num_list = [random.random() for i in range(k)]
		summation = sum(random_num_list)
		temp_list = [x/summation for x in random_num_list]
		membership_mat.append(temp_list)
	#print (membership_mat)
	return membership_mat
	
def calculateClusterCenter(membership_mat):
	cluster_mem_val = list(zip(*membership_mat))
	cluster_centers = []
	for j in range(k):
		x = list(cluster_mem_val[j])
		xraised = [e ** m for e in x]
		denominator = sum(xraised)
		temp_num = []
		for i in range(n):
			data_point = list(df.iloc[i])
			prod = [xraised[i] * val for val in data_point]
			temp_num.append(prod)
		numerator = list(map(sum, zip(*temp_num)))
		center = [z/denominator for z in numerator]
		cluster_centers.append(center)
	return cluster_centers
	
def updateMembershipValue(membership_mat, cluster_centers):
	p = float(2/(m-1))
	for i in range(n):
		x = list(df.iloc[i])
		distances = [np.linalg.norm(np.array((list(map(operator.sub, x, cluster_centers[j]))),dtype=np.float128)) for j in range(k)]
		print (distances)
		for j in range(k):
			#den = sum([math.pow(float(distances[j])/distances[c], p) for c in range(k)])
			#for c in range(k):
			#	print (distances[c])
			den = sum([mpmath.power((distances[j]/distances[c]), p) for c in range(k)])
			a = 1/den
			membership_mat[i][j] = a     
	return membership_mat

def getClusters(membership_mat):
	cluster_labels = []
	for i in range(n):
		max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
		cluster_labels.append(idx)
	return cluster_labels


def fuzzyCMeansClustering():
	# Membership Matrix
	membership_mat = initializeMembershipMatrix()
	curr = 0
	while curr <= MAX_ITER:
		cluster_centers = calculateClusterCenter(membership_mat)
		membership_mat = updateMembershipValue(membership_mat, cluster_centers)
		cluster_labels = getClusters(membership_mat)
		curr += 1
	#print(membership_mat)
	return cluster_labels, cluster_centers


labels_1, centers = fuzzyCMeansClustering()
#print (labels_1)
#print (centers)
