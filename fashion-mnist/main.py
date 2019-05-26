# imports
import os
import tensorflow as tf
from tensorflow import keras

# Helper Libraries
import numpy as np
import matplotlib.pyplot as plt

# initial check
print(tf.__version__);

# initialize fashion mnist
mnist = keras.datasets.fashion_mnist;
(train_image,train_label),(test_image,test_label)=mnist.load_data();

# listing classnames
names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'];

'''
plt.figure();
plt.imshow(train_image[0],cmap=plt.cm.binary);
plt.colorbar();
plt.grid(False);
plt.show();
input();
'''

print("**************************************Begin show off part\n");
# change image from [0,255] to [0,1] (int to float)
train_image = train_image / 255.0;
test_image = test_image / 255.0;

# present first 25 images
'''
plt.figure(figsize=(10,10));
for i in range(1,26):
	plt.subplot(5,5,i);
	plt.xticks([]);
	plt.yticks([]);
	plt.grid(False);
	plt.imshow(train_image[i-1],cmap=plt.cm.binary);
	plt.colorbar();
	plt.xlabel(names[train_label[i-1]]);
plt.show();
input();
'''

print("**************************************Begin preparation part\n");
# graph construction: flatten the images and set layer1 as [128], layer2 as [10]
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(128,activation = tf.nn.relu),
	keras.layers.Dense(10,activation = tf.nn.softmax)
]);

# choose the optimizer, loss evaluation function and the monitor metrics
model.compile(optimizer = tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics = ['accuracy']);

print("**************************************Begin training part\n");
# send data in, set epochs(time of execution), choose from batch_size(size of single group) or steps_per_epoch(groups in one epoch)
model.fit(train_image,train_label,epochs = 5,steps_per_epoch = 50);

# use testdata to test accuracy
test_loss,test_accu = model.evaluate(test_image,test_label);
print("Test accuracy: ",test_accu);
