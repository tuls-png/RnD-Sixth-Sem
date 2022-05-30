import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keract import get_activations, display_activations
from keras import layers, losses
from sklearn.model_selection import train_test_split
from keras.models import Model
import cv2 as cv
import array as arr

df = pd.read_csv('finalcombined.csv', header=None)
df.head()

# Now we will separate the data and labels so that it will be easy for us
data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values
print("data",data)
print("labels",labels)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
print("trailabeb",train_labels)
min = tf.reduce_min(train_data)
max = tf.reduce_max(train_data)
print("min",min)
print("max",max)
# Now we will use the formula (data - min)/(max - min)
train_data = (train_data - min) / (max - min)
test_data = (test_data - min) / (max - min)

# I have converted the data into float
train_data = tf.cast(train_data, dtype=tf.float32)
test_data = tf.cast(test_data, dtype=tf.float32)
print("train_data",train_data)
print("test_data",test_data)
print("?????????????????????????????????????????????????")
print(train_labels)
print(test_labels)
# The labels are either 0 or 1, so I will convert them into boolean(true or false)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
print("train_labels",train_labels)
print("test_labels",test_labels)
# Now let's separate the data for normal ECG from that of abnormal ones
# Normal ECG data
print("#################################")
print(train_labels)
print(~train_labels)
print("#################################")
n_train_data = train_data[train_labels]
n_test_data = test_data[test_labels]
print("n_train_data",n_train_data)
print("n_test_data",n_test_data)
print("Normal Train Data\n",n_train_data)
print("Normal Test Data\n",n_test_data)
# Abnormal ECG data
an_train_data = train_data[~train_labels]
an_test_data = test_data[~test_labels]
print("an_train_data",an_train_data)
print("an_test_data",an_test_data)
print("Abnormal Train Data\n",an_train_data)
print("Abnormal Test Data\n",an_test_data)

print(an_test_data[0][-1])
# Lets plot a normal ECG
plt.plot(np.arange(187), n_train_data[0])
plt.grid()
plt.title('Normal ECG')
plt.show()

# Lets plot one from abnormal ECG
plt.plot(np.arange(187), an_train_data[0])
plt.grid()
plt.title('Abnormal ECG')
plt.show()




class detector(Model):
    def __init__(self):
        super(detector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(187, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Let's compile and train the model!!
autoencoder = detector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(n_train_data, n_train_data, epochs=100, batch_size=512, validation_data=(test_data, test_data))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.show()
# Now let's define a function in order to plot the original ECG and reconstructed ones and also show the error
def plot(data, n):
    enc_img = autoencoder.encoder(data)
    dec_img = autoencoder.decoder(enc_img)
    plt.plot(data[n], 'b')
    plt.plot(dec_img[n], 'r')
    plt.fill_between(np.arange(187), data[n], dec_img[n], color='lightcoral')
    plt.legend(labels=['Input', 'Reconstruction', 'Error'])
    plt.show()


plot(n_test_data, 0)
print("n_test_data, 0")


reconstructed = autoencoder.predict(n_train_data)
train_loss = tf.keras.losses.mae(reconstructed, n_train_data)

t = np.mean(train_loss) + np.std(train_loss)
print("Threshold", t)


reconstructed = autoencoder.predict(an_test_data)
test_loss = tf.keras.losses.mae(reconstructed, an_test_data)

plt.hist(train_loss[None,:], bins=50, label='Train', alpha=0.7, edgecolor='red')
plt.hist(test_loss[None,:], bins=50, label='Test', alpha=0.7, edgecolor='yellow')
plt.xlabel("Loss")
plt.ylabel("Number of examples")
plt.legend()
plt.show()


print(train_loss)
print(test_loss)


plot(an_test_data, 0)
print("an_test_data, 0")

plot(an_test_data, 1)
print("an_test_data, 1")
plot(an_test_data, 2)





#Now we display our activations
number = int(input("Enter the amount of data you would like to see:"))
for i in range(number):
    data = test_data[i:i + 1]
    activations = get_activations(autoencoder.encoder, data, nodes_to_evaluate=None, layer_names="dense_2",
                                  output_format='simple', nested=False, auto_compile=True)

    with open("allactivationrecord.txt", 'a') as f:
        for key, value in activations.items():
            f.write(" ------------------------\n \n")
            f.write(f'Activation Value of  Image{i + 1}:\n')
            f.write('%s:%s\n' % (key, value))

    display_activations(activations, cmap=None, save=False, fig_size=(30, 30))
    display_activations(activations, cmap="Dark2", save=True, directory=f'Images/Image{i + 1}')
f.close()
fig = plt.figure(figsize=(10, 100))

# setting values to rows and column variables
rows = 3
columns = number // 2

# reading images
for i in range(number):
    Img1 = cv.imread(f'Images/Image{i + 1}/0_dense_2.png')
    fig.add_subplot(rows, columns, i + 1)
    plt.title(f"Image{i + 1}")
    plt.imshow(Img1)
    plt.axis('off')
plt.show()


antest = []
ntest =[]
datanumber = int(input("Enter sample size:"))

for i in range(datanumber):
    an = an_test_data[i:i + 1]
    n = n_test_data[i:i + 1]
    activations = get_activations(autoencoder.encoder, an, layer_names='dense_2', nodes_to_evaluate=None,
                                  output_format='simple', nested=False, auto_compile=True)

    print('--------------')
    print("Activation 1")
    print(activations)
    print('--------------')

    activations2 = get_activations(autoencoder.encoder, n, layer_names='dense_2', nodes_to_evaluate=None,
                                  output_format='simple', nested=False, auto_compile=True)


    print("Activation 2")
    print(activations2)
    print('--------------')


    with open("activation_record.txt", 'a') as f:
        for key, value in activations.items():
            f.write(" ------------------------\n \n")
            f.write(f'Activation Value of  Image{i+1}:\n')
            f.write('%s:%s\n' % (key, value))
    f.close()
    antest.append(activations['dense_2'][0])
    ntest.append(activations2['dense_2'][0])

    #display_activations(activations, cmap="Dark2", save=False, fig_size=(50, 50))
    #display_activations(activations, cmap="Dark2", save=True, directory=f'Images/Image{i + 1}')
antest1 = []
ntest1 =[]
node=int(input('Enter node number:'))
for i in range(len(antest)):
    antest1.append(antest[i][node-1])
    ntest1.append(ntest[i][node-1])
print(f"The activation values of {node} node for {datanumber} abnormal samples are: ")
print(antest1)
print(" ")
print(f"The activation values of {node} node for {datanumber} normal samples are: ")
print(ntest1)

barWidth = 0.15
fig = plt.subplots(figsize=(12, 8))
br1 = np.arange(1, len(antest1)+1)
br2 = [x + barWidth for x in br1]
plt.bar(br1, antest1, color='r', width=barWidth,
        edgecolor='grey', label='Abnormal Test Data')
plt.bar(br2, ntest1, color ='g', width = barWidth,
        edgecolor ='grey', label ='Normal Test Data')

plt.xlabel('Sample Number', fontsize=15)
plt.ylabel('Activation Value', fontsize=15)
plt.xticks(np.arange(1, len(antest1)+1))

plt.legend()
plt.show()
