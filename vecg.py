import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keract import get_activations, display_activations
from keras import layers, losses
from sklearn.model_selection import train_test_split
from keras.models import Model
import cv2 as cv

df = pd.read_csv('combined-ptb.csv', header=None)
df.head()

# Now we will separate the data and labels so that it will be easy for us
data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values
print("data",data)
print("labels",labels)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

min = tf.reduce_min(train_data)
max = tf.reduce_max(train_data)

# Now we will use the formula (data - min)/(max - min)
train_data = (train_data - min) / (max - min)
test_data = (test_data - min) / (max - min)

# I have converted the data into float
train_data = tf.cast(train_data, dtype=tf.float32)
test_data = tf.cast(test_data, dtype=tf.float32)
print("train_data",train_data)
print("test_data",test_data)
# The labels are either 0 or 1, so I will convert them into boolean(true or false)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
print("train_labels",train_labels)
print("test_labels",test_labels)
# Now let's separate the data for normal ECG from that of abnormal ones
# Normal ECG data
n_train_data = train_data[train_labels]
n_test_data = test_data[test_labels]
print("n_train_data",n_train_data)
print("n_test_data",n_test_data)
# Abnormal ECG data
an_train_data = train_data[~train_labels]
an_test_data = test_data[~test_labels]
print("an_train_data",an_train_data)
print("an_test_data",an_test_data)
#print(n_train_data)
# Lets plot a normal ECG
plt.plot(np.arange(187), n_train_data[1])
plt.grid()
plt.title('Normal ECG')
plt.show()

# Lets plot one from abnormal ECG
plt.plot(np.arange(187), an_train_data[1])
plt.grid()
plt.title('Abnormal ECG')
plt.show()


# Now let's define the model!
# Here I have used the Model Subclassing API (but we can also use the Sequential API)
# The model has 2 parts : 1. Encoder and 2. Decoder

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
autoencoder.fit(train_data, train_data, epochs=20, batch_size=512, validation_data=(test_data, test_data))


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


reconstructed = autoencoder(train_data)
train_loss = losses.mae(reconstructed, train_data)
t = np.mean(train_loss) + np.std(train_loss)


def prediction(model, data, threshold):
    rec = model(data)
    loss = losses.mae(rec, data)
    return tf.math.less(loss, threshold)


print(t)

pred = prediction(autoencoder, test_data, t)
print(pred)

plot(an_test_data, 0)
print("an_test_data, 0")

'''plot(an_test_data, 1)
print("an_test_data, 1")
plot(an_test_data, 2)
print("an_test_data, 2")
plot(an_test_data, 3)
print("an_test_data, 3")'''



for i in range (12):
    an_train_data = train_data[i:i+1]
    activations = get_activations(autoencoder.encoder, an_train_data, layer_names='dense_2', nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
    
    display_activations(activations, cmap="Dark2", save=False, fig_size=(50, 50))
    display_activations(activations, cmap="Dark2", save=True, directory=f'Images/Image{i+1}')

fig = plt.figure(figsize=(10, 100))

# setting values to rows and column variables
rows =3
columns = 4

# reading images
for i in range(1, 13):
    Img1 = cv.imread(f'Images/Image{i}/0_dense_2.png')
    fig.add_subplot(rows, columns, i)
    plt.title(f"Image{i}")
    plt.imshow(Img1)
    plt.axis('off')



plt.show()