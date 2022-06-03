import numpy as np
import matplotlib.pyplot as plt
from CNN import NN

train_image_path = "C:\\Users\\d.king\\Documents\\Data Science From Scratch\\Data\\Train\\train-images.idx3-ubyte"
train_image_path = "/mnt/c/users/d.king/Documents/Data Science From Scratch/Data/Train/train-images.idx3-ubyte"
train_label_path = "C:\\Users\\d.king\\Documents\\Data Science From Scratch\\Data\\Train\\train-labels.idx1-ubyte"
train_label_path = "/mnt/c/users/d.king/Documents/Data Science From Scratch/Data/Train/train-labels.idx1-ubyte"
images = np.zeros([60000, 28, 28])
labels = np.zeros([60000])
target = np.zeros([60000, 10])

with open(train_image_path, 'rb') as raw_images, open(train_label_path, 'rb') as raw_labels:
    image_data = list(raw_images.read())
    label_data = list(raw_labels.read())
    for dp in range(0, 60000):
        l = label_data[dp + 8]
        labels[dp] = l
        target[dp, l] = 1
        for y in range(28):
            for x in range (28):
                images[dp,y,x] = image_data[784*dp + 28*y + x + 16] / 255

def display_image(image):
    plt.imshow(image, cmap="Greys")
    plt.show()

labels = labels.astype(int)
nn = NN(in_width=28, in_height=28, filter_width=3, filter_height=3, no_filters=3, pool_width=2, pool_height=2, output_size=10)
nn.set_filters(np.random.randn(3,3,3) / 9)
nn.xavier_init()



test_image_path = "C:\\Users\\d.king\\Documents\\Data Science From Scratch\\Data\\Test\\t10k-images.idx3-ubyte"
test_image_path = "/mnt/c/users/d.king/Documents/Data Science From Scratch/Data/Test/t10k-images.idx3-ubyte"
test_label_path = "C:\\Users\\d.king\\Documents\\Data Science From Scratch\\Data\\Test\\t10k-labels.idx1-ubyte"
test_label_path = "/mnt/c/users/d.king/Documents/Data Science From Scratch/Data/Test/t10k-labels.idx1-ubyte"

test_images = np.zeros([10000, 28, 28])
test_labels = np.zeros([10000])
test_target = np.zeros([10000, 10])

with open(test_image_path, 'rb') as raw_images, open(test_label_path, 'rb') as raw_labels:
    test_image_data = list(raw_images.read())
    test_label_data = list(raw_labels.read())
    for dp in range(0, 10000):
        l = label_data[dp + 8]
        test_labels[dp] = l
        test_target[dp, l] = 1
        for y in range(28):
            for x in range (28):
                test_images[dp,y,x] = image_data[784*dp + 28*y + x + 16] /255
