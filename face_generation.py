# Face Generation
import numpy as np
import helper

import os
from glob import glob
from matplotlib import pyplot
from PIL import Image
import matplotlib.pylab as plt

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

def DownloadData(data_dir):
    helper.download_extract('mnist', data_dir)
    helper.download_extract('celeba', data_dir)

def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))


def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

class Dataset56x56(object):
    """
    Dataset
    """
    def __init__(self, dataset_name, data_files):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = 64
        IMAGE_HEIGHT = 64

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size

            yield data_batch / IMAGE_MAX_VALUE - 0.5

def CheckTensorFlowVersion():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # TODO: Implement Function
    inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

    return inputs_real, inputs_z, learning_rate

def conv2d_block(x, filters, kernel_size, strides, padding, is_train, alpha):
    x1 = tf.layers.conv2d(x, filters, kernel_size, strides, padding)
    bn = tf.layers.batch_normalization(x1, training=is_train)
    relu = tf.maximum(alpha * bn, bn)
    #print(relu.get_shape().as_list())

    return relu

def discriminator(images, reuse):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function
    with tf.variable_scope('discriminator', reuse=reuse):
        alpha=0.2
        
        # Input layer is 32x32x3
        x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
        relu = tf.maximum(alpha * x1, x1)
        # 14x14x64
        
        size = relu.get_shape().as_list()[1]
        filters = relu.get_shape().as_list()[3] // 2
  
        while size > 2: 
            relu = conv2d_block(relu, filters=filters, kernel_size=5, strides=2, padding='same', is_train=True, alpha=alpha)
            size = size // 2
            filters = filters * 2

        # Flatten it
        flat = tf.reshape(relu, (-1, 4*4*256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        # 1
        
    return out, logits

def conv2d_transpose_block(x, filters, kernel_size, strides, padding, is_train, alpha):
    x1 = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding)
    bn = tf.layers.batch_normalization(x1, training=is_train)
    relu = tf.maximum(alpha * bn, bn)
    print(relu.get_shape().as_list())

    return relu

def generator(z, out_channel_dim, out_size_dim, is_train):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function
    with tf.variable_scope('generator', reuse=not is_train):
        alpha=0.2
        
        # First fully connected layer
        x1 = tf.layers.dense(z, 4*4*512)

        # Reshape it to start the convolutional stack
        x1 = tf.reshape(x1, (-1, 2, 2, 512))
        bn1 = tf.layers.batch_normalization(x1, training=is_train)
        relu = tf.maximum(alpha * bn1, bn1)
        # 2x2x512 now

        size = relu.get_shape().as_list()[1]
        filters = relu.get_shape().as_list()[3] // 2

        while size < out_size_dim // 2: 
            relu = conv2d_transpose_block(relu, filters=filters, kernel_size=5, strides=2, padding='same', is_train=is_train, alpha=alpha)
            size = size * 2
            filters = filters // 2
                 
        # Output layer
        logits = tf.layers.conv2d_transpose(relu, out_channel_dim, 5, strides=2, padding='same')
        # 28x28x3 now
        print(logits.get_shape().as_list())
               
        out = tf.tanh(logits)
        
    return out


def model_loss(input_real, input_z, out_channel_dim, out_size_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    
    g_model = generator(input_z, out_channel_dim, out_size_dim, is_train=True)
    
    d_model_real, d_logits_real = discriminator(input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # TODO: Implement Function
    
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    
    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def show_generator_output(sess, n_images, input_z, out_channel_dim, out_size_dim, image_mode, file_name):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    n_images = 1
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, out_size_dim, False),
        feed_dict={input_z: example_z})

    #images_grid = helper.images_square_grid(samples, image_mode)

    # Scale to 0-255
    images = samples[0]
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    #pyplot.imshow(images, cmap=cmap)
   # pyplot.savefig(file_name)

    save_image(images, file_name)


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # TODO: Build Model
    
    # Image params
    _, image_width, image_height, image_channels = data_shape  
    print("image_width={}, image_height={}, image_channels={}".format(image_width, image_height, image_channels))
    
    # Get the placeholders
    inputs_real, inputs_z, lr = model_inputs(image_width, image_height, image_channels, z_dim)
    
    # Get the loss
    out_size_dim = image_width
    d_loss, g_loss = model_loss(inputs_real, inputs_z, image_channels, out_size_dim)
    
    # Get the optimization operations
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    steps = 0
    print_every = 100
    show_every = 100
    n_images = 1
    losses = []
    file_name_index = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # TODO: Train Model
                
                steps += 1
                
                # Scale images to range of -1 to 1 to match generator range
                batch_images *= 2.0
                               
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images, inputs_z: batch_z, lr : learning_rate})
                
                _ = sess.run(g_train_opt, feed_dict={inputs_z: batch_z, inputs_real: batch_images, lr : learning_rate})
                
                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({inputs_z: batch_z, inputs_real: batch_images})
                    
                    train_loss_g = g_loss.eval({inputs_z: batch_z})
                   
                    print("Epoch {}/{}: Discriminator Loss: {:.4f}, Generator Loss: {:.4f}".format(epoch_i, epoch_count, train_loss_d, train_loss_g))
                          
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))
                    
                if steps % show_every == 0:
                    file_name = 'temp/progress_' + str(file_name_index)
                    show_generator_output(sess, n_images, inputs_z, image_channels, out_size_dim, data_image_mode, file_name)
                    file_name_index += 1

    
def TrainMinst(output_dir):
    batch_size = 64
    z_dim = 100
    learning_rate = 0.001
    beta1 = 0.5
    epochs = 4
    data_dir = './data'

    mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
            mnist_dataset.shape, mnist_dataset.image_mode)


def save_image(data, filename):
    sizes = np.shape(data)     
    fig = plt.figure(figsize=(1,1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap = None)
    plt.savefig(filename, dpi = sizes[0]) 
    plt.close()

def TrainCelebA():
    batch_size = 64
    z_dim = 100
    learning_rate = 0.001
    beta1 = 0.5
    epochs = 10
    data_dir = './data'

    celeba_dataset = Dataset56x56('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))

    print(celeba_dataset.shape)

    for test_batch in celeba_dataset.get_batches(batch_size):

        print(type(test_batch))
        print(test_batch.shape)

        test_image = test_batch[0]
        test_image = (((test_image - test_image.min()) * 255) / (test_image.max() - test_image.min())).astype(np.uint8)

        save_image(test_image, 'temp/test_image')
        #pyplot.imshow(test_image, cmap=None)
        #pyplot.savefig('temp/test_image')
        break

    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
            celeba_dataset.shape, celeba_dataset.image_mode)

if __name__ == "__main__":
    CheckTensorFlowVersion()

    #TrainMinst()
    TrainCelebA()