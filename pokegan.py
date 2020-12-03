import tensorflow as tf # for deep learning related tasks
import matplotlib.pyplot as plt # for ploting
import os
import time # for time related ops
from data_handler import get_pokemon_dataset
from functools import partial # for partial functions

Conv2DT = partial(tf.keras.layers.Conv2DTranspose, kernel_size=(5,5), strides=(2,2), padding="same", use_bias=False)
Conv2D = partial(tf.keras.layers.Conv2D,kernel_size=(5,5), strides=(2,2), padding="same")

train_dataset = get_pokemon_dataset("data")

class PokemonGenerative(tf.keras.models.Model):
    """
    A generative model for pokemon generator
    """

    def __init__(self, *args, **kwargs):
        super(PokemonGenerative, self).__init__(*args, **kwargs)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8*8*512, use_bias=False, input_shape=(100,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((8,8,512)),

            Conv2DT(filters=256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            Conv2DT(filters=128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            Conv2DT(filters=64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            Conv2DT(filters=4, activation=tf.nn.tanh)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class PokemonDisctiminator(tf.keras.models.Model):
    """
    A discriminative model for pokemon discriminator
    """

    def __init__(self, *args, **kwargs):
        super(PokemonDisctiminator, self).__init__(*args, **kwargs)

        self.model = tf.keras.models.Sequential([
            Conv2D(filters=64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            Conv2D(filters=128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            Conv2D(filters=256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            Conv2D(filters=512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)

        ])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)

z = tf.random.normal([1,100])
generator = PokemonGenerative()

output = generator(z)
plt.imshow(output[0,:,:,:])
plt.show()

discriminator = PokemonDisctiminator()
print(discriminator(output))

# create a loss for generative
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


# create a loss function for discriminator
def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# optimizers for the generative models
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# creation of checkpoint dirs
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

# all the epochs for the training
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 256

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    """
    TF Graph function to be compiled into a graph
    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    """
    Function to perform training ops on given set of epochs
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1,
                                                   time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def train(dataset, epochs):
    """
    Function to perform training ops on given set of epochs
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1,
                                                   time.time() - start))

def generate_and_save_images(model, epoch, test_input):
    """
    A helper function for generating and saving images during training ops
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.0)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()




# call to train function
train(train_dataset, EPOCHS)

# restoring the checkpoints for the
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

