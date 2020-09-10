import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications import vgg19


def preprocess_image(image_path, nrows, ncols):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(image_path, target_size=(nrows, ncols))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x, nrows, ncols):
    # Util function to convert a tensor into a valid image
    x = x.reshape((nrows, ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    # The gram matrix of an image tensor (used to compute the style loss)
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination, nrows, ncols):
    # The "style loss" is designed to maintain the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of feature maps from the style reference image
    # and from the generated image and keeps the generated image close to the local textures
    # of the style reference image
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = nrows * ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    # An auxiliary loss function designed to maintain the "content" of the base image in the generated image
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(x, nrows, ncols):
    # The 3rd (regularization) loss function, designed to keep the generated image locally coherent
    a = tf.square(x[:, : nrows - 1, : ncols - 1, :] - x[:, 1:, : ncols - 1, :])
    b = tf.square(x[:, : nrows - 1, : ncols - 1, :] - x[:, : nrows - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def compute_loss(feature_extractor, combination_image, base_image, style_reference_image, nrows, ncols,
                 style_layer_names,
                 content_layer_name, total_variation_weight=1e-6, style_weight=1e-6, content_weight=2.5e-8):
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())  # Initialize the loss
    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features, combination_features)
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features, nrows, ncols)
        loss += (style_weight / len(style_layer_names)) * sl
    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image, nrows, ncols)
    return loss


@tf.function    # Add a tf.function decorator to loss & gradient computation
def compute_loss_and_grads(feature_extractor, combination_image, base_image, style_reference_image, nrows, ncols,
                           style_layer_names, content_layer_name):
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor, combination_image, base_image, style_reference_image, nrows, ncols,
                            style_layer_names, content_layer_name)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


def generate_image(base_img_path, style_reference_img_path, result_prefix, res_width=600, iterations=200, learning_rate=10.0):
    # Visualize both base and style reference images
    base_image = Image.open(base_img_path)
    base_image.show()
    style_reference_image = Image.open(style_reference_img_path)
    style_reference_image.show()
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
    # List of layers to use for the style loss.
    style_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    content_layer_name = "block5_conv2"  # The layer to use for the content loss.
    # Dimensions of the generated picture.
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = res_width
    img_ncols = int(width * img_nrows / height)
    # The training loop
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    base_image = preprocess_image(base_image_path, img_nrows, img_ncols)
    style_reference_image = preprocess_image(style_reference_image_path, img_nrows, img_ncols)
    combination_image = tf.Variable(preprocess_image(base_image_path, img_nrows, img_ncols))
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(feature_extractor, combination_image, base_image, style_reference_image,
                                             img_nrows, img_ncols, style_layer_names, content_layer_name)
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 50 == 0:
            print("Iteration %d:\t loss = %.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy(), img_nrows, img_ncols)
            plt.imshow(img)
            plt.title(result_prefix + " at iteration %d" % i)
            plt.axis("off")
            plt.show()
    img = deprocess_image(combination_image.numpy(), img_nrows, img_ncols)
    fname = "output/" + result_prefix
    keras.preprocessing.image.save_img(fname, img)
    result_image = Image.open(fname)
    return result_image


base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg", cache_dir="input/", cache_subdir="")
style_reference_image_path = keras.utils.get_file("starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg", cache_dir="styles/", cache_subdir="")
result_image_name = "paris_starry_night.png"
result_img = generate_image(base_image_path, style_reference_image_path, result_image_name)
result_img.show()
