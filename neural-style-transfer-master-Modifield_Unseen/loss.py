import tensorflow as tf 
import pdb

def vgg_preprocess(x, mean):
    x = x[..., ::-1]  # RGB -> BGR
    return tf.nn.bias_add(x, -mean)


def feature_reconstruction_loss(content, combination):
    with tf.name_scope('scale_factor'):
        shape = tf.shape(combination)
        factor = tf.cast(tf.reduce_prod(shape), tf.float32)
    return tf.reduce_sum(tf.squared_difference(combination, content)) / factor


def gram_matrix(x,batch_size_v):
    batch_size, _, _, channels = x.get_shape().as_list()
    if batch_size==None:
        batch_size=batch_size_v
    assert(batch_size== batch_size_v, "We have a batch_size minimatch _ problem")
    with tf.name_scope('scale_factor'):
        shape = tf.shape(x)
        factor = tf.cast(tf.reduce_prod(shape) / batch_size, tf.float32)
    features = tf.reshape(x, shape=(batch_size, -1, channels), name='features')
    features_T = tf.transpose(features, perm=[0, 2, 1], name='features_T')
    gram = tf.matmul(features_T, features, name='gram') / factor
    return gram


def style_reconstruction_loss(style_f, combination,b_size):

    
    with tf.name_scope('scale_factor'):
        shape = tf.shape(combination)
        factor = tf.cast(tf.reduce_prod(shape), tf.float32)
    with tf.name_scope('style_gram'):
        style_gram=gram_matrix(style_f,b_size)
    with tf.name_scope('combination_gram'):
        combination_gram = gram_matrix(combination,b_size)
    return tf.reduce_sum(tf.squared_difference(style_gram, combination_gram)) / factor


def total_variation_regularization(x):
    shape = tf.shape(x, name='shape')
    height, width = shape[1], shape[2]
    a = tf.squared_difference(x[:, :height - 1, :width - 1, :], x[:, 1:, :width - 1, :])
    b = tf.squared_difference(x[:, :height - 1, :width - 1, :], x[:, :height - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def get_total_loss(transformation_model,
                   batch_size=1,
                   content_layer='block2_conv2',
                   content_weight=1.0,
                   style_weight=1.0,
                   total_variation_weight=1e-4,
                   name='loss'):
    

    style_layers=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    with tf.variable_scope(name):
        content_reference_image = transformation_model.input[0] #This is the content image our real image
        combination_image = transformation_model.output #This is our outout image from the generator
        style_reference_img=transformation_model.input[1] #This is the image that uses to ttyle


        imagenet_mean = tf.constant([103.939, 116.779, 123.68])
        preprocessed_image = vgg_preprocess(style_reference_img, imagenet_mean)
        style_vgg = tf.keras.applications.VGG16(input_tensor=preprocessed_image, weights='imagenet', include_top=False)
        style_outputs_dict = dict([(layer.name, layer.output) for layer in style_vgg.layers])




        with tf.variable_scope('vgg'):
            vgg_input = tf.concat([content_reference_image, combination_image], axis=0, name='vgg_input') #Add both generated and real image
            with tf.name_scope('normalization'):
                vgg_norm_input = tf.keras.applications.vgg16.preprocess_input(vgg_input)

                
            loss_model = tf.keras.applications.VGG16(input_tensor=vgg_norm_input, weights='imagenet', include_top=False) #Get pre trained VGG layer details for trined one

        outputs_dict = dict([(layer.name, layer.output) for layer in loss_model.layers]) #collect all the layer names in pre trained VGG net




        # Feature reconstruction loss
        with tf.name_scope('content'): 
            layer_features = outputs_dict[content_layer] #Get the relevenet content layer to check  whther the cntent is similar we can change it
            content_features, combination_features = tf.split(layer_features, [batch_size, batch_size], axis=0)
            content_loss = feature_reconstruction_loss(content_features, combination_features)


       
        # Style reconstruction loss
        with tf.name_scope('style'):
            style_losses = []
            for layer_name in style_layers:
                with tf.name_scope(layer_name):
                    layer_features = outputs_dict[layer_name]
                    _, combination_features = tf.split(layer_features, [batch_size, batch_size], axis=0)
                    style_features = style_outputs_dict[layer_name]
                    style_losses.append(style_reconstruction_loss(style_features, combination_features,batch_size))
            with tf.name_scope('sum'):
                style_loss = sum(style_losses)
        

        # Total variation regularization
        with tf.name_scope('total_variation'):
            tv_loss = total_variation_regularization(combination_image) / batch_size

        with tf.name_scope('total_loss'):
            content_weight_tensor = tf.constant(content_weight, name='content_weight', dtype=tf.float32)
            style_weight_tensor = tf.constant(style_weight, name='style_weight', dtype=tf.float32)
            tv_weight_tensor = tf.constant(total_variation_weight, name='total_variation_weight', dtype=tf.float32)

            weighted_content_loss = content_weight_tensor * content_loss
            weighted_style_loss = style_weight_tensor * style_loss
            weighted_tv_loss = tv_weight_tensor * tv_loss

            total_loss = weighted_content_loss + weighted_style_loss + weighted_tv_loss
    with tf.name_scope('summaries'):
        tf.summary.scalar('content_loss', content_loss)
        tf.summary.scalar('weighted_content_loss', weighted_content_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('weighted_style_loss', weighted_style_loss)
        tf.summary.scalar('tv_regularization', tv_loss)
        tf.summary.scalar('weighted_tv_regularization', weighted_tv_loss)
        tf.summary.scalar('total_loss', total_loss)
    return total_loss
