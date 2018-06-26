import tensorflow as tf

from layer_utils import ConvBnAct, DeconvBnAct, ResidualBlock
import pdb


def normalize_image(x):
    return x / 255.0


def denormalize_image(x):
    return (x + 1) * 127.5


def create_resnet(input_shape,style_shape, name='resnet'):
    with tf.variable_scope(name):
        with tf.variable_scope('Style'):
            x_style = tf.keras.layers.Input(style_shape, name='style_input')
            y = tf.keras.layers.Lambda(normalize_image, name='style_normalize')(x_style)
            y= ConvBnAct(32, (9, 9), name='conv1_s')(y)
            y = ConvBnAct(64, (3, 3), stride=2, name='conv2_s')(y)
            y = ConvBnAct(128, (3, 3), stride=2, name='conv3_s')(y)



        x_input = tf.keras.layers.Input(input_shape, name='input')
        x = tf.keras.layers.Lambda(normalize_image, name='normalize')(x_input)
        x = ConvBnAct(32, (9, 9), name='conv1')(x)
        x = ConvBnAct(64, (3, 3), stride=2, name='conv2')(x)
        x = ConvBnAct(128, (3, 3), stride=2, name='conv3')(x)
        

        with tf.variable_scope('Concat'):
            merged=tf.keras.layers.concatenate(inputs=[x,y],axis=-1)
            shaped= ConvBnAct(128, (1, 1), stride=1, name='conv1_Merged')(merged)
    



        z = ResidualBlock(128, (3, 3), stride=1, name='resblock1', first=True)(shaped)
        z = ResidualBlock(128, (3, 3), stride=1, name='resblock2')(z)
        z = ResidualBlock(128, (3, 3), stride=1, name='resblock3')(z)
        z = ResidualBlock(128, (3, 3), stride=1, name='resblock4')(z)
        z = ResidualBlock(128, (3, 3), stride=1, name='resblock5')(z)

        z = DeconvBnAct(64, (3, 3), stride=2, name='deconv1')(z)
        z = DeconvBnAct(32, (3, 3), stride=2, name='deconv2')(z)
        z = DeconvBnAct(3, (9, 9), activation='tanh', name='deconv3')(z)
        z = tf.keras.layers.Lambda(denormalize_image, name='denormalize')(z)
  
        

        model = tf.keras.models.Model(inputs=[x_input,x_style], outputs=z, name=name)
        return model
