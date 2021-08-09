'''
Prepara o modelo de transfer learning

https://keras.io/guides/transfer_learning/
https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
https://stackoverflow.com/questions/53907681/how-to-fine-tune-a-functional-model-in-keras
'''

### Imports
import os
from matplotlib import pyplot as plt

### Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, load_model

### Módulos próprios
import utils

# norm_layer = tf.keras.layers.BatchNormalization
norm_layer = tfa.layers.InstanceNormalization

# Modo de inicialização dos pesos
initializer = tf.random_normal_initializer(0., 0.02)


#%% BLOCOS

def upsample_block(x, filters, name_prefix = None, name_suffix = None):
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN

    if name_prefix == None or name_suffix == None:
        x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same", 
                                            kernel_initializer=initializer, use_bias = True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    else:
        x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same", 
            kernel_initializer=initializer, use_bias = True, name = name_prefix + 'upsample_conv2dtrans' + name_suffix)(x)
        x = norm_layer(name = name_prefix + 'upsample_norm' + name_suffix)(x)
        x = tf.keras.layers.ReLU(name = name_prefix + 'upsample_relu' + name_suffix)(x)
    
    return x

def simple_upsample(x, scale = 2, interpolation = 'bilinear', name_prefix = None, name_suffix = None):
    # Faz um umpsample simplificado, baseado no Progressive Growth of GANs
    
    if name_prefix == None or name_suffix == None:
        x = tf.keras.layers.UpSampling2D(size = (scale, scale), interpolation = interpolation)(x)

    else:
        x = tf.keras.layers.UpSampling2D(size = (scale, scale), interpolation = interpolation, 
                                        name = name_prefix + 'upsampling2d' + name_suffix)(x)

    return x

def VT_simple_upsample_block(x, filters, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear', name_prefix = None, name_suffix = None):
    
    if name_prefix == None or name_suffix == None:

        x = simple_upsample(x, scale = scale, interpolation = interpolation) 
        
        x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    else:
        x = simple_upsample(x, scale = scale, interpolation = interpolation, name_prefix = name_prefix, name_suffix = name_suffix) 
        
        x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", 
                                            kernel_initializer=initializer, use_bias = True, name = name_prefix + 'conv2dtrans1_' + name_suffix)(x)
        x = norm_layer(name = name_prefix + 'upsample_norm1_' + name_suffix)(x)
        x = tf.keras.layers.ReLU(name = name_prefix + 'upsample_relu1_' + name_suffix)(x)
        
        x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", 
                                            kernel_initializer=initializer, use_bias = True, name = name_prefix + 'conv2dtrans2_' + name_suffix)(x)
        x = norm_layer(name = name_prefix + 'upsample_norm2_' + name_suffix)(x)
        x = tf.keras.layers.ReLU(name = name_prefix + 'upsample_relu2_' + name_suffix)(x)
    
    return x

#%% BASE DO MODELO DE TRANSFER TRAINING

def get_encoder(generator, encoder_last_layer, trainable):
    '''
    Separa o Encoder
    Para o encoder é fácil, ele usa o mesmo input do modelo, e é só "cortar" ele antes do final
    '''
    encoder = Model(inputs = generator.input, outputs = generator.get_layer(encoder_last_layer).output, name = 'encoder')
    encoder.trainable = trainable
    return encoder

def get_decoder(generator, encoder_last_layer, decoder_first_layer, trainable):
    '''
    Separa o Decoder
    Se usar o mesmo método do encoder no decoder ele vai dizer que o input não está certo, 
    porque ele é output da camada anterior. Deve-se usar um keras.layer.Input
    '''
    # Descobre o tamanho do input e cria um layer para isso
    decoder_input_shape = generator.get_layer(encoder_last_layer).output_shape
    inputlayer = tf.keras.layers.Input(shape=decoder_input_shape[1:])
    # print(decoder_input_shape[1:])

    # Descobre o índice de cada layer (colocando numa lista)
    layers = []
    for layer in generator.layers:
        layers.append(layer.name)

    # Descobre o índice que eu quero
    layer_index = layers.index(decoder_first_layer)

    # Separa os layers que serão usados
    layers = layers[layer_index:]

    # Cria o modelo
    x = inputlayer
    for layer in layers:
        x = generator.get_layer(layer)(x)
    decoder = Model(inputs = inputlayer, outputs = x, name = 'decoder')
    decoder.trainable = trainable
    return decoder

def transfer_model(IMG_SIZE, generator_path, generator_filename, middle_model, encoder_last_layer, decoder_first_layer, transfer_trainable,
                    disentangle = False, smooth_vector = False):
    '''
    Carrega o modelo, separa em encoder e decoder, insere um modelo no meio e retorna o modelo final
    '''
    # Carrega o modelo e separa entre encoder e decoder
    generator = load_model(generator_path + generator_filename)
    encoder = get_encoder(generator, encoder_last_layer, transfer_trainable)
    decoder = get_decoder(generator, encoder_last_layer, decoder_first_layer, transfer_trainable)

    # Cria o modelo final
    inputlayer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name = 'transfer_model_input')
    x = encoder(inputlayer)
    if middle_model != None:
        x = include_vector_resnet(IMG_SIZE, middle_model, disentangle, smooth_vector)(x)
    x = decoder(x)
    transfer_model = Model(inputs = inputlayer, outputs = x)

    return transfer_model


#%% MODELOS DE "MEIO"


def include_vector_resnet(IMG_SIZE, upsample, disentangle = False, smooth = False, vecsize = 512):
    # Apenas cria a redução para o vetor latente e depois retorna para a dimensão (31, 31, 256)
    # disentangle = True -> Inclui camadas de disentanglement (de z para w, StyleGAN)
    # smooth = True -> Faz a redução de dimensão das camadas 
    # upsample = 'conv' -> Usa convoluções para fazer o upsample pelo bloco "net.upsample"
    #          = 'simple' -> Usa o bloco "net.simple_upsample_block"

    if not (upsample == 'conv' or upsample == 'simple'):
        raise utils.TransferUpsampleError(upsample)

    inputlayer = tf.keras.layers.Input(shape=(31, 31, 256))
    x = inputlayer

    # Criação do vetor latente 
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True, name = 'middle_conv1')(x)
    x = norm_layer(name = 'middle_norm1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name = 'middle_leakyrelu1')(x)
    
    # Flatten da convolução. 
    # Se IMG_SIZE = 256, a saída terá 3721 elementos
    # Se IMG_SIZE = 128, a saída terá 841 elementos
    x = tf.keras.layers.Flatten(name = 'middle_flatten')(x)

    if smooth:
        # Redução gradual até vecsize
        if IMG_SIZE == 256:
            x = tf.keras.layers.Dense(units = 3072, kernel_initializer = initializer, name = 'middle_dense_s1')(x) #2048 + 512
            x = tf.keras.layers.Dense(units = 2048, kernel_initializer = initializer, name = 'middle_dense_s2')(x)
            x = tf.keras.layers.Dense(units = 1024, kernel_initializer = initializer, name = 'middle_dense_s3')(x)
            x = tf.keras.layers.Dense(units = 512, kernel_initializer = initializer, name = 'middle_dense_s4')(x)

        elif IMG_SIZE == 128:
            x = tf.keras.layers.Dense(units = 768, kernel_initializer = initializer, name = 'middle_dense_s1')(x) #512 + 256
            x = tf.keras.layers.Dense(units = 512, kernel_initializer = initializer, name = 'middle_dense_s2')(x)

    # Vetor latente (z)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer, name = 'middle_dense_z')(x)
    
    if disentangle:
        # Disentanglement (de z para w, StyleGAN)
        for i in range(8):
            layer_name = 'middle_dense_w'+str(i+1)
            x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer, name = layer_name)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1, name = 'middle_expand_dims1')
    x = tf.expand_dims(x, axis = 1, name = 'middle_expand_dims2')
    
    # Upsamples
    if upsample == 'conv':
        x = upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '1')
        x = upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '2')
        x = upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '3')
        x = upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '4')
        x = upsample_block(x, 256, name_prefix = 'middle_', name_suffix = '5')

    if upsample == 'simple':
        x = VT_simple_upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '1')
        x = VT_simple_upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '2')
        x = VT_simple_upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '3')
        x = VT_simple_upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '4')
        x = VT_simple_upsample_block(x, 512, name_prefix = 'middle_', name_suffix = '5')

    # Finaliza para deixar com  a dimensão correta (31, 31, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size = 2, strides = 1, padding = 'valid', name = 'middle_conv2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name = 'middle_leakyrelu2')(x)
    # print(x.shape)

    # Cria o modelo
    model = Model(inputs = inputlayer, outputs = x, name = "bottleneckmodel")

    return model
