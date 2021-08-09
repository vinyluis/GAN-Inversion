import os

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_addons as tfa

## Tipo de normalização
# norm_layer = tf.keras.layers.BatchNormalization
norm_layer = tfa.layers.InstanceNormalization

## Modo de inicialização dos pesos
initializer = tf.random_normal_initializer(0., 0.02)


#%% BLOCOS 

def resnet_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


def resnet_block_transpose(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34, mas invertido (convoluções transpostas)
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def resnet_bottleneck_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet bottleneck, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


def resnet_downsample_bottleneck_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet bottleneck, com redução de dimensão, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Convolução da skip connection
    skip = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(skip)
    skip = norm_layer()(skip)
    
    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


def upsample(x, filters):
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x


def downsample(x, filters):
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN    
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


def simple_upsample(x, scale = 2, interpolation = 'bilinear'):
    # Faz um umpsample simplificado, baseado no Progressive Growth of GANs
    x = tf.keras.layers.UpSampling2D(size = (scale, scale), interpolation = interpolation)(x)
    return x


def VT_simple_upsample_block(x, filters, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear'):
    
    x = simple_upsample(x, scale = scale, interpolation = interpolation) 
    
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x


def simple_downsample(x, scale = 2):
    # Faz um downsample simplificado, baseado no Progressive Growth of GANs
    x = tf.keras.layers.AveragePooling2D(pool_size = (scale, scale))(x)
    return x


#%% GERADORES

def dcgan_generator(IMG_SIZE, VEC_SIZE, disentanglement = True):
        
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [VEC_SIZE])
    x = inputs

    # Disentanglement (de z para w, StyleGAN)
    if disentanglement:
        for i in range(8):
            x = tf.keras.layers.Dense(units = VEC_SIZE, kernel_initializer = initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Reconstrução da imagem
    if IMG_SIZE == 256:
        upsamples = 8
    elif IMG_SIZE == 128:
        upsamples = 7

    for i in range(upsamples):
        div = int(i/2)
        x = tf.keras.layers.Conv2DTranspose(filters = VEC_SIZE/(2**div), kernel_size = (4, 4) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (4, 4) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def pix2pix_adapted_decoder(IMG_SIZE, VEC_SIZE, disentanglement = True):
        
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [VEC_SIZE])
    x = inputs

    # Disentanglement (de z para w, StyleGAN)
    if disentanglement:
        for i in range(8):
            x = tf.keras.layers.Dense(units = VEC_SIZE, kernel_initializer = initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Reconstrução da imagem
    if IMG_SIZE == 256:
        upsamples = 5
    elif IMG_SIZE == 128:
        upsamples = 4

    for i in range(upsamples):
        x = tf.keras.layers.Conv2DTranspose(filters = VEC_SIZE, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)
        
    x = tf.keras.layers.Conv2DTranspose(filters = VEC_SIZE/2, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = VEC_SIZE/4, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = VEC_SIZE/8, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[2, 2],[2, 2]])(x)
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def VT_resnet_decoder(IMG_SIZE, VEC_SIZE, disentanglement = True):
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [VEC_SIZE])
    x = inputs
    
    # Disentanglement (de z para w, StyleGAN)
    if disentanglement:
        for i in range(8):
            x = tf.keras.layers.Dense(units = VEC_SIZE, kernel_initializer = initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamplings
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block_transpose(x, 256)
    
    # Reconstrução pós blocos residuais
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    #--
    if IMG_SIZE == 256:
        x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def VT_simple_decoder(IMG_SIZE, VEC_SIZE, disentanglement = True):

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    Após o vetor latente, usar 8 camadas Dense para "desembaraçar" o espaço latente, como feito na StyleGAN
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [VEC_SIZE])
    x = inputs
    
    # Disentanglement (de z para w, StyleGAN)
    if disentanglement:
        for i in range(8):
            x = tf.keras.layers.Dense(units = VEC_SIZE, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamples
    # Todos os upsamples vão ser feitos com o simple_upsample, seguidos de duas convoluções na mesma dimensão
    if IMG_SIZE == 256:
        x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 2, 2, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 4, 4, 512 ou 2, 2, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 8, 8, 512 ou 4, 4, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 16, 16, 512 ou 8, 8, 512
    x = VT_simple_upsample_block(x, 256, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 32, 32, 256 ou 16, 16, 256
    x = VT_simple_upsample_block(x, 128, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 64, 64, 128 ou 32, 32, 128
    x = VT_simple_upsample_block(x, 64, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 128, 128, 64 ou 64, 64, 64
    x = VT_simple_upsample_block(x, 32, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 256, 256, 32 ou 128, 128, 32

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


#%% DISCRIMINADORES

def dcgan_discriminator(IMG_SIZE, constrained = False, use_logits = True):
        
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name='input_image')
    x = inputs
        
    # Reconstrução da imagem
    if IMG_SIZE == 256:
        downsamples = 8
    elif IMG_SIZE == 128:
        downsamples = 7

    for i in range(downsamples):
        div = int((downsamples-i)/2)
        x = tf.keras.layers.Conv2D(filters = 512/(2**div), kernel_size = (4, 4) , strides = (2, 2), padding = "same", kernel_initializer=initializer)(x)    
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Camadas finais
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (4, 4) , strides = (1, 1), padding = "same", kernel_initializer=initializer)(x)
    if not use_logits:
        x = tf.keras.layers.Activation('sigmoid')(x)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def patchgan_discriminator(IMG_SIZE, constrained = False, use_logits = True):
    
    '''
    Versão original do discriminador utilizado nos papers Pix2Pix e CycleGAN,
    mas adaptado para treinamento não condicional e não-supervisionado
    '''
    
    # Inicializa a rede e os inputs
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name='input_image')
    x = inputs
    
    # Convoluções
    x = tf.keras.layers.Conv2D(64, 4, strides=2, kernel_initializer=initializer, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    if IMG_SIZE == 256:
        x = tf.keras.layers.Conv2D(128, 4, strides=2, kernel_initializer=initializer, padding = 'valid')(x)
        x = tf.keras.layers.LeakyReLU()(x)
    elif IMG_SIZE == 128:
        x = tf.keras.layers.Conv2D(128, 2, strides=1, kernel_initializer=initializer, padding = 'valid')(x)
        x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, kernel_initializer=initializer, padding = 'valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Camada final (30 x 30 x 1) - Para usar o L1 loss
    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, padding = 'same')(x)
    if not use_logits:
        x = tf.keras.layers.Activation('sigmoid')(x)
    # print(x.shape)

    return tf.keras.Model(inputs = inputs, outputs=x)


def progan_adapted_discriminator(IMG_SIZE, constrained = False, use_logits = True):

    '''
    Adaptado do discriminador utilizado nos papers ProgGAN e styleGAN
    1ª adaptação é para usar imagens 256x256 (ou IMG_SIZE x IMG_SIZE):
        As primeiras 3 convoluições são mantidas (filters = 16, 16, 32) com as dimensões 256 x 256
        Então "pula" para a sexta convolução, que já é originalmente de tamanho 256 x 256 e continua daí para a frente
    '''

    ## Restrições para o discriminador (usado na WGAN original)
    constraint = tf.keras.constraints.MinMaxNorm(min_value = -0.01, max_value = 0.01)
    if constrained == False:
        constraint = None
    
    # Inicializa a rede e os inputs
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name='input_image')
    x = inputs 

    # Primeiras três convoluções adaptadas para IMG_SIZE x IMG_SIZE
    x = tf.keras.layers.Conv2D(16, (1 , 1), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint = constraint)(x) # (bs, 16, IMG_SIZE, IMG_SIZE)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(16, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint = constraint)(x) # (bs, 16, IMG_SIZE, IMG_SIZE)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint = constraint)(x) # (bs, 32, IMG_SIZE, IMG_SIZE)
    x = tf.keras.layers.LeakyReLU()(x)
    
    if IMG_SIZE == 256:
        # Etapa 256 (convoluções 6 e 7)
        x = tf.keras.layers.Conv2D(64, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 64, 256, 256)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(128, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 128, 256, 256)
        x = tf.keras.layers.LeakyReLU()(x)
        x = simple_downsample(x, scale = 2) # (bs, 128, 128, 128)
    
    # Etapa 128
    x = tf.keras.layers.Conv2D(128, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 128, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 256, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 256, 64, 64)
    
    # Etapa 64
    x = tf.keras.layers.Conv2D(256, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 256, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 32, 32)
    
    # Etapa 32 
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 32, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 32, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 16, 16)
    
    # Etapa 16
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 16, 16)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 16, 16)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 8, 8)
    
    # Etapa 8
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 8, 8)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 8, 8)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 4, 4)
    
    # print(x.shape)

    # Final - 4 para 1
    # Nesse ponto ele faz uma minibatch stddev. Avaliar depois fazer BatchNorm
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 4, 4)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (4 , 4), strides=1, kernel_initializer=initializer, kernel_constraint=constraint)(x) # (bs, 512, 1, 1)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # print(x.shape)

    # Finaliza com uma Fully Connected 
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, kernel_constraint=constraint)(x)
    if not use_logits:
        x = tf.keras.layers.Activation('sigmoid')(x)
    
    return tf.keras.Model(inputs = inputs, outputs = x)


# Só roda quando este arquivo for chamado como main
if __name__ == "__main__":

    img_size = 128
    vec_size = 512

    # Testa os shapes dos modelos
    print("Geradores:")
    print("DCGAN                   ", dcgan_generator(img_size, vec_size).output.shape)
    print("Pix2Pix adapted decoder ", pix2pix_adapted_decoder(img_size, vec_size).output.shape)
    print("Resnet decoder          ", VT_resnet_decoder(img_size, vec_size).output.shape)
    print("Simple decoder          ", VT_simple_decoder(img_size, vec_size).output.shape)
    print("")
    print("Discriminadores:")
    print("DCGAN                   ", dcgan_discriminator(img_size).output.shape)
    print("PatchGAN                ", patchgan_discriminator(img_size).output.shape)
    print("ProGAN adapted          ", progan_adapted_discriminator(img_size).output.shape)