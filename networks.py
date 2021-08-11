import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_addons as tfa

## Tipo de normalização
# norm_layer = tf.keras.layers.BatchNormalization
norm_layer = tfa.layers.InstanceNormalization

## Modo de inicialização dos pesos
# initializer = tf.random_uniform_initializer()
# initializer = tf.random_normal_initializer(0., 0.02)
initializer = tf.random_normal_initializer()

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


#%% BLOCOS DA PROGAN

# Bloco de convolução com Normalized Learning Rate (Weight-Scaled) - ProGAN
class WSConv2D(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels, kernel_size = 3, strides = 1, padding = 'same', gain = 2):
        super(WSConv2D, self).__init__()

        # A escala é basicamente o sqrt de (ganho dividido por (kernel² * canais da camada anterior))
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        # Cria a camada de convolução
        self.conv = tf.keras.layers.Conv2D(filters = out_channels, kernel_size = kernel_size, strides = strides,
                                           padding = padding, kernel_initializer = initializer) 
        # O bias não precisa ser escalado
        # self.bias = self.conv.bias # Salva o bias
        # self.conv.bias = None # Zera o bias da convolução para ela não sofrer o efeito da escala    

    def call(self, inputs):
        # return self.conv(inputs * self.scale) + self.bias
        return self.conv(inputs * self.scale)

# Bloco básico da ProGAN
class progan_block(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels, use_norm = True):
        super(progan_block, self).__init__()

        self.conv1 = WSConv2D(in_channels, out_channels)
        self.conv2 = WSConv2D(out_channels, out_channels)
        self.leaky = tf.keras.layers.LeakyReLU(0.2)
        self.use_norm = use_norm
        self.norm_layer = norm_layer

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.norm_layer()(x) if self.use_norm else x
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.norm_layer()(x) if self.use_norm else x
        return x
    
# Pixel Normalization (usada no paper original)
class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def call(self, inputs):
        x = inputs
        # axis = -1 -> A normalização atua nos canais
        return x / tf.sqrt(tf.reduce_mean(x**2, axis = -1, keepdims = True) + self.epsilon)


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


class progan_generator(tf.keras.Model):

    def __init__(self, CHANNELS = 512, IMG_CHANNELS = 3, VEC_SIZE = 512, disentanglement = False):

        super(progan_generator, self).__init__()
        self.disentanglement = disentanglement
        self.norm_layer = PixelNorm

        # CHANNELS = Quantidade de canais que será usada na construção das primeiras camadas da rede.
        # IMG_CHANNELS = 3 para RGB e 1 para grayscale 

        ### CRIA OS BLOCOS

        # As camadas começam com o número de canais definido em CHANNELS
        # Para cada iteração de crescimento, o CHANNELS é dividido pelo factors correspondente
        self.factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

        # Blocos iniciais do gerador
        self.initial_block = tf.keras.models.Sequential(layers = [
                self.norm_layer(),
                tf.keras.layers.Conv2DTranspose(VEC_SIZE, 4, 1, padding = 'valid'), # 1x1 -> 4x4
                tf.keras.layers.LeakyReLU(0.2),
                WSConv2D(CHANNELS, CHANNELS, kernel_size = 3, strides = 1, padding = 'same'),
                tf.keras.layers.LeakyReLU(0.2),
                self.norm_layer()
            ])
        
        self.initial_rgb = WSConv2D(CHANNELS, IMG_CHANNELS, kernel_size = 1, strides = 1, padding = 'valid')

        # Sequência de crescimento progressivo
        self.prog_blocks = []
        self.rgb_layers = [self.initial_rgb]

        for i in range(len(self.factors) - 1):
                # factors[i] -> factors[i+1]
                conv_in_channels = int(CHANNELS * self.factors[i]) # in channels for the block
                conv_out_channels = int(CHANNELS * self.factors[i+1]) # out channels for the block
                self.prog_blocks.append(progan_block(conv_in_channels, conv_out_channels))
                self.rgb_layers.append(WSConv2D(conv_out_channels, IMG_CHANNELS, kernel_size = 1, strides = 1, padding = 'valid'))

    
    def fade_in(self, alpha, upscaled, generated):
        # on the start, alpha is 0 and the network sends forward the image of the last layer upscaled
        # as alpha grows, the new layer gets more important over time, until alpha = 1
        return tf.tanh(alpha * generated + (1 - alpha) * upscaled)


    def call(self, inputs, alpha, steps):

        ### FAZ O "FORWARD PROPAGATION"
        # steps = 0 -> 4x4, steps = 1 -> 8x8, ...

        # O input será um vetor, então temos que expandir a dimensão dele
        # Transforma novamente num tensor de terceira ordem
        inputs = tf.expand_dims(inputs, axis = 1)
        inputs = tf.expand_dims(inputs, axis = 1)

        # print(inputs.shape)

        # A rede sempre começa com o "initial_block"
        out = self.initial_block(inputs) # 4x4

        # Se for o primeiro bloco, não há mais nada, só transformar para RGB
        if steps == 0:
            # return self.initial_rgb(out)
            return tf.tanh(self.initial_rgb(out))

        # Para próximos passos do crescimento:
        for step in range(steps):
            upscaled = tf.keras.layers.UpSampling2D(size = (2, 2), interpolation = "nearest")(out)
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        # Retorna o resultado com o fadein
        return self.fade_in(alpha, final_upscaled, final_out)


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


class progan_discriminator(tf.keras.Model):

    def __init__(self, CHANNELS = 512, IMG_CHANNELS = 3):

        super(progan_discriminator, self).__init__()
        self.norm_layer = PixelNorm
        self.leaky = tf.keras.layers.LeakyReLU(0.2)

        # CHANNELS = Quantidade de canais que será usada na construção das primeiras camadas da rede.
        # IMG_CHANNELS = 3 para RGB e 1 para grayscale 

        ### CRIA OS BLOCOS
        # O discriminador é o inverso do gerador

        self.factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

        # Sequência de crescimento progressivo
        self.prog_blocks = []
        self.rgb_layers = []

        for i in range(len(self.factors) -1, 0, -1):
                # factors[i] -> factors[i+1]
                conv_in_channels = int(CHANNELS * self.factors[i]) # in channels for the block
                conv_out_channels = int(CHANNELS * self.factors[i-1]) # out channels for the block
                self.prog_blocks.append(progan_block(conv_in_channels, conv_out_channels))
                self.rgb_layers.append(WSConv2D(IMG_CHANNELS, conv_in_channels, kernel_size = 1, strides = 1, padding = 'valid'))

        # Blocos finais do discriminador
        self.final_block = tf.keras.models.Sequential(layers = [
            WSConv2D(CHANNELS+1, CHANNELS, kernel_size=3, strides = 1, padding = 'same'),
            self.leaky,
            WSConv2D(CHANNELS, CHANNELS, kernel_size = 4, strides = 1, padding = 'valid'),
            self.leaky,
            # This last layer replaces the fully connected, without the need to build a WSDense
            WSConv2D(CHANNELS, 1, kernel_size=1, strides = 1, padding = 'valid')
            ])
        
        self.final_rgb = WSConv2D(IMG_CHANNELS, CHANNELS, kernel_size = 1, strides = 1, padding = 'valid')
        self.rgb_layers.append(self.final_rgb)
        
        # Average Pooling (usado para o downscale)
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size = 2, strides = 2)
        

    def fade_in(self, alpha, downscaled, out):
        # on the start, alpha is 0 and the network sends forward the image of the last layer upscaled
        # as alpha grows, the new layer gets more important over time, until alpha = 1
        return alpha * out + (1 - alpha) * downscaled

    # Minibatch std - não vou usar por enquanto
    def minibatch_std(self, x):
        batch_statistics = tf.math.reduce_mean(tf.math.reduce_std(x, axis = 0)) # é um número escalar
        constant_feature_map = tf.reduce_mean(tf.ones_like(x), axis = -1, keepdims = True) # Cria um feature map do tamanho de x, com um único canal
        constant_feature_map = constant_feature_map * batch_statistics # Aplica a batch_statistics em todo o feature map
        return tf.concat([x, constant_feature_map], axis = -1)

    def call(self, inputs, alpha, steps):

        ### FAZ O "FORWARD PROPAGATION"
        # steps = 0 -> 4x4, steps = 1 -> 8x8, ...
        x = inputs

        # Iverte a lógica para que seja possível usar os mesmos passos do gerador
        cur_step = len(self.prog_blocks) - steps 

        # Passa pela primeira RGB layer
        out = self.leaky(self.rgb_layers[cur_step](x))

        # Se steps == 0, estamos na situação 4x4 e o único bloco vai ser o final
        if steps == 0:
            out = self.minibatch_std(out)
            out = self.final_block(out)
            return tf.keras.layers.Flatten()(out)

        # Para entender melhor, olhar o esquema no paper da ProGAN.
        # Na parte que vem do "downscaled" precisamos diminuir a dimensão ANTES do from_rgb
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))

        # Na parte que vem do "out" a ordem é "from rgb" -> "conv" -> downscale -> resto do modelo
        out = self.avg_pool(self.prog_blocks[cur_step](out)) # O out já passou por um from_rgb

        # Fade in
        out = self.fade_in(alpha, downscaled, out)

        # Crescimento progressivo. Como já fizemos o "curr_step", precisamos fazer os anteriores
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        out = self.final_block(out)
        return tf.keras.layers.Flatten()(out)


#%% TESTE DAS REDES
#  Só roda quando este arquivo for chamado como main
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
    print("")

    # Testa os modelos progressivos
    print("--- Teste da ProGAN ---")
    from math import log2
    CHANNELS = 256
    save_models = True

    gen = progan_generator(CHANNELS, 3, vec_size)
    disc = progan_discriminator(CHANNELS, 3)
    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        inp = tf.random.normal(shape = [1, vec_size])
        gen_img = gen(inp, alpha = 0.5, steps = num_steps)
        assert gen_img.shape == (1, img_size, img_size, 3)
        out = disc(gen_img, alpha= 0.5, steps = num_steps)
        assert out.shape == (1, 1)
        print(f"Sucesso com img_size: {img_size}")