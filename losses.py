## Definition of the losses for the GANs used on this project

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

#%% DEFINIÇÃO DAS LOSSES

'''
PatchGAN: Em vez de o discriminador usar uma única predição (0 = falsa, 1 = real), o discriminador da PatchGAN (Pix2Pix e CycleGAN) usa
uma matriz 30x30x1, em que cada elemento equivale a uma região da imagem, e o discriminador tenta classificar cada região como normal ou falsa
# Adaptado para o treinamento não condicional:
- A Loss do gerador é basicamente a Loss de GAN que é a BCE entre a matriz 30x30x1 do gerador e uma matriz de mesma dimensão preenchida com "1"s
- A Loss do discriminador usa apenas a Loss de Gan, mas com uma matriz "0"s para a imagem do gerador (falsa) e uma de "1"s para a imagem real
'''
def loss_patchgan_generator(disc_fake_output, use_logits):
    # Lg = GANLoss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits = use_logits)
    gen_loss = BCE(tf.ones_like(disc_fake_output), disc_fake_output)
    return gen_loss

def loss_patchgan_discriminator(disc_real_output, disc_fake_output, use_logits):
    # Ld = RealLoss + FakeLoss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits = use_logits)
    real_loss = BCE(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = BCE(tf.zeros_like(disc_fake_output), disc_fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss, real_loss, fake_loss

'''
Wasserstein GAN (WGAN): A PatchGAN e as GANs clássicas usam a BCE como medida de distância entre a distribuição real e a inferida pelo gerador,
e esse método é na prática a divergência KL. A WGAN usa a distância de Wasserstein, que é mais estável, então evita o mode collapse.
- O discriminador tenta maximizar E(D(x_real)) - E(D(x_fake)), pois quanto maior a distância, pior o gerador está sendo
- O gerador tenta minimizar -E(D(x_fake)), ou seja, o valor esperado (média) da predição do discriminador para a sua imagem
- Os pesos do discriminador precisam passar por Clipping de -0.01 a 0.01 para garantir a continuidade de Lipschitz
'''
def loss_wgan_generator(disc_fake_output):
    # O output do discriminador é de tamanho BATCH_SIZE x 1, o valor esperado é a média
    gen_loss = -tf.reduce_mean(disc_fake_output)
    return gen_loss

def loss_wgan_discriminator(disc_real_output, disc_fake_output):
    # Maximizar E(D(x_real)) - E(D(x_fake)) é equivalente a minimizar -(E(D(x_real)) - E(D(x_fake))) ou E(D(x_fake)) -E(D(x_real))
    fake_loss = tf.reduce_mean(disc_fake_output)
    real_loss = tf.reduce_mean(disc_real_output)
    total_disc_loss = fake_loss - real_loss
    return total_disc_loss, real_loss, fake_loss

'''
Wasserstein GAN - Gradient Penalty (WGAN-GP): A WGAN tem uma forma muito bruta de assegurar a continuidade de Lipschitz, então
os autores criaram o conceito de Gradient Penalty para manter essa condição de uma forma mais suave.
- O gerador tem a MESMA loss da WGAN
- O discriminador, em vez de ter seus pesos limitados pelo clipping, ganham uma penalidade de gradiente que deve ser calculada
'''
def loss_wgangp_generator(disc_fake_output):
    return loss_wgan_generator(disc_fake_output)

def loss_wgangp_discriminator(disc_real_output, disc_fake_output, discriminator, real_img, fake_img, lambda_gp, training = 'direct'):
    real_loss = tf.reduce_mean(disc_real_output)
    fake_loss = tf.reduce_mean(disc_fake_output)
    gp = gradient_penalty(discriminator, real_img, fake_img, training)
    total_disc_loss = -(real_loss - fake_loss) + lambda_gp * gp + (0.001 * tf.reduce_mean(disc_real_output**2))
    return total_disc_loss, real_loss, fake_loss, gp

def gradient_penalty(discriminator, real_img, fake_img, training):
    ''' 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    '''
    # Get the Batch Size
    batch_size = real_img.shape[0]

    # Calcula gamma
    gamma = tf.random.uniform([batch_size, 1, 1, 1])

    # Calcula a imagem interpolada
    interpolated = real_img * gamma + fake_img * (1 - gamma)

    with tf.GradientTape() as gp_tape:

        gp_tape.watch(interpolated)

        # 1. Get the discriminator output for this interpolated image.
        if training == 'direct':
            pred = discriminator(interpolated, training=True) # O discriminador usa duas imagens como entrada
        elif training == 'progressive':
            pred = discriminator(interpolated)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
