#%% INÍCIO

### Imports
import os
import time
from math import ceil, log2

from matplotlib import pyplot as plt
import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks as net
import utils

#%% Weights & Biases

# wandb.init(project='gan_inversion', entity='vinyluis', mode="disabled")
wandb.init(project='gan_inversion', entity='vinyluis', mode="online")
# wandb.init(project='gan_inversion', entity='vinyluis', mode="online", resume="1jx53k66")

#%% Config Tensorflow
# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
print(tf.config.list_physical_devices('GPU'))
print("")


#%% HIPERPARÂMETROS
config = wandb.config # Salva os hiperparametros no Weights & Biases também

# Root do sistema
# base_root = "../"
base_root = ""

# Parâmetros de treinamento
config.BATCH_SIZE = 16
config.BUFFER_SIZE = 150
config.LEARNING_RATE_G = 1e-3
config.LEARNING_RATE_D = 1e-3
config.EPOCHS = 3
config.LAMBDA_GP = 10 # Intensidade do Gradient Penalty da WGAN-GP
config.CACHE_DATASET = True
config.USE_RANDOM_JITTER = False
config.DATASET = "InsetosFlickr" # "CelebaHQ" ou "InsetosFlickr"

# Parâmetros de modelo
config.DISENTANGLEMENT = False
config.VEC_SIZE = 256
config.EVALUATE_ACCURACY = False
config.DISCRIMINATOR_USE_LOGITS = True
config.IMG_SIZE = 128 # Tamanho da imagem no treinamento direto, ou tamanho máximo no caso do treinamento progressivo
# config.update({"IMG_SIZE" : 64}, allow_val_change = True)

# Específico do treinamento progressivo
config.BATCH_SIZES = [128, 64, 32, 16, 8, 4, 2, 2, 1]
# config.update({"BATCH_SIZES" : [[128, 64, 32, 16, 8, 4, 2, 2, 1]}, allow_val_change = True)
config.EPOCHS_FADE_IN = 10 # Quantas épocas por "step" de crescimento faremos o fade-in (variação do alpha)
config.EPOCHS_NORMAL = 20 # Quantas épocas por "step" de crescimento após o final do fade-in
config.CHANNELS = 256 # Quantidade máxima de canais para o treinamento progressivo
config.STEPS = int(log2(config.IMG_SIZE / 4)) # Quantidade de "steps" até chegar em IMG_SIZE
# config.update({"STEPS" : int(log2(config.IMG_SIZE / 4))}, allow_val_change = True)

# Parâmetros de plot
config.QUIET_PLOT = True
config.NUM_TEST_PRINTS = 10

# Controle do Checkpoint
config.SAVE_CHECKPOINT = True
config.LOAD_CHECKPOINT = True
config.CHECKPOINT_EPOCHS = 1
config.KEEP_CHECKPOINTS = 1
config.SAVE_MODELS = False

# Permite alterar variáveis do config
wandb.config.update(config, allow_val_change = True)

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = '05I'

# Modelo do gerador. Possíveis direto = 'dcgan', 'pix2pix_adapted', 'resnet_decoder', 'simple_decoder'
# Possíveis com treinamento progressivo = 'progan'
config.gen_model = 'progan'

# Modelo do discriminador. Possíveis = 'dcgan', 'patchgan', 'progan_adapted'
# Possíveis com treinamento progressivo = 'progan'
config.disc_model = 'progan'

# Tipo de treinamento. Possíveis = 'direct', 'progressive'
config.training_type = 'progressive'

# Tipo de loss. Possíveis = 'ganloss', 'wgan', 'wgan-gp'
config.loss_type = 'wgan-gp'


# Valida se pode ser usado o tipo de loss com o tipo de discriminador
if config.loss_type == 'ganloss':
    config.ADAM_BETA_1 = 0.5
elif config.loss_type == 'wgan' or config.loss_type == 'wgan-gp':
    config.ADAM_BETA_1 = 0 if config.training_type == 'progressive' else 0.9
    if not(config.disc_model == 'progan_adapted' or config.disc_model == 'dcgan' or config.disc_model == 'progan'):
        raise utils.LossCompatibilityError(config.loss_type, config.disc_model)
else:
    config.ADAM_BETA_1 = 0.9

# Valida se o tipo de treinamento pode ser usado com o gerador ou o discriminador atual
if config.training_type == 'progressive':
    if config.gen_model != 'progan' or config.disc_model != 'progan':
        raise utils.TrainingCompatibilityError(config.training_type, config.gen_model, config.disc_model)
elif config.training_type == 'direct':
    if config.gen_model == 'progan' or config.disc_model == 'progan':
        raise utils.TrainingCompatibilityError(config.training_type, config.gen_model, config.disc_model)

# Valida o IMG_SIZE
if not(config.IMG_SIZE == 256 or config.IMG_SIZE == 128) and config.training_type == 'direct':
    raise utils.SizeCompatibilityError(config.IMG_SIZE)


#%% Prepara as pastas

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP' + config.exp + '_'
experiment_folder += 'gen_'
experiment_folder += config.gen_model
experiment_folder += '_disc_'
experiment_folder += config.disc_model
experiment_folder += '_training_'
experiment_folder += config.training_type
experiment_folder += '/'

### Pastas do dataset
if config.DATASET == 'CelebaHQ':
    dataset_folder = 'C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'
    # dataset_folder = 'C:/Users/Vinícius/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'

    train_folder = dataset_folder + 'train/'
    test_folder = dataset_folder + 'val/'

    full_dataset_string = dataset_folder + '*/*/*.jpg'
    train_dataset_string = train_folder + '*/*.jpg'
    test_dataset_string = test_folder + '*/*.jpg'
    dataset_string = full_dataset_string

if config.DATASET == 'InsetosFlickr':
    dataset_folder = "C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/flickr_internetarchivebookimages_prepared"
    full_dataset_string = dataset_folder + '*/*/*.jpg'
    dataset_string = full_dataset_string

### Pastas dos resultados
result_folder = experiment_folder + 'results/'
model_folder = experiment_folder + 'model/'

### Cria as pastas, se não existirem
if not os.path.exists(experiment_root):
    os.mkdir(experiment_root)

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

if not os.path.exists(result_folder):
    os.mkdir(result_folder)
    
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

#%% CONTROLE DO CHECKPOINT

# Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Definição de época e step para controle no ckeckpoint
tf_epoch = tf.Variable(0)
tf_step = tf.Variable(0)

#%% DEFINIÇÃO DAS LOSSES

# Configura o Binary Cross Entropy de acordo com o DISCRIMINATOR_USE_LOGITS
BCE = tf.keras.losses.BinaryCrossentropy(from_logits = config.DISCRIMINATOR_USE_LOGITS)

'''
PatchGAN: Em vez de o discriminador usar uma única predição (0 = falsa, 1 = real), o discriminador da PatchGAN (Pix2Pix e CycleGAN) usa
uma matriz 30x30x1, em que cada elemento equivale a uma região da imagem, e o discriminador tenta classificar cada região como normal ou falsa
# Adaptado para o treinamento não condicional:
- A Loss do gerador é basicamente a Loss de GAN que é a BCE entre a matriz 30x30x1 do gerador e uma matriz de mesma dimensão preenchida com "1"s
- A Loss do discriminador usa apenas a Loss de Gan, mas com uma matriz "0"s para a imagem do gerador (falsa) e uma de "1"s para a imagem real
'''
def loss_patchgan_generator(disc_fake_output):
    # Lg = GANLoss
    gen_loss = BCE(tf.ones_like(disc_fake_output), disc_fake_output)
    return gen_loss

def loss_patchgan_discriminator(disc_real_output, disc_fake_output):
    # Ld = RealLoss + FakeLoss
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

"""
def loss_wgangp_discriminator(disc_real_output, disc_fake_output, discriminator, real_img, fake_img, training = 'direct', alpha = 0, step = 0):
    total_disc_loss, real_loss, fake_loss = loss_wgan_discriminator(disc_real_output, disc_fake_output)
    gp = gradient_penalty(discriminator, real_img, fake_img, training, alpha, step)
    total_disc_loss = total_disc_loss + config.LAMBDA_GP * gp
    return total_disc_loss, real_loss, fake_loss, gp

def gradient_penalty(discriminator, real_img, fake_img, training, alpha, step):
    ''' 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    '''
    # Get the Batch Size
    batch_size = real_img.shape[0]

    # Calculates gamma
    gamma = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)

    diff = fake_img - real_img
    interpolated = real_img + gamma * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)

        # 1. Get the discriminator output for this interpolated image.
        if training == 'direct':
            pred = discriminator(interpolated, training=True) # O discriminador usa duas imagens como entrada
        elif training == 'progressive':
            pred = discriminator(interpolated, alpha, step)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
"""

# Acertada para condizer com o tutorial do Aladdin Persson
# https://www.youtube.com/watch?v=nkQHASviYac
def loss_wgangp_discriminator(disc_real_output, disc_fake_output, discriminator, real_img, fake_img, training = 'direct', alpha = 0, step = 0):
    real_loss = tf.reduce_mean(disc_real_output)
    fake_loss = tf.reduce_mean(disc_fake_output)
    gp = gradient_penalty(discriminator, real_img, fake_img, training, alpha, step)
    total_disc_loss = -(real_loss - fake_loss) + config.LAMBDA_GP * gp + (0.001 * tf.reduce_mean(disc_real_output**2))
    return total_disc_loss, real_loss, fake_loss, gp

def gradient_penalty(discriminator, real_img, fake_img, training, alpha, step):
    ''' 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
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
            pred = discriminator(interpolated, alpha, step)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

#%% FUNÇÕES DO TREINAMENTO

'''
FUNÇÕES DE TREINAMENTO DIRETO
'''

@tf.function
def train_step_direct(generator, discriminator, real_image, input_vector):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Gera a imagem e faz as discriminações
        fake_image = generator(input_vector)
        disc_real = discriminator(real_image, training=True)
        disc_fake = discriminator(fake_image, training=True)

        if config.loss_type == 'ganloss':
            gen_loss = loss_patchgan_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss = loss_patchgan_discriminator(disc_real, disc_fake)
            gp = 0

        elif config.loss_type == 'wgan':
            gen_loss = loss_wgan_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss = loss_wgan_discriminator(disc_real, disc_fake)
            gp = 0

        elif config.loss_type == 'wgan-gp':
            gen_loss = loss_wgangp_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss, gp = loss_wgangp_discriminator(disc_real, disc_fake, discriminator, real_image, fake_image)

        # Incluído o else para não dar erro 'gen_loss' is used before assignment
        else:
            gen_loss = 0
            disc_loss = 0
            print("Erro de modelo. Selecione uma Loss válida")

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return (gen_loss, disc_loss, disc_real_loss, disc_fake_loss, gp)

def fit_direct(generator, discriminator, tf_epoch, epochs):

    first_epoch = tf_epoch.numpy()

    # Prepara o dataset de treino
    train_ds, config.TRAIN_SIZE = utils.prepare_dataset(dataset_string, config.IMG_SIZE, config.BATCH_SIZE, config.BUFFER_SIZE)
    print("O dataset de treino tem {} imagens".format(config.TRAIN_SIZE))
    
    # Listas para o cálculo da acurácia do discriminador
    y_real = []
    y_pred = []

    # Prepara a progression bar
    progbar = tf.keras.utils.Progbar(int(ceil(config.TRAIN_SIZE / config.BATCH_SIZE)))

    # Cria um fixed noise de 16 imagens para ver a evolução da rede
    fixed_noise = tf.random.normal(shape = [16, config.VEC_SIZE])

    # Mostra como está a geração pelo fixed_noise antes do treinamento
    filename = "epoca_" + str(first_epoch-1).zfill(len(str(config.EPOCHS))) + ".jpg"
    fig = utils.print_generated_images(generator, fixed_noise, result_folder, filename)

    # Loga a figura no wandb
    s = "Época {}".format(first_epoch-1)
    wandbfig = wandb.Image(fig, caption="epoca:{}".format(first_epoch-1))
    wandb.log({s: wandbfig})

    if config.QUIET_PLOT:
        plt.close(fig)

    ########## LOOP DE TREINAMENTO ##########
    for epoch in range(first_epoch, epochs):
        t_start = time.time()
        
        print(utils.get_time_string(), " - Época: ", epoch)

        # Train
        i = 0 # Para o progress bar
        for n, real_image in train_ds.enumerate():

            # Faz o update da Progress Bar
            i += 1
            progbar.update(i)

            # Gera o vetor de ruído aleatório
            noise = tf.random.normal(shape = [config.BATCH_SIZE, config.VEC_SIZE])
            
            # Step de treinamento
            gen_loss, disc_loss, disc_real_loss, disc_fake_loss, gp = train_step_direct(generator, discriminator, real_image, noise)

            # Calcula a acurácia:
            if config.EVALUATE_ACCURACY:
                y_real, y_pred, acc = utils.evaluate_accuracy(generator, discriminator, real_image, noise, y_real, y_pred)
            else:
                acc = 0

            # Log as métricas no wandb 
            wandb.log({ 'gen_loss': gen_loss.numpy(), 'disc_loss': disc_loss.numpy(), 'disc_real_loss': disc_real_loss.numpy(),
                        'disc_fake_loss': disc_fake_loss.numpy(), 'gradient_penalty': gp.numpy(), 'accuracy': acc})            
        
        # Atualiza o tracker de época
        tf_epoch.assign_add(1)

        # Salva o checkpoint
        if config.SAVE_CHECKPOINT and ((epoch) % config.CHECKPOINT_EPOCHS == 0):
            ckpt_manager.save()
            print("\nSalvando checkpoint...")        

        # Mostra o andamento do treinamento com uma imagem sintética do fixed noise
        filename = "epoca_" + str(epoch).zfill(len(str(config.EPOCHS))) + ".jpg"
        fig = utils.print_generated_images(generator, fixed_noise, result_folder, filename)
        # Loga a figura no wandb
        s = "Época {}".format(epoch)
        wandbfig = wandb.Image(fig, caption="epoca:{}".format(epoch))
        wandb.log({s: wandbfig})
        # Fecha a figura, se necessário
        if config.QUIET_PLOT:
            plt.close(fig)
        
        dt = time.time() - t_start
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
        

'''
FUNÇÕES DE TREINAMENTO PROGRESSIVO
'''

def train_step_progressive(generator, discriminator, real_image, input_vector, alpha, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Gera a imagem e faz as discriminações
        fake_image = generator(input_vector, alpha, step)
        disc_real = discriminator(real_image, alpha, step)
        disc_fake = discriminator(fake_image, alpha, step)

        if config.loss_type == 'ganloss':
            gen_loss = loss_patchgan_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss = loss_patchgan_discriminator(disc_real, disc_fake)
            gp = 0

        elif config.loss_type == 'wgan':
            gen_loss = loss_wgan_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss = loss_wgan_discriminator(disc_real, disc_fake)
            gp = 0

        elif config.loss_type == 'wgan-gp':
            gen_loss = loss_wgangp_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss, gp = loss_wgangp_discriminator(disc_real,
                    disc_fake, discriminator, real_image, fake_image, 'progressive', alpha, step)

        # Incluído o else para não dar erro 'gen_loss' is used before assignment
        else:
            gen_loss = 0
            disc_loss = 0
            print("Erro de modelo. Selecione uma Loss válida")

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return (gen_loss, disc_loss, disc_real_loss, disc_fake_loss, gp)

def fit_progressive(generator, discriminator, tf_epoch, tf_step):
    
    # Recupera o step e a época atual
    first_step = tf_step.numpy()
    first_epoch = tf_epoch.numpy()
    
    # Listas para o cálculo da acurácia do discriminador
    y_real = []
    y_pred = []

    # Cria um fixed noise de 16 imagens para ver a evolução da rede
    fixed_noise = tf.random.normal(shape = [16, config.VEC_SIZE])
    utils.print_fixed_noise(fixed_noise, result_folder, "Ruído Fixo")

    # Mostra como está a geração pelo fixed_noise antes do treinamento
    if first_step == 0 and first_epoch == 0:
        filename = "Estado_Inicial.jpg"
        fig = utils.print_generated_images_prog(generator, fixed_noise, 1, 0, result_folder, filename)

        # Loga a figura no wandb
        s = "Estado Inicial"
        wandbfig = wandb.Image(fig, caption="estado_inicial")
        wandb.log({s: wandbfig})

        if config.QUIET_PLOT:
            plt.close(fig)

    ########## LOOP DE TREINAMENTO ##########
    t_start_step = time.time()
    for step in range(first_step, config.STEPS + 1):
        curr_img_size = 4*(2**step)
        curr_batch_size = config.BATCH_SIZES[step]
        print(utils.get_time_string(), f" - Passo {step}: Imagem {curr_img_size}x{curr_img_size}")

        # Define o dataset usado neste step
        train_ds, config.TRAIN_SIZE = utils.prepare_dataset(dataset_string, curr_img_size, curr_batch_size, 
                                      config.BUFFER_SIZE, use_jitter = config.USE_RANDOM_JITTER, use_cache = config.CACHE_DATASET)
        # print("O dataset de treino tem {} imagens".format(config.TRAIN_SIZE))

        # Prepara a progression bar
        total_iterations = int(ceil(config.TRAIN_SIZE / curr_batch_size))
        progbar = tf.keras.utils.Progbar(total_iterations)

        for epoch in range(first_epoch, config.EPOCHS_FADE_IN + config.EPOCHS_NORMAL):
            t_start = time.time()
            print(utils.get_time_string(), " - Época: ", epoch)

            # Define o alpha
            if epoch < config.EPOCHS_FADE_IN:
                alpha_min = (epoch)/(config.EPOCHS_FADE_IN)
                alpha_max = (epoch+1)/(config.EPOCHS_FADE_IN)
            else:
                alpha_min = 1
                alpha_max = 1
            print(f"Alpha: {alpha_min:.2f} - {alpha_max:.2f}")

            # Train
            i = 0 # Para o progress bar
            for n, real_image in train_ds.enumerate():

                # Faz o update da Progress Bar
                i += 1
                progbar.update(i)

                # Define o alpha
                if epoch < config.EPOCHS_FADE_IN:
                    inc = i / total_iterations
                    alpha = alpha_min + inc * (alpha_max - alpha_min)
                else: 
                    alpha = 1

                # Gera o vetor de ruído aleatório
                noise = tf.random.normal(shape = [real_image.shape[0], config.VEC_SIZE])

                # Step de treinamento
                gen_loss, disc_loss, disc_real_loss, disc_fake_loss, gp = train_step_progressive(generator, discriminator, 
                                                                          real_image, noise, alpha, step)

                # Calcula a acurácia:
                if config.EVALUATE_ACCURACY:
                    y_real, y_pred, acc = utils.evaluate_accuracy(generator, discriminator, real_image, noise, y_real, y_pred, training = 'progressive', alpha = alpha, step = step)
                else:
                    acc = 0

                # Acerta o GP
                gp = gp.numpy() if type(gp) != int else gp 
                # Log as métricas no wandb 
                wandb.log({ 'gen_loss': gen_loss.numpy(), 'disc_loss': disc_loss.numpy(), 'disc_real_loss': disc_real_loss.numpy(),
                            'disc_fake_loss': disc_fake_loss.numpy(), 'gradient_penalty': gp, 'accuracy': acc, 
                            'alpha': alpha, 'step': step})            
            
            # Mostra o andamento do treinamento com uma imagem sintética do fixed noise
            filename = "passo_" + str(step).zfill(len(str(config.STEPS)))
            filename += "_epoca_" + str(epoch).zfill(len(str(config.EPOCHS))) + ".jpg"
            fig = utils.print_generated_images_prog(generator, fixed_noise, 1, step, result_folder, filename)
            # Loga a figura no wandb
            s = f"Passo {step} Época {epoch}"
            wandbfig = wandb.Image(fig, caption=f"passo{step}_epoca:{epoch}")
            wandb.log({s: wandbfig})
            # Fecha a figura, se necessário
            if config.QUIET_PLOT:
                plt.close(fig)
            
            # Calcula o tempo da época
            dt = time.time() - t_start
            print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
            wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})

            # Atualiza o tracker de época
            tf_epoch.assign_add(1)

            # Salva o checkpoint
            if config.SAVE_CHECKPOINT and ((epoch) % config.CHECKPOINT_EPOCHS == 0):
                ckpt_manager.save()
                print("Salvando checkpoint...\n")

        # Atualiza o tracker de step
        tf_step.assign_add(1)

        # Reseta o tracker de época
        tf_epoch.assign(0)
        first_epoch = 0

        # Salva o checkpoint
        if config.SAVE_CHECKPOINT and ((epoch) % config.CHECKPOINT_EPOCHS == 0):
            ckpt_manager.save()
            print("Salvando checkpoint...\n")

#%% TESTA O CÓDIGO E MOSTRA UMA IMAGEM DO DATASET

if config.DATASET == "CelebaHQ":
    inp = utils.load(train_folder+'/female/000085.jpg')
if config.DATASET == "InsetosFlickr":
    inp = utils.load(dataset_folder+'/trainB/insects_winged_001_d.jpg')

# casting to int for matplotlib to show the image
if not config.QUIET_PLOT:
    plt.figure()
    plt.imshow(inp/255.0)

#%% PREPARAÇÃO DOS MODELOS

# Define se irá ter a restrição de tamanho de peso da WGAN (clipping)
constrained = False
if config.loss_type == 'wgan':
        constrained = True

# CRIANDO O MODELO DE GERADOR
if config.gen_model == 'dcgan': 
    generator = net.dcgan_generator(config.IMG_SIZE, config.VEC_SIZE, disentanglement = config.DISENTANGLEMENT)
elif config.gen_model == 'pix2pix_adapted': 
    generator = net.pix2pix_adapted_decoder(config.IMG_SIZE, config.VEC_SIZE, disentanglement = config.DISENTANGLEMENT)
elif config.gen_model == 'resnet_decoder':
    generator = net.VT_resnet_decoder(config.IMG_SIZE, config.VEC_SIZE, disentanglement = config.DISENTANGLEMENT)
elif config.gen_model == 'simple_decoder':
    generator = net.VT_simple_decoder(config.IMG_SIZE, config.VEC_SIZE, disentanglement = config.DISENTANGLEMENT)
elif config.gen_model == 'progan':
    generator = net.progan_generator(config.CHANNELS, 3, config.VEC_SIZE, config.DISENTANGLEMENT)
else:
    raise utils.GeneratorError(config.gen_model)

# CRIANDO O MODELO DE DISCRIMINADOR
if config.disc_model == 'dcgan':
    discriminator = net.dcgan_discriminator(config.IMG_SIZE, constrained = constrained, use_logits = config.DISCRIMINATOR_USE_LOGITS)
elif config.disc_model == 'patchgan':
    discriminator = net.patchgan_discriminator(config.IMG_SIZE, constrained = constrained, use_logits = config.DISCRIMINATOR_USE_LOGITS)
elif config.disc_model == 'progan_adapted':
    discriminator = net.progan_adapted_discriminator(config.IMG_SIZE, constrained = constrained, use_logits = config.DISCRIMINATOR_USE_LOGITS)
elif config.disc_model == 'progan':
    discriminator = net.progan_discriminator(config.CHANNELS, 3)
else:
    raise utils.DiscriminatorError(config.disc_model)

# Define os otimizadores
generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_G, beta_1=config.ADAM_BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_D, beta_1=config.ADAM_BETA_1)

#%% EXECUÇÃO

# Prepara o checkpoint 
if config.training_type == 'progressive':
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    disc = discriminator,
                                    tf_epoch = tf_epoch,
                                    tf_step = tf_step)
else:
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    disc = discriminator,
                                    tf_epoch = tf_epoch)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep = config.KEEP_CHECKPOINTS)

# Se for carregar o checkpoint mais recente
if config.LOAD_CHECKPOINT:
    print("Carregando checkpoint mais recente, se houver...")
    restore = ckpt_manager.restore_or_initialize()
    if restore != None:
        print(f"Carregado - Passo {tf_step.numpy()} e Época {tf_epoch.numpy()}")
    print("")

# Salva os modelos (principalmente para visualização)
if config.SAVE_MODELS:
    generator.save(model_folder+'ae_generator.h5')
    discriminator.save(model_folder+'ae_discriminator.h5')

#%% TREINAMENTO

# Treinamento direto
if config.training_type == 'direct':
    if (tf_epoch.numpy() <= (config.EPOCHS)):
        fit_direct(generator, discriminator, tf_epoch, config.EPOCHS)
    else:
        ckpt_manager.save()

# Treinamento progressivo
elif config.training_type == 'progressive':
    if (tf_step.numpy() <= config.STEPS):
        fit_progressive(generator, discriminator, tf_epoch, tf_step)
    else:
        ckpt_manager.save()

else:
    raise utils.TrainingTypeError(config.training_type)


#%% VALIDAÇÃO

# Finaliza o Weights and Biases
wandb.finish()

## Salva os modelos
if config.SAVE_MODELS:
    generator.save(model_folder+'ae_generator.h5')
    discriminator.save(model_folder+'ae_discriminator.h5')
