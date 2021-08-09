#%% INÍCIO

### Imports
import os
import time
from math import ceil

import pandas as pd
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks as net
import utils

#%% Weights & Biases

import wandb
# wandb.init(project='gan_inversion', entity='vinyluis', mode="disabled")
wandb.init(project='gan_inversion', entity='vinyluis', mode="online")

#%% Config Tensorflow

# Evita o erro "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
# tfconfig = tf.compat.v1.ConfigProto()
# tfconfig.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=tfconfig)

# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
print(tf.config.list_physical_devices('GPU'))
# Verifica se a GPU está sendo usada na sessão
# print("---- VERIFICA SE A GPU ESTÁ SENDO USADA NA SESSÃO:")
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(sess)
print("")


#%% HIPERPARÂMETROS
config = wandb.config # Salva os hiperparametros no Weights & Biases também

# Root do sistema
# base_root = "../"
base_root = ""

# Parâmetros de treinamento
config.BATCH_SIZE = 16
config.BUFFER_SIZE = 150
config.LEARNING_RATE_G = 1e-4
config.LEARNING_RATE_D = 1e-4
config.EPOCHS = 3
config.LAMBDA_GP = 10 # Intensidade do Gradient Penalty da WGAN-GP
# config.ADAM_BETA_1 = 0.5 #0.5 para a PatchGAN e 0.9 para a WGAN - Definido no código

# Parâmetros de modelo
config.DISENTANGLEMENT = False
config.VEC_SIZE = 512
config.IMG_SIZE = 128 # Tamanho máximo da imagem, no caso do treinamento progressivo
config.EVALUATE_ACCURACY = True
config.DISCRIMINATOR_USE_LOGITS = True

# Parâmetros de plot
config.QUIET_PLOT = True
config.NUM_TEST_PRINTS = 10

# Controle do Checkpoint
config.CHECKPOINT_EPOCHS = 1
config.LOAD_CHECKPOINT = True
config.KEEP_CHECKPOINTS = 1

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = "03C"

# Modelo do gerador. Possíveis = 'dcgan', 'pix2pix_adapted', 'resnet_decoder', 'simple_decoder'
config.gen_model = 'resnet_decoder'

# Modelo do discriminador. Possíveis = 'dcgan', 'patchgan', 'progan_adapted'
config.disc_model = 'progan_adapted'

# Tipo de treinamento. Possíveis = 'direct', 'progressive'
config.training_type = 'direct'

# Tipo de loss. Possíveis = 'patchganloss', 'wgan', 'wgan-gp'
config.loss_type = 'patchganloss'
    
# Valida se pode ser usado o tipo de loss com o tipo de discriminador
if config.loss_type == 'patchganloss':
    config.ADAM_BETA_1 = 0.5
elif config.loss_type == 'wgan' or config.loss_type == 'wgan-gp':
    config.ADAM_BETA_1 = 0.9
    if not(config.disc_model == 'progan_adapted' or config.disc_model == 'dcgan'):
        raise utils.LossCompatibilityError(config.loss_type, config.disc_model)
else:
    config.ADAM_BETA_1 = 0.9

# Valida o IMG_SIZE
if not(config.IMG_SIZE == 256 or config.IMG_SIZE == 128):
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
dataset_folder = 'C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'
# dataset_folder = 'C:/Users/Vinícius/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'

train_folder = dataset_folder+'train/'
test_folder = dataset_folder+'val/'

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
    
### Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#%% FUNÇÕES DE APOIO

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

def random_crop(input_image):
    cropped_image = tf.image.random_crop(value = input_image, size = [config.IMG_SIZE, config.IMG_SIZE, 3])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

# Equivalente a random_jitter = tf.function(random.jitter)
@tf.function()
def random_jitter(input_image):
    # resizing to 286 x 286 x 3
    if config.IMG_SIZE == 256:
        input_image = resize(input_image, 286, 286)
    elif config.IMG_SIZE == 128:
        input_image = resize(input_image, 142, 142)
    
    # randomly cropping to IMGSIZE x IMGSIZE x 3
    input_image = random_crop(input_image)
    
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
    
    return input_image

def load_image_train(image_file):
    input_image = load(image_file)    
    input_image = random_jitter(input_image)
    input_image = normalize(input_image)
    return input_image

def load_image_test(image_file):
    input_image = load(image_file)    
    input_image = resize(input_image, config.IMG_SIZE, config.IMG_SIZE)
    input_image = normalize(input_image)
    return input_image

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

def loss_wgangp_discriminator(disc_real_output, disc_fake_output, discriminator, real_img, fake_img):
    total_disc_loss, real_loss, fake_loss = loss_wgan_discriminator(disc_real_output, disc_fake_output)
    gp = gradient_penalty(discriminator, real_img, fake_img)
    total_disc_loss = total_disc_loss + config.LAMBDA_GP * gp
    return total_disc_loss, real_loss, fake_loss, gp

def gradient_penalty(discriminator, real_img, fake_img):
    ''' 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    '''
    # Get the interpolated image
    alpha = tf.random.normal([config.BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
    diff = fake_img - real_img
    interpolated = real_img + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True) # O discriminador usa duas imagens como entrada

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


#%% FUNÇÕES DO TREINAMENTO

'''
FUNÇÕES DE TREINAMENTO PARA GERADOR ÚNICO
'''

@tf.function
def train_step_direct(generator, discriminator, real_image, input_vector):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Gera a imagem e faz as discriminações
        fake_image = generator(input_vector)
        disc_real = discriminator(real_image, training=True)
        disc_fake = discriminator(fake_image, training=True)

        if config.loss_type == 'patchganloss':
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


# Função para o gerador
def fit_direct(generator, discriminator, train_ds, first_epoch, epochs):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])
    
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
    for epoch in range(first_epoch, epochs+1):
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

            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            # Log as métricas no wandb 
            wandb.log({ 'gen_loss': gen_loss.numpy(), 'disc_loss': disc_loss.numpy(), 'disc_real_loss': disc_real_loss.numpy(),
                        'disc_fake_loss': disc_fake_loss.numpy(), 'gradient_penalty': gp.numpy(), 'accuracy': acc})            
            
        # Salva o checkpoint
        if (epoch) % config.CHECKPOINT_EPOCHS == 0:
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
            
        # Salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        if not config.QUIET_PLOT:
            utils.plot_losses(loss_df)
        
        dt = time.time() - t_start
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
        

#%% TESTA O CÓDIGO E MOSTRA UMA IMAGEM DO DATASET

inp = load(train_folder+'/female/000085.jpg')
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
else:
    raise utils.GeneratorError(config.gen_model)

# CRIANDO O MODELO DE DISCRIMINADOR
if config.disc_model == 'dcgan':
    discriminator = net.dcgan_discriminator(config.IMG_SIZE, constrained = constrained, use_logits = config.DISCRIMINATOR_USE_LOGITS)
elif config.disc_model == 'patchgan':
    discriminator = net.patchgan_discriminator(config.IMG_SIZE, constrained = constrained, use_logits = config.DISCRIMINATOR_USE_LOGITS)
elif config.disc_model == 'progan_adapted':
    discriminator = net.progan_adapted_discriminator(config.IMG_SIZE, constrained = constrained, use_logits = config.DISCRIMINATOR_USE_LOGITS)
else:
    raise utils.DiscriminatorError(config.disc_model)

# Define os otimizadores
generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_G, beta_1=config.ADAM_BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_D, beta_1=config.ADAM_BETA_1)

#%% EXECUÇÃO

# Prepara os inputs
train_dataset = tf.data.Dataset.list_files(dataset_folder+'*/*/*.jpg') # Pega o dataset inteiro para treino
# train_dataset = tf.data.Dataset.list_files(train_folder+'*/*.jpg') 
config.TRAIN_SIZE = len(list(train_dataset))
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
train_dataset = train_dataset.batch(config.BATCH_SIZE)
print("O dataset de treino tem {} imagens".format(config.TRAIN_SIZE))

# Prepara o checkpoint 
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                discriminator_optimizer = discriminator_optimizer,
                                generator = generator,
                                disc = discriminator)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep = config.KEEP_CHECKPOINTS)

# Se for o caso, recupera o checkpoint mais recente
if config.LOAD_CHECKPOINT:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint != None:
        print("Carregando checkpoint mais recente...")
        checkpoint.restore(latest_checkpoint)
        config.FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
    else:
        config.FIRST_EPOCH = 1
        
# Salva os modelos (principalmente para visualização)
generator.save(model_folder+'ae_generator.h5')
discriminator.save(model_folder+'ae_discriminator.h5')

#%% TREINAMENTO

if config.FIRST_EPOCH <= config.EPOCHS:
    # Treinamento direto
    if config.training_type == 'direct':
        fit_direct(generator, discriminator, train_dataset, config.FIRST_EPOCH, config.EPOCHS)
    # Treinamento progressivo
    elif config.training_type == 'progressive':
        print("Ainda não desenvolvido")
    else:
        raise utils.TrainingTypeError(config.training_type)


#%% VALIDAÇÃO

# Finaliza o Weights and Biases
wandb.finish()

## Salva os modelos
generator.save(model_folder+'ae_generator.h5')
discriminator.save(model_folder+'ae_discriminator.h5')

## Plota as losses
try: 
    loss_df = pd.read_csv(experiment_folder + "losses.csv")
    fig = utils.plot_losses(loss_df)

    if config.QUIET_PLOT:
        plt.close(fig)
except:
    None


# Salva os hiperparametros utilizados num arquivo txt
'''
f = open(experiment_folder + "parameters.txt","w+")
f.write("EPOCHS = " + str(config.EPOCHS) + "\n")
f.write("BATCH_SIZE = " + str(config.BATCH_SIZE) + "\n")
f.write("BUFFER_SIZE = " + str(config.BUFFER_SIZE) + "\n")
f.write("IMG_SIZE = " + str(config.IMG_SIZE) + "\n")
f.write("VEC_SIZE = " + str(config.VEC_SIZE) + "\n")
f.write("LEARNING_RATE_G = " + str(config.LEARNING_RATE_G) + "\n")
f.write("LEARNING_RATE_D = " + str(config.LEARNING_RATE_D) + "\n")
f.write("ADAM_BETA_1 = " + str(config.ADAM_BETA_1) + "\n")
f.write("LAMBDA_GP = " + str(config.LAMBDA_GP) + "\n")
f.write("\n")
f.write("CHECKPOINT_EPOCHS = " + str(config.CHECKPOINT_EPOCHS) + "\n")
f.write("LOAD_CHECKPOINT = " + str(config.LOAD_CHECKPOINT) + "\n")
f.write("FIRST_EPOCH = " + str(config.FIRST_EPOCH) + "\n")
f.write("NUM_TEST_PRINTS = " + str(config.NUM_TEST_PRINTS) + "\n")
f.write("\n")
f.write("gen_model = " + str(config.gen_model) + "\n")
f.write("disc_model = " + str(config.disc_model) + "\n")
f.write("loss_type = " + str(config.loss_type) + "\n")
f.write("training_type = " + str(config.training_type) + "\n")
f.close()
'''
