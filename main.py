#!/usr/bin/python3

## Main code for Progressive GANs
## Created for the Master's degree dissertation
## Vinícius Trevisan 2020-2022

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
import utils, losses

#%% Weights & Biases

# wandb.init(project='unsupervised_gans', entity='vinyluis', mode="disabled")
wandb.init(project='unsupervised_gans', entity='vinyluis', mode="online")

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
config.SAVE_CHECKPOINT = False
config.LOAD_CHECKPOINT = False
config.CHECKPOINT_EPOCHS = 1
config.KEEP_CHECKPOINTS = 1
config.SAVE_MODELS = True

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = '05J'

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
    dataset_folder = '../../0_Datasets/celeba_hq/'

    train_folder = dataset_folder + 'train/'
    test_folder = dataset_folder + 'val/'

    full_dataset_string = dataset_folder + '*/*/*.jpg'
    train_dataset_string = train_folder + '*/*.jpg'
    test_dataset_string = test_folder + '*/*.jpg'
    dataset_string = full_dataset_string

if config.DATASET == 'InsetosFlickr':
    dataset_folder = "../../0_Datasets/flickr_internetarchivebookimages_prepared/"
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
    
if not os.path.exists(model_folder) and config.SAVE_MODELS:
    os.mkdir(model_folder)

#%% CONTROLE DO CHECKPOINT

# Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Definição de época e step para controle no ckeckpoint
tf_epoch = tf.Variable(0)
tf_step = tf.Variable(0)

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
            gen_loss = losses.loss_patchgan_generator(disc_fake, config.DISCRIMINATOR_USE_LOGITS)
            disc_loss, disc_real_loss, disc_fake_loss = losses.loss_patchgan_discriminator(disc_real, disc_fake, config.DISCRIMINATOR_USE_LOGITS)
            gp = 0

        elif config.loss_type == 'wgan':
            gen_loss = losses.loss_wgan_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss = losses.loss_wgan_discriminator(disc_real, disc_fake)
            gp = 0

        elif config.loss_type == 'wgan-gp':
            gen_loss = losses.loss_wgangp_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss, gp = losses.loss_wgangp_discriminator(disc_real, disc_fake, discriminator, real_image, fake_image, config.LAMBDA_GP)

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

def train_step_progressive(generator, discriminator, real_image, input_vector):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Gera a imagem e faz as discriminações
        fake_image = generator(input_vector)
        disc_real = discriminator(real_image)
        disc_fake = discriminator(fake_image)

        if config.loss_type == 'ganloss':
            gen_loss = losses.loss_patchgan_generator(disc_fake, config.DISCRIMINATOR_USE_LOGITS)
            disc_loss, disc_real_loss, disc_fake_loss = losses.loss_patchgan_discriminator(disc_real, disc_fake, config.DISCRIMINATOR_USE_LOGITS)
            gp = 0

        elif config.loss_type == 'wgan':
            gen_loss = losses.loss_wgan_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss = losses.loss_wgan_discriminator(disc_real, disc_fake)
            gp = 0

        elif config.loss_type == 'wgan-gp':
            gen_loss = losses.loss_wgangp_generator(disc_fake)
            disc_loss, disc_real_loss, disc_fake_loss, gp = losses.loss_wgangp_discriminator(disc_real,
                    disc_fake, discriminator, real_image, fake_image, config.LAMBDA_GP, 'progressive')

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
        fig = utils.print_generated_images_prog(generator, fixed_noise, result_folder, filename)

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
        print("\n", utils.get_time_string(), f" - Passo {step}: Imagem {curr_img_size}x{curr_img_size}")

        # Salva o step no gerador e no discriminador
        generator.step = step
        discriminator.step = step

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

                # Salva o alpha no gerador e no discriminador
                generator.alpha = alpha
                discriminator.alpha = alpha

                # Gera o vetor de ruído aleatório
                noise = tf.random.normal(shape = [real_image.shape[0], config.VEC_SIZE])

                # Step de treinamento
                gen_loss, disc_loss, disc_real_loss, disc_fake_loss, gp = train_step_progressive(generator, discriminator, real_image, noise)

                # Calcula a acurácia:
                if config.EVALUATE_ACCURACY:
                    y_real, y_pred, acc = utils.evaluate_accuracy(generator, discriminator, real_image, noise, y_real, y_pred, training = 'progressive')
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
            fig = utils.print_generated_images_prog(generator, fixed_noise, result_folder, filename)
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
                print("Salvando checkpoint...\n")
                ckpt_manager.save()

        # Atualiza o tracker de step
        tf_step.assign_add(1)

        # Reseta o tracker de época
        tf_epoch.assign(0)
        first_epoch = 0

        # Salva o checkpoint
        if config.SAVE_CHECKPOINT and ((epoch) % config.CHECKPOINT_EPOCHS == 0):
            print("Salvando checkpoint...\n")
            ckpt_manager.save()


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


#%% FINAL

# Finaliza o Weights and Biases
wandb.finish()

## Salva os modelos
if config.SAVE_MODELS:
    print("Salvando modelos...\n")
    if config.training_type == 'direct':
        generator.save(model_folder+'generator.h5')
        discriminator.save(model_folder+'discriminator.h5')
    elif config.training_type == 'progressive':
        generator.save_weights(model_folder+'generator.tf')
        discriminator.save_weights(model_folder+'discriminator.tf')
