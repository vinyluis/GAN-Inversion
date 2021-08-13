# FUNÇÕES DE APOIO PARA O AUTOENCODER
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from datetime import timedelta

from sklearn.metrics import accuracy_score as accuracy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Prepara a string de data e hora conforme necessário
def get_time_string(mode = "complete", days_offset = 0):
    
    #horário atual
    now = datetime.now()
    now = now + timedelta(days = days_offset) # adiciona um offset de x dias
    yr = str(now.year)
    mt = str(now.month)
    dy = str(now.day)
    
    if(len(mt)==1):
        mt = "0" + mt
    
    if(len(dy)==1):
        dy = "0" + dy
    
    m = str(now.minute)
    h = str(now.hour)
    s = str(now.second)
    
    if(len(m)==1):
        m = "0" + m
    if(len(h)==1):
        h = "0" + h
    if(len(s)==1):
        s = "0" + s
    
    if(mode == "complete"):
        st = dy + "-" + mt + "-" + yr + " " + h + ":" + m + ":" + s
        return st
    
    if(mode == "normal"):
        st = dy + "-" + mt + "-" + yr
        return st
    
    if(mode == "file"):
        st = yr+mt+dy
        return st
    
def print_generated_images(generator, inp, save_destination = None, filename = None):
    
    # Gera um batch de imagens
    gen_img_batch = generator(inp, training=True)
    f = plt.figure(figsize=(10,10))

    # Define quantas imagens serão plotadas no máximo
    max_plots = 16
    num_plots = min(len(gen_img_batch), max_plots)
    
    # Define o título
    title = ['Imagens Sintéticas']
    plt.title(title)

    for i in range(num_plots):
        # Acessa o subplot
        plt.subplot(4, 4, i+1)
        # Transforma imagem em [-1, 1] e plota
        plt.imshow(gen_img_batch[i] * 0.5 + 0.5)
        plt.axis('off')
    f.show()

    if save_destination != None and filename != None:
        f.savefig(save_destination + filename)

    return f

def print_generated_images_prog(generator, inp, alpha, step, save_destination = None, filename = None):
    
    # Gera um batch de imagens
    gen_img_batch = generator(inp, alpha, step)
    f = plt.figure(figsize=(10, 10))

    # Define quantas imagens serão plotadas no máximo
    max_plots = 16
    num_plots = min(len(gen_img_batch), max_plots)
    
    # Define o título
    title = ['Imagens Sintéticas']
    plt.title(title)

    for i in range(num_plots):
        # Acessa o subplot
        plt.subplot(4, 4, i+1)
        # Transforma imagem em [-1, 1] e plota
        plt.imshow(gen_img_batch[i] * 0.5 + 0.5)
        plt.title(i)
        plt.axis('off')
    f.show()

    if save_destination != None and filename != None:
        f.savefig(save_destination + filename)

    return f

def plot_losses(loss_df, plot_ma = True, window = 100):
    
    # Plota o principal
    f = plt.figure()
    sns.lineplot(x = range(loss_df.shape[0]), y = loss_df["Loss G"])
    sns.lineplot(x = range(loss_df.shape[0]), y = loss_df["Loss D"])
    
    # Plota as médias móveis
    if plot_ma:
        
        lossG_ma = loss_df["Loss G"].rolling(window = window, min_periods = 1).mean()
        lossD_ma = loss_df["Loss D"].rolling(window = window, min_periods = 1).mean()
        sns.lineplot(x = range(loss_df.shape[0]), y = lossG_ma)
        sns.lineplot(x = range(loss_df.shape[0]), y = lossD_ma)
        plt.legend(["Loss G", "Loss D", "Loss G - MA", "Loss D - MA"])
    else:
        plt.legend(["Loss G", "Loss D"])
    
    f.show()
    
    return f

def evaluate_accuracy(generator, discriminator, real_image, input_vector, y_real, y_pred, window = 100, training = 'direct', alpha = 1, step = 0):
    
    if training == 'direct':
        # Cria a imagem fake
        fake_image = generator(input_vector, training = True)

        # Avalia ambas
        disc_real = discriminator(real_image, training = True)
        disc_fake = discriminator(fake_image, training = True)

    elif training == 'progressive':
        # Cria a imagem fake
        fake_image = generator(input_vector, alpha, step)

        # Avalia ambas
        disc_real = discriminator(real_image, alpha, step)
        disc_fake = discriminator(fake_image, alpha, step)
        
    # Para o caso de ser um discriminador PatchGAN, tira a média
    disc_real = np.mean(disc_real)
    disc_fake = np.mean(disc_fake)

    # Aplica o threshold
    disc_real = 1 if disc_real > 0.5 else 0
    disc_fake = 1 if disc_fake > 0.5 else 0

    # Acrescenta a observação real como y_real = 1
    y_real.append(1)
    y_pred.append(disc_real)

    # Acrescenta a observação fake como y_real = 0
    y_real.append(0)
    y_pred.append(disc_fake)
    
    # Calcula a acurácia pela janela
    if len(y_real) > window:
        acc = accuracy(y_real[-window:], y_pred[-window:])    
    else:
        acc = accuracy(y_real, y_pred)

    return y_real, y_pred, acc

def print_fixed_noise(noise, save_destination = None, filename = None):
    
    f = plt.figure(figsize=(7,3))

    # Define o título
    title = ['Ruído Fixo']
    plt.imshow(noise, cmap = 'coolwarm')
    plt.colorbar()

    if save_destination != None and filename != None:
        f.savefig(save_destination + filename)

    return f

def random_print_prog(generator, step, vec_size, save_destination = None, filename = None):
    
    # Gera um vetor de ruído
    noise = tf.random.normal(shape = (1, vec_size))

    # Gera um batch de imagens
    gen_img = generator(noise, 1, step)
    f = plt.figure(figsize=(5,5))
    plt.imshow(gen_img[0] * 0.5 + 0.5)
    plt.axis('off')
    f.show()

    if save_destination != None and filename != None:
        f.savefig(save_destination + filename)

    return f



#%% PREPARAÇÃO DO DATASET
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

def random_crop(input_image, height, width):
    cropped_image = tf.image.random_crop(value = input_image, size = [height, width, 3])
    return cropped_image

def normalize(input_image):
    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    return input_image

def random_jitter(input_image, img_size):
    # Resize
    jitter_size = int(1.11 * img_size)
    input_image = resize(input_image, jitter_size, jitter_size)
    # Crop aleatório para IMGSIZE x IMGSIZE x 3
    input_image = random_crop(input_image, img_size, img_size)
    # Random mirroring
    if tf.random.uniform(()) > 0.5: 
        input_image = tf.image.flip_left_right(input_image)
    
    return input_image

def load_image_train(image_file, img_size, use_jitter = True):
    input_image = load(image_file)    
    if use_jitter:
        input_image = random_jitter(input_image, img_size)
    else:
        input_image = resize(input_image, img_size, img_size)
    input_image = normalize(input_image)
    return input_image

def prepare_dataset(files_string, image_size, batch_size, buffer_size = None, use_jitter = True, use_cache = False):
    dataset = tf.data.Dataset.list_files(files_string) # Pega o dataset inteiro para treino
    dataset_size = len(list(dataset))
    dataset = dataset.map(lambda x: load_image_train(x, image_size, use_jitter))
    if use_cache:
        dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size) if buffer_size != None else dataset
    dataset = dataset.batch(batch_size)
    return dataset, dataset_size

#%% TRATAMENTO DE EXCEÇÕES
    
class GeneratorError(Exception):
    def __init__(self, gen_model):
        print("O gerador " + gen_model + " é desconhecido")
    
class DiscriminatorError(Exception):
    def __init__(self, disc_model):
        print("O discriminador " + disc_model + " é desconhecido")
        
class LossError(Exception):
    def __init__(self, loss_type):
        print("A loss " + loss_type + " é desconhecida")

class TrainingTypeError(Exception):
    def __init__(self, training_type):
        print("O treinamento " + training_type + " é desconhecido")
        
class LossCompatibilityError(Exception):
    def __init__(self, loss_type, disc_model):
        print("A loss " + loss_type + " não é compatível com o discriminador " + disc_model)

class SizeCompatibilityError(Exception):
    def __init__(self, img_size):
        print("IMG_SIZE " , img_size , " não está disponível")

class TransferUpsampleError(Exception):
    def __init__(self, upsample):
        print("Tipo de upsampling " + upsample + " não definido")

class TrainingCompatibilityError(Exception):
    def __init__(self, training, generator, discriminator):
        print(f"A combinação training = {training}, generator = {generator} e discriminator = {discriminator} não é possível")