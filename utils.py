# FUNÇÕES DE APOIO PARA O AUTOENCODER
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from datetime import timedelta

from sklearn.metrics import accuracy_score as accuracy

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
    
def print_generated_images(generator, input, save_destination = None, filename = None):
    
    # Gera um batch de imagens
    gen_img_batch = generator(input, training=True)
    f = plt.figure(figsize=(15,15))

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

def evaluate_accuracy(generator, discriminator, real_image, input_vector, y_real, y_pred, window = 100):
    
    # Cria a imagem fake
    fake_image = generator(input_vector, training = True)

    # Avalia ambas
    disc_real = discriminator(real_image, training = True)
    disc_fake = discriminator(fake_image, training = True)

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
        print("IMG_SIZE " + img_size + " não está disponível")

class TransferUpsampleError(Exception):
    def __init__(self, upsample):
        print("Tipo de upsampling " + upsample + " não definido")