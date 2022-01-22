### Imports
import os
import pandas as pd
from matplotlib import pyplot as plt
from math import log2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks as net
import utils

#%% HIPERPARÂMETROS E CONFIGURAÇÕES

CHANNELS = 256 
VEC_SIZE = 256
DISENTANGLEMENT = False
LEARNING_RATE_G = 1e-3
LEARNING_RATE_D = 1e-3
ADAM_BETA_1 = 0
NUM_TESTS = 100
IMG_SIZE = 128
STEPS = int(log2(IMG_SIZE / 4))

# Pasta do experimento
experiment_folder = "Experimentos/EXP05J_gen_progan_disc_progan_training_progressive/"

# Cria a pasta de resultado
validation_folder = experiment_folder + "validation/"

if not os.path.exists(validation_folder):
    os.mkdir(validation_folder)

#%% EXECUÇÃO

# CRIANDO O MODELO DE GERADOR
generator = net.progan_generator(CHANNELS, 3, VEC_SIZE, DISENTANGLEMENT)
generator.load_weights(experiment_folder + "model/generator.tf")
generator.step = STEPS

# Cria as imagens
noises = []
for i in range(NUM_TESTS):

    # Gera o vetor de ruído aleatório
    # noise = tf.random.normal(shape = [1, VEC_SIZE])

    # Mostra o andamento do treinamento com uma imagem sintética do fixed noise
    filename = "Teste_" + str(i).zfill(len(str(i))) + ".jpg"
    
    # Cria a imagem aleatória
    fig, vec = utils.random_print_prog(generator, VEC_SIZE, validation_folder, filename)
    plt.close(fig)

    # Transforma o vetor
    vec = vec[0].numpy()

    # Salva o vetor na lista
    noises.append(vec)


# Gera e salva um dataframe com os vetores
df = pd.DataFrame(noises)
df.to_csv(validation_folder + "vetores.csv")