import sys
import numpy as np
import yaml
from keras.models import Sequential

# from keras.datasets import mnist
# from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import *
from keras.layers.normalization import *
from keras.layers.advanced_activations import LeakyReLU

###
# NN parameters
CROP_SIZE = 300#width of sequence to look at (originally had to sample in 1d space since output was centered)
BATCH_SIZE = 1024#number of sequences to looks at at once
CONVSIZE=64#how much dimensionality expansion are we going to do?
CONVWIDTH=12#width in 1d sequence to look
NUCNUM=4 #ACGT
TESTFRAC = 0.2
###


# Reading in the seq data
DNA_onehot_dict = {'A': np.array([1, 0, 0, 0]),
                   'C': np.array([0, 1, 0, 0]),
                   'G': np.array([0, 0, 1, 0]),
                   'T': np.array([0, 0, 0, 1]),
                   'N': np.array([0, 0, 0, 0])
                   }
# load config file
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream)

# testdata files from config file
simseqfile = config['data']['testdata1'][0]
simsigfile = config['data']['testdata1'][1]

# initialize sequence data object
sequences = []

with open(config['data']['testdata1'][0]) as f:
    for line in f:
        bases = list(line.rstrip().upper())
        bases = [DNA_onehot_dict[base] for base in bases]
        bases = np.vstack(bases)
        bases = np.reshape(bases, (1, bases.shape[0], bases.shape[1]))
        sequences.append(bases)

sequences = np.vstack(sequences).astype(np.float32)

print('Got sequences.')

CAGEs = np.loadtxt(simsigfile, delimiter=',')
     # Let's normalize the number of observations per sample
#         # I'm normalizing the sum to 1024 instead of 1
#         # to avoid numbers getting really tiny and running into
#         # float rounding problems
for i in range(CAGEs.shape[0]):
    CAGEs[i,] /= (np.sum(CAGEs[i,]) / 1024.0)







print('Got CAGEs.')

# Now we do mean/stdev normalization
# across the whole dataset
CAGE_mean = np.mean(CAGEs)
CAGEs -= CAGE_mean
CAGE_stdev = np.std(CAGEs)
CAGEs /= CAGE_stdev

CAGEs = CAGEs.astype(np.float32)



test_ind = int(sequences.shape[0]*0.2)
testsequences,sequences = sequences[0:test_ind,:,:],sequences[test_ind:,:,:]
testCAGEs,CAGEs = CAGEs[0:test_ind,:],CAGEs[test_ind:,:]


print('data processed')


def training_batch_generator(sequences, CAGEs, batch_size, seq_len):
    # Now we define a generator that makes random crops for us
    # this was inted for the original data which was centered on the TSS
    # probably not needed for the simmed data
    while True:
        # Randomly shuffle order but retain element pairings
        # this yields a random permutation of the range from 0 to X
        # where X is the the number of sequences
        p = np.random.permutation(sequences.shape[0])
        sequences = sequences[p]
        CAGEs = CAGEs[p]
        for i in range(0, sequences.shape[0], batch_size):
            crop_start = np.random.randint(0, CAGEs.shape[1] - seq_len + 1)
            crop_end = crop_start + seq_len
            # crop_start=0
            # crop_end = seq_len
            seq_batch = sequences[i:i + batch_size, crop_start:crop_end, :]
            CAGE_batch = CAGEs[i:i + batch_size, crop_start:crop_end]
            CAGE_batch = np.reshape(CAGE_batch, (-1, seq_len, 1))
            yield (seq_batch, CAGE_batch)

model = Sequential()

for i in range(5):
    # so now we change from a (say)300 bp long vectors of 4 colors
    # to ~300bp long vectors of CONVSIZE colors, with each of those colors
    # based on 12 positions in the layer below
    model.add(Convolution1D(CONVSIZE, CONVWIDTH, border_mode='same', input_shape=(CROP_SIZE, NUCNUM)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout1D(0.2))
model.add(Convolution1D(512, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(SpatialDropout1D(0.1))
model.add(Convolution1D(1, 1, border_mode='same'))

model.compile(loss='mse', optimizer='nadam')
print('Done compiling!')

model.fit_generator(training_batch_generator(sequences, CAGEs, BATCH_SIZE, CROP_SIZE),
                    samples_per_epoch=CAGEs.shape[0],
                    nb_epoch=20, verbose=1)

print('done Done Training!')


posmin = int((300-192)/2)
posmax = int((300-posmin))

prediction = model.predict(testsequences,batch_size=BATCH_SIZE)

assert( CAGEs[:,posmin:posmax].shape == prediction[:,:,0].shape)


print('done Predicting!')







