

# ##verifying indexing is correct
# issig = np.where( testsig[1,] == (1000) )[0][0]
# motiflen = 5
# testseq [ 1 , range(issig,issig+motiflen) ]


#let's try a basic algorithm to predict the signal

###
CROP_SIZE = 192
BATCH_SIZE = 1024
###

DNA_onehot_dict = {'A': np.array([1, 0, 0, 0]),
                   'C': np.array([0, 1, 0, 0]),
                   'G': np.array([0, 0, 1, 0]),
                   'T': np.array([0, 0, 0, 1]),
                   'N': np.array([0, 0, 0, 0])
                   }

sequences = []


with open('/Users/harnett/Dropbox/ml_promoters/tssreg_seqs.txt') as f:
    for line in f:
        pdb.set_trace()
        bases = list(line.rstrip().upper())
        bases = [DNA_onehot_dict[base] for base in bases]
        bases = np.vstack(bases)
        bases = np.reshape(bases, (1, bases.shape[0], bases.shape[1]))
        sequences.append(bases)

sequences = np.vstack(sequences).astype(np.float32)

print('Got sequences.')



CAGEs = []
with open('/Users/harnett/Dropbox/ml_promoters/tssreg_cagemat_68h.txt') as f:
    for line in f:
        CAGE = [int(element) for element in line.rstrip().split(' ')]
        CAGE = np.array(CAGE, dtype=np.float64)
        # Let's normalize the number of observations per sample
        # I'm normalizing the sum to 1024 instead of 1
        # to avoid numbers getting really tiny and running into
        # float rounding problems
        # CAGE /= (np.sum(CAGE) / 1024.0)
        CAGEs.append(CAGE)

CAGEs = np.vstack(CAGEs)

print('Got CAGEs.')

# Now we do mean/stdev normalization
# across the whole dataset
CAGE_mean = np.mean(CAGEs)
CAGEs -= CAGE_mean
CAGE_stdev = np.std(CAGEs)
CAGEs /= CAGE_stdev

CAGEs = CAGEs.astype(np.float32)


# Now we define a generator that makes random crops for us

def training_batch_generator(sequences, CAGEs, batch_size, seq_len):
    while True:
        # Randomly shuffle order but retain element pairings
        p = np.random.permutation(sequences.shape[0])
        sequences = sequences[p]
        CAGEs = CAGEs[p]
        for i in range(0, sequences.shape[0], batch_size):
            crop_start = random.randint(0, CAGEs.shape[1] - seq_len)
            crop_end = crop_start + seq_len
            sequence_batch = sequences[i:i + batch_size, crop_start:crop_end, :]
            CAGE_batch = CAGEs[i:i + batch_size, crop_start:crop_end]
            CAGE_batch = np.reshape(CAGE_batch, (-1, seq_len, 1))
            yield (sequence_batch, CAGE_batch)

xmodel = Sequential()
for i in range(5):
#so now we change from a (say)300 bp long vectors of 4 colors
#to ~300bp long vectors of 64 colors, with each of those colors based on 12 positions in the layer below
    model.add(Convolution1D(64, 12, border_mode='same', input_shape=(CROP_SIZE, 4)))
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

model.fit_generator(training_batch_generator(sequences, CAGEs, BATCH_SIZE, 192),
                    samples_per_epoch=CAGEs.shape[0],
                    nb_epoch=20, verbose=1)