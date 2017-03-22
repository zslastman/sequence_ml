import pdb
import numpy as np
import yaml

from Bio.Alphabet import IUPAC


# -------------------------------------------------------------------------------
# --------Biopython sketch
# -------------------------------------------------------------------------------

# We are going to generate fake training data for an NN, to see if it can learn
# sequence rules

print ("Starting test script")

MOTIFLIST = ['TATAA', 'GAATCC']
BASEPROBS = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
SEQLEN = 300
SEQNUM = 1000

# load config file
with open("config.yaml", 'r') as stream:
    config = yaml.load(stream)

# testdata files from config file
simseqfile = config['data']['testdata1'][0]
simsigfile = config['data']['testdata1'][1]


def generate_sequence(SEQLEN, SEQNUM):
    # get the base probablities iS the right order
    probs = [BASEPROBS[letter] for letter in IUPAC.unambiguous_dna.letters]
    # now sample the iupac letters N times
    seqs = np.random.choice(
        list(IUPAC.unambiguous_dna.letters),
        p=probs,
        size=(SEQNUM, SEQLEN)
    )
    sig = np.zeros(seqs.shape)
    # Now add the Motifs
    for i in range(SEQNUM):
        # choose a random motif
        motif = MOTIFLIST[np.random.randint(len(MOTIFLIST))]
        motlen = len(motif)
        # choose a random position
        pos = np.random.randint(0, SEQLEN + 1 - motlen)
        #
        # modify seq
        seqs[i, pos:(pos + motlen)] = list(motif)
        # and signal
        sig[i, pos] = 1000

    # now concatenate the sequencees, and make the bio objects
    #  seqs = [''.join(seqs[i, :]) for i in range(n)]
    # return
    return seqs, sig


print ("Faking data")

testseq, testsig = generate_sequence(SEQLEN, SEQNUM)

print (testseq[1, 1:10], "\n\n\n")

print (testsig[1, 1:10], "\n\n\n")

print ("Writing fake data")

with open(simseqfile, 'w') as seqfile:
    for seqslice in np.rollaxis(testseq, 0):
        seqfile.write(''.join(seqslice) + '\n')

np.savetxt(simsigfile, testsig, delimiter=",")

print("Exiting data generation script")

# ##verifying indexing is correct
# issig = np.where( testsig[1,] == (1000) )[0][0]
# motiflen = 5
# testseq [1 , range(issig,issig+motiflen)]
