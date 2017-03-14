
# from keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# y_train = y_train.reshape(-1, 1)
# y_transformed = enc.fit_transform(y_train)

# y_transformed.shape

# y_transformed[0:1,:]



# from keras.layers import Dense
# import numpy
# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)


# -------------------------------------------------------------------------------
# --------Biopython sketch
# -------------------------------------------------------------------------------

#We are going to generate fake training data for an NN, to see if it can learn
#sequence rules

print ("Starting test script")


import Bio
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb;


my_seq = Seq("GATC", IUPAC.unambiguous_dna)

import sys

motiflist = ['TATAA','GAATCC']
baseprobs = {"A": 0.3, "C":0.2 , "G":0.2 ,"T": 0.3} 
seqlen = 200
n=100


simseqfile = '/Users/harnett/Desktop/testsequence.txt'
simsigfile = '/Users/harnett/Desktop/testsig.txt'


def generate_sequence(seqlen,n):
    #get the base probablities in the right order
    probs = [ baseprobs[letter] for letter in IUPAC.unambiguous_dna.letters ]
    #now sample the iupac letters N times
    seqs = np.random.choice(list(IUPAC.unambiguous_dna.letters),p = probs , size = (n,seqlen))
    sig = np.zeros(seqs.shape)
    #Now add the Motifs
    for i in range(n):
        #choose a random motif
        motif = motiflist[np.random.randint(len(motiflist))]
        motlen = len(motif)
        #choose a random position
        pos = np.random.randint(0,seqlen+1-motlen)
        #
        #modify seq
        seqs[i,pos:(pos+motlen)] = list(motif)
        #and signal
        sig[i,pos] = 1000

    #now concatenate the sequencees, and make the bio objects
    # seqs = [''.join(seqs[i,:]) for i in range(n) ]
    #return
    return seqs,sig


print ("Faking data")

testseq,testsig = generate_sequence(300,1000)

print (testseq[1,1:10],"\n\n\n")

print (testsig[1,1:10], "\n\n\n")

print ("Writing fake data")

with open(simseqfile,'w') as seqfile:
    for seqslice in np.rollaxis(testseq,0):
        seqfile.write(''.join(seqslice)+'\n')

with open(sigseqfile,'w') as sigfile:
    for sigslice in np.rollaxis(testsig,0):
        sigfile.write(''.join(str(sigslice))+'\n')

print("Exiting data generation script")
