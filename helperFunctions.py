import numpy as np
import torch

################################################

# Check if word is in the given vocabulary
def inVocab(vocabulary, word):
    for char in word:
        if char not in vocabulary:
            return False
    return True

################################################

# Convert word to array of char-indices, for given vocabulary
def toID(vocabulary, word):
    vec = []
    for char in word:
        vec.append(vocabulary.index(char))
    # EOS token
    vec.append(55)
    return np.array(vec)

################################################

# Convert vector to list of characters
def vecToChars(vocabulary, vec):
    chars = [vocabulary[index] for index in vec]
    return chars

################################################

# This methods makes each word have the same lenght.
# By adding a <.> token to the end, such that the model can be trained using batches
def adjustWordLenght(word, max = 11):
    while len(word)<max:
        # <.> Token to make all words have the same lenght
        # To allow batches
        word = np.append(word, 54)
    return word

################################################

# This methods removes the added <.> tokens at the end
# Only for the nicer printing of the predictions
def removePaddingChars(word, predictions):
    change = torch.zeros_like(predictions[0, :, 0])
    change[-2] = 1.0
    stops = []
    for i in range(0, len(word)):
        stop = 0
        for j in range(0, len(word[0])):
            if word[i][j] == 54:
                predictions[i, :, j] = change
            else:
                stop += 1
        stops.append(stop)
    return predictions, stops