import os
import pickle
import string
import scipy.io
from PIL import Image
from tqdm import trange
from helperFunctions import *
#####################################################################

# Load the ground truth data .mat file
# In this script we once iterate over all files, load the image, check if the cropping works
# and then store the actual image paths, bounding boxes and words in separate files
# By doing this, the actual dataloading can work more efficient

mat = scipy.io.loadmat("data/SynthText/SynthText/gt.mat")
wordBB =mat["wordBB"]
wordBB = np.array(wordBB)[0]
imnames =mat["imnames"]
imnames = imnames[0]
txt = mat["txt"]
txt = txt[0]

vocabulary = [char for char in str(string.ascii_lowercase) + str(string.ascii_uppercase) + " "]
vocabulary.append("<S>")
vocabulary.append("<.>")
vocabulary.append("<E>")

images = []
words = []
boxes = []
for i in trange(0 ,len(imnames)):
    path = os.path.join("data/SynthText/SynthText/", str(imnames[i][0]))
    count = 0
    for j in range(0, len(txt[i])):
        for word in txt[i][j].split():
            if inVocab(vocabulary, str(word)) and 2 < len(word) < 10:
                if len(wordBB[i].shape) == 3:
                    x1, x2, x3, x4 = wordBB[i][0, :, count]
                    y1, y2, y3, y4 = wordBB[i][1, :, count]
                elif len(wordBB[i].shape) == 2:
                    x1, x2, x3, x4 = wordBB[i][0, :]
                    y1, y2, y3, y4 = wordBB[i][1, :]
                box = np.array([[[x1, y1]],[[x2, y2]],[[x3, y3]],[[x4, y4]]])
                try:
                    img = Image.open(path)
                    img_cropped = img.crop((x1, y1, x2, y4))
                    width, height = img_cropped.size
                    filename = os.path.basename(path)
                    filename = os.path.join("data/images",  str(count)+".jpg")
                    images.append(path)
                    words.append(word)
                    boxes.append(box)
                except:
                    pass
            count+=1

images_name = "weights/images.pkl"
open_file = open(images_name, "wb")
pickle.dump(images, open_file)
open_file.close()

words_name = "weights/words.pkl"
open_file = open(words_name, "wb")
pickle.dump(words, open_file)
open_file.close()

boxes_name = "weights/boxes.pkl"
open_file = open(boxes_name, "wb")
pickle.dump(boxes, open_file)
open_file.close()