import pickle
import random
import string
from torch.utils.data import DataLoader
from PIL import Image
from helperFunctions import *

################################################

class DataSet():
    def __init__(self, vocabulary, batch, images, words, boxes):
        self.vocabulary = vocabulary
        self.batch = batch
        self.images = images
        self.words = words
        self.boxes = boxes

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        box = self.boxes[index]
        x1 = box[0][0][0]
        x2 = box[1][0][0]
        y1 = box[0][0][1]
        y4 = box[3][0][1]
        img_cropped = img.crop((x1, y1, x2, y4))
        img_resized = img_cropped.resize((89, 26))
        img_resized = np.array(img_resized)
        img_resized = img_resized.reshape((3, 26, 89))
        word = self.words[index]
        word = toID(self.vocabulary, word)
        word = adjustWordLenght(word)
        return  img_resized, word

    def __len__(self):
        return len(self.words)

################################################

def createData(BATCH):
    vocabulary = [char for char in str(string.ascii_lowercase) + str(string.ascii_uppercase) + " "]
    vocabulary.append("<S>")
    vocabulary.append("<.>")
    vocabulary.append("<E>")

    open_file = open("weights/images.pkl", "rb")
    images = pickle.load(open_file)
    open_file.close()

    open_file = open("weights/words.pkl", "rb")
    words = pickle.load(open_file)
    open_file.close()

    open_file = open("weights/boxes.pkl", "rb")
    boxes = pickle.load(open_file)
    open_file.close()

    c = list(zip(images, words, boxes))
    random.shuffle(c)
    images, words, boxes = zip(*c)

    separation = int(len(images) * 0.9)

    images_train = images[:separation]
    boxes_train = boxes[:separation]
    words_train = words[:separation]

    images_rest = images[separation:]
    boxes_rest = boxes[separation:]
    words_rest = words[separation:]

    separation = int(len(images_rest) * 0.5)

    images_test = images_rest[:separation]
    boxes_test = boxes_rest[:separation]
    words_test = words_rest[:separation]

    images_eval = images_rest[separation:]
    boxes_eval = boxes_rest[separation:]
    words_eval = words_rest[separation:]

    dataset_Train = DataSet(vocabulary, BATCH, images_train, words_train, boxes_train)
    dataloader_Train = DataLoader(dataset=dataset_Train, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)

    dataset_Test = DataSet(vocabulary, BATCH, images_test, words_test, boxes_test)
    dataloader_Test = DataLoader(dataset=dataset_Test, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)

    dataset_Eval = DataSet(vocabulary, BATCH, images_eval, words_eval, boxes_eval)
    dataloader_Eval = DataLoader(dataset=dataset_Eval, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)

    return vocabulary, dataloader_Train, dataloader_Test, dataloader_Eval