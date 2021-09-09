import argparse
from dataloader import *
from model2 import *
import yaml
from helperFunctions import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

####################################################################################

def main(args):

    # Load configurations
    modelType = args.modelType
    config = "config2.yml"
    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
    batch = int(cfg["batch"])
    embeddingDimEncoder= int(cfg["embeddingDimEncoder"])
    embeddingDimDecoder = int(cfg["embeddingDimDecoder"])
    hiddendDimDecoder = int(cfg["hiddendDimDecoder"])
    hiddendDimEncoder = int(cfg["hiddendDimEncoder"])
    n_layersEncoder = int(cfg["n_layersEncoder"])
    n_layersDecoder = int(cfg["n_layersDecoder"])
    dropout_Encoder = float(cfg["dropout_Encoder"])
    dropout_Decoder = float(cfg["dropout_Decoder"])

####################################################################################

    # Create dataset and dataloader
    vocabulary, dataloader_Train, dataloader_Test, dataloader_TEval = createData(batch)
    print("Train Data Size:   ", len(dataloader_Train.dataset))
    print("Test Data Size:   ", len(dataloader_Test.dataset))
    print("Eval Data Size:   ", len(dataloader_TEval.dataset))
    print("")

####################################################################################

    # Wheter load pretrained models or initialize new ones
    if modelType == "TRAINED":
        try:
            enc = torch.load("weights/enc.pth")
            dec = torch.load("weights/dec.pth")
            model = torch.load("weights/seq.pth")
        except:
            raise ValueError('Model not found!')
    else:
        enc = encoder(embeddingDimEncoder, hiddendDimEncoder, n_layersEncoder, dropout_Encoder)
        dec = decoder_Attention(len(vocabulary), embeddingDimDecoder, hiddendDimDecoder, n_layersDecoder, dropout_Decoder, n_layersEncoder)
        model = seq2seq(enc, dec)

####################################################################################

    # Test Loop
    model.eval()
    iterator = iter(dataloader_Test)
    for index in range(0, len(dataloader_Test)-1):
        img, word = iterator.next()
        predictions = model(img, word)
        size = word.shape[0]
        w = word.clone()
        p = predictions.clone()
        for j in range(0, size):
            word = w[j]
            predictions = p[j]
            trueWord = ""
            trueWord = trueWord.join(vecToChars(vocabulary, word))
            predictedWord = ""
            predictedWord = predictedWord.join(vecToChars(vocabulary, torch.argmax(predictions, dim=0)))
            print("True:  ", trueWord, "   ", "Predicted:  ", predictedWord)
            print("")
            plt.imshow(img[j].reshape((26, 89, 3)))
            plt.show()
            plt.axis("off")

####################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--modelType', type=str, default='NEW', help = "NEW or TRAINED")
    args = parser.parse_args()
    main(args)