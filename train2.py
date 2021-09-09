import argparse
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from dataloader import *
from model2 import *
from tqdm import trange
torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import f1_score
import yaml
from helperFunctions import *

####################################################################################

def testLoop(dataloader_Test,  model, criterion, writer, counter_test, trainer, testSize):
    print(" Test...")
    model.eval()
    errors = []
    scoresReal = []
    iterator = iter(dataloader_Test)
    for index in trange(0, testSize):
        img, word = iterator.next()
        predictions = model(img, word)
        # Only measure accuracy without the padded tokens
        # Otherwise, the accuracy would be falsely high, since the model fastly learns
        # to add the <.> padding tokens
        predictions, stops = removePaddingChars(word, predictions)
        error = criterion(predictions, word)
        errors.append(float(error))
        predictions = torch.argmax(predictions, dim=1)
        for g in range(0, len(predictions)):
            scReal = f1_score(predictions[g][:stops[g]], word[g][:stops[g]], average="micro", labels=np.array(range(0, 56)))
            scoresReal.append(scReal)
    f1s_real = np.mean(np.array(scoresReal))
    # Logg losses to tensorboard
    if trainer:
        writer.add_scalar('CEL/train/', float(np.mean(np.array(errors))), counter_test)
        writer.add_scalar('F1/train/', f1s_real, counter_test)
    else:
        writer.add_scalar('CEL/test/', float(np.mean(np.array(errors))), counter_test)
        writer.add_scalar('F1/test/', f1s_real, counter_test)
    model.train()

####################################################################################

def trainLoop(dataloader_Train,  dataloader_Test, model, epochs, lr, enc, dec):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    counter_train = 0
    counter_test = 0
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    testLoop(dataloader_Test, model, criterion, writer, counter_test, trainer=False,testSize=len(dataloader_Test))
    testLoop(dataloader_Train, model, criterion, writer, counter_train, trainer=True,testSize=len(dataloader_Test))
    counter_train += 1
    counter_test += 1

    model.train()
    losses = []
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        iterator = iter(dataloader_Train)
        size = len(dataloader_Train)
        for index in trange(0, size):
            model.zero_grad()
            img, word = iterator.next()
            predictions = model(img, word)
            loss = criterion(predictions.float(), word.long())
            losses.append(float(loss))
            loss.backward(retain_graph=True)
            optimizer.step()
            if index % 500 == 0:
                testLoop(dataloader_Test, model, criterion, writer, counter_test, trainer = False, testSize = len(dataloader_Test))
                testLoop(dataloader_Train, model, criterion, writer, counter_train, trainer = True, testSize=len(dataloader_Test))
                counter_train+=1
                counter_test+=1
                torch.save(enc, "weights/enc.pth")
                torch.save(dec, "weights/dec.pth")
                torch.save(model, "weights/seq.pth")

####################################################################################

def main(args):

    # Load configurations
    modelType = args.modelType
    config = "config2.yml"
    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
    batch = int(cfg["batch"])
    epochs= int(cfg["epochs"])
    embeddingDimEncoder= int(cfg["embeddingDimEncoder"])
    embeddingDimDecoder = int(cfg["embeddingDimDecoder"])
    hiddendDimDecoder = int(cfg["hiddendDimDecoder"])
    hiddendDimEncoder = int(cfg["hiddendDimEncoder"])
    n_layersEncoder = int(cfg["n_layersEncoder"])
    n_layersDecoder = int(cfg["n_layersDecoder"])
    dropout_Encoder = float(cfg["dropout_Encoder"])
    dropout_Decoder = float(cfg["dropout_Decoder"])
    lr = float(cfg["lr"])

####################################################################################

    # Load dataset and dataloader
    vocabulary, dataloader_Train, dataloader_Test, dataloader_TEval = createData(batch)
    print("Train Data Size:   ",  len(dataloader_Train.dataset), "Test Data Size:   ", len(dataloader_Test.dataset))

####################################################################################

    # Whether load pretrained models or initialize new ones
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

    # Perform actual training
    trainLoop(dataloader_Train, dataloader_Test,  model, epochs, lr, enc, dec)

####################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--modelType', type=str, default='NEW', help = "NEW or TRAINED")
    args = parser.parse_args()
    main(args)