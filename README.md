# OCR

## Preliminary steps to use Code 
1. Download "SynthText in the Wild Dataset" from https://www.robots.ox.ac.uk/~vgg/data/scenetext/
2. Unzip it into the folder structure "data/SynthText/SynthText/..."
3. pip3 install -r requirements.txt
4. Execute the preprocessing script: `python3 preprocess.py` (takes a while)

## Folders

**data/:** Contains the dataset

**runs/:** For saving the pytorch runs, using tensorboard event files 

**weights/:** For saving the model weights 

## Files

**dataloader.py:** Contains code for reading and loading the data

**model1.py :** Contains the first of the two implemented OCR models

**model2.py :** Contains the second of the two implemented OCR models

**train1.py :** Contains the training and evaluation code for the first model

**train2.py :** Contains the training and evaluation code for the second model

**test1.py :** Contains test code for the first model

**test2.py :** Contains test code for the second model

**config1.yml:** Contains the configuration (hyperparameters) for model1

**config2.yml:** Contains the configuration (hyperparameters) for model2

**preprocess.py:** Contains code for preprocessing the data

## Example usages

1. Loading a pretrained model1 and train it: `python3 train1.py -m TRAINED`
2. Create a new model2 and train it: `python3 train1.py`
3. Exectue test script for model2: `python3 test2.py`






