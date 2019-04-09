import json
import nltk
from nltk import word_tokenize

idfilename = 'training_data/id.txt'
datadirname = 'training_data/feat/'
labelfilename = 'training_label.json'

#======Initialize the dictionary======#
DIC_index_word = {}
DIC_word_index = {}

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"

DIC_word_index[BOS] = 0
DIC_word_index[EOS] = 1
DIC_word_index[PAD] = 2
DIC_word_index[UNK] = 3

#======Read json file, tokenize and add it to DIC
index = 4
rawlabels = json.load(open(labelfilename, 'r'))
for data in rawlabels:
    for caption in data['caption']:
        caption = caption.lower()
        for word in word_tokenize(caption):
            if word not in DIC_word_index:
                DIC_word_index[word] = index
                index += 1
                
#====== key, value inversion
DIC_index_word = dict((v,k) for k,v in DIC_word_index.items())

with open("DIC_word_index.json", 'w') as f:
    json.dump(DIC_word_index, f)
    
with open("DIC_index_word.json", 'w') as f:
    json.dump(DIC_index_word, f)