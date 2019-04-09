import json
#import nltk
#from nltk import word_tokenize
import re
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

#start from 1, because padding will pad zero into it.

DIC_word_index[BOS] = 1
DIC_word_index[EOS] = 2
DIC_word_index[PAD] = 3
DIC_word_index[UNK] = 4

#======Read json file, tokenize and add it to DIC
index = 5
rawlabels = json.load(open(labelfilename, 'r'))
for data in rawlabels:
    for caption in data['caption']:
        caption = re.sub(r'[^\w\s\<\>\-]','',caption)
        tokens = caption.lower().split()
        for word in tokens:
            if word not in DIC_word_index:
                DIC_word_index[word] = index
                index += 1
                
#====== key, value inversion
DIC_index_word = dict((v,k) for k,v in DIC_word_index.items())

with open("DIC_word_index.json", 'w') as f:
    json.dump(DIC_word_index, f)
    
with open("DIC_index_word.json", 'w') as f:
    json.dump(DIC_index_word, f)