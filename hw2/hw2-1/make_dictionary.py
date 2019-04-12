import json
import nltk
from nltk import word_tokenize
#import re
idfilename = 'training_data/id.txt'
datadirname = 'training_data/feat/'
labelfilename = 'training_label.json'

#======Initialize the dictionary======#
DIC_index_word = {}
DIC_word_index = {}


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

DIC_word_index[PAD] = 0
DIC_word_index[BOS] = 1
DIC_word_index[EOS] = 2
DIC_word_index[UNK] = 3

#======Read json file, tokenize and add it to DIC
def make_naive_dic():
    index = 4
    rawlabels = json.load(open(labelfilename, 'r'))
    for data in rawlabels:
        for caption in data['caption']:
            tokens = word_tokenize(caption.lower())
            for word in tokens:         
                if word not in DIC_word_index:
                    DIC_word_index[word] = index
                    index += 1
def make_freq_dic():    
    word_freq = {}
    rawlabels = json.load(open(labelfilename, 'r'))
    for data in rawlabels:
        for caption in data['caption']:
            tokens = word_tokenize(caption.lower())
            for word in tokens:
                if word not in word_freq:
                    word_freq[word] = 1
                elif word in word_freq:
                    word_freq[word] += 1
    
    index = 4 
    for word in word_freq:
        if word_freq[word] >= 4:
            DIC_word_index[word] = index
            index +=1
                
    
if __name__ == '__main__':
    #====== choose one 
    #make_naive_dic()
    make_freq_dic()
    
    #====== key, value inversion
    DIC_index_word = dict((v,k) for k,v in DIC_word_index.items())
    
    #====== write to json file
    with open("DIC_word_index.json", 'w') as f:
        json.dump(DIC_word_index, f)
    
    with open("DIC_index_word.json", 'w') as f:
        json.dump(DIC_index_word, f)



