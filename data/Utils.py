import codecs
from Constants import *
import torch
import random
import string
from torch.nn.init import *

class SampleCache(object):

    def __init__(self, size=100):
        super(SampleCache, self).__init__()
        self.cache={}
        self.size=size

    def put(self,ids,samples):
        ids=ids.cpu()
        samples=samples.cpu()
        for i in range(len(ids)):
            id=ids[i].item()
            if id not in self.cache:
                self.cache[id]=[]
            for sent in samples:
                if len(self.cache[id])>=self.size:
                    j=random.randint(0,self.size+1)
                    if j<self.size:
                        self.cache[id][j]=sent
                else:
                    self.cache[id].append(sent)

    def get(self,ids):
        sents=[]
        for i in range(len(ids)):
            id=ids[i].item()
            j = random.randint(0, len(self.cache[id]) - 1)
            sents.append(self.cache[id][j].unsqueeze(0))
        sents=torch.cat(sents,dim=0)
        return sents



def disorder_sampling(batch_instance):
    batch_size = len(batch_instance)
    results = []
    for i in range(batch_size):
        n = random.randint(0, batch_size - 1)
        while n == i:
            n = random.randint(0, batch_size - 1)
        results.append(batch_instance[n].unsqueeze(0))
    return torch.cat(results,dim=0)

def replace_sampling(batch_instance,replaces, max=5):
    batch_size = len(batch_instance)
    vocab_len=len(replaces)
    results = []
    for i in range(batch_size):
        this_sent=batch_instance[i].copy()
        sen_len=len(this_sent)
        n=min(random.randint(1, sen_len),max)
        for j in range(n):
            p = random.randint(0, sen_len-1)
            v = random.randint(0, vocab_len - 1)
            this_sent[p] = replaces[v]
        results.append(this_sent)
    return torch.cat(results,dim=0)

def build_words_vocab_map(words,vocab2id):
    src_map1 = torch.zeros((len(words), len(vocab2id)), requires_grad=False)
    for j in range(len(words)):
        src_map1[j, vocab2id[words[j]]] = 1
    return src_map1

def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    b_map_ = torch.zeros(batch_size, b_len, max)
    if torch.cuda.is_available():
        b_map_ = b_map_.cuda()
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    return b_map_

def build_vocab_vocab_map(id2vocab1, vocab2id2):
    src_map1 = torch.zeros((len(id2vocab1), len(vocab2id2)), requires_grad=False)
    vocab_overlap = torch.ones(len(id2vocab1), requires_grad=False)
    for id in id2vocab1:
        if id2vocab1[id] in vocab2id2:
            src_map1[id, vocab2id2[id2vocab1[id]]]=1
            vocab_overlap[id] = 0

    return src_map1, vocab_overlap

def build_vocab(words, max=100000):
    dyn_vocab2id = dict({PAD_WORD:0})
    dyn_id2vocab = dict({0:PAD_WORD})
    for w in words:
        if w not in dyn_vocab2id and len(dyn_id2vocab)<max:
            dyn_vocab2id[w] = len(dyn_vocab2id)
            dyn_id2vocab[len(dyn_id2vocab)] = w
    return dyn_vocab2id,dyn_id2vocab

def merge1D(sequences,max_len=None,pad_value=None):
    lengths = [len(seq) for seq in sequences]
    max_len=max(lengths) if max_len is None else max_len
    if pad_value is None:
        padded_seqs = torch.zeros(len(sequences), max_len,requires_grad=False).type_as(sequences[0])
    else:
        padded_seqs = torch.full((len(sequences), max_len),pad_value,requires_grad=False).type_as(sequences[0])

    for i, seq in enumerate(sequences):
        end = min(lengths[i], max_len)
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs

def merge2D(sequences,max_len1=None,max_len2=None):
    lengths1 = [seq.size(0) for seq in sequences]
    lengths2 = [seq.size(1) for seq in sequences]
    max_len1 = max(lengths1) if max_len1 is None else max_len1
    max_len2 = max(lengths2) if max_len2 is None else max_len2
    padded_seqs = torch.zeros((len(sequences), max_len1, max_len2),requires_grad=False).type_as(sequences[0])
    for i, seq in enumerate(sequences):
        end1 = lengths1[i]
        end2 = lengths2[i]
        padded_seqs[i, :end1, :end2] = seq[:end1, :end2]
    return padded_seqs

def get_data(i,data):
    ones=dict()
    for key, value in data.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                ones[key] = value[i].unsqueeze(0)
            elif isinstance(value, dict):
                ones[key] = value[data['id'][i].item()]
            else:
                ones[key] = [value[i]]
        else:
            ones[key] =None
    return ones

def concat_data(datalist):
    data=dict()

    size=len(datalist)

    for key in datalist[0]:
        value=datalist[0][key]
        if value is not None:
            if isinstance(value, torch.Tensor):
                data[key]=torch.cat([datalist[i][key] for i in range(size)], dim=0)
            elif isinstance(value, dict):
                data[key] =dict()
                for i in range(size):
                    data[key][datalist[i]['id'].item()]=datalist[i][key]
            else:
                data[key] =[datalist[i][key] for i in range(size)]
        else:
            data[key]=None
    return data

def load_vocab(vocab_file,t=0):
    thisvocab2id = dict({PAD_WORD:0, BOS_WORD:1, UNK_WORD:2, EOS_WORD:3, SEP_WORD:4, CLS_WORD:5, MASK_WORD:6})
    thisid2vocab = dict({0:PAD_WORD, 1:BOS_WORD, 2:UNK_WORD, 3:EOS_WORD, 4:SEP_WORD, 5:CLS_WORD, 6:MASK_WORD})
    id2freq = {}

    sum_freq = 0
    with codecs.open(vocab_file, encoding='utf-8') as f:
        for line in f:
            try:
                name,freq = line.strip('\n').strip('\r').split('\t')
            except Exception:
                continue
            if int(freq)>=t:
                id=len(thisvocab2id)
                thisvocab2id[name] = id
                thisid2vocab[id] = name
                id2freq[id]=int(freq)
                sum_freq+=int(freq)
    id2freq[0] = sum_freq/len(id2freq)
    id2freq[1] = id2freq[0]
    id2freq[2] = id2freq[0]
    id2freq[3] = id2freq[0]

    print('item size: ', len(thisvocab2id))

    return thisvocab2id, thisid2vocab, id2freq


def load_embedding(src_vocab2id, file):
    model=dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            model[word] = torch.tensor([float(val) for val in splitLine[1:]])
    matrix = torch.zeros((len(src_vocab2id), 100))
    xavier_uniform_(matrix)
    for word in model:
        if word in src_vocab2id:
            matrix[src_vocab2id[word]]=model[word]
    return matrix