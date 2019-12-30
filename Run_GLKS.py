from GLKSDataset import *
from GLKS import *
from torch import optim
from trainers.DefaultTrainer import *
import torch.backends.cudnn as cudnn
import argparse
import os

version='oracle'
base_output_path = 'GLKS/holl.'+version+'/'
data_path = 'holl/'
embedding_size = 300
hidden_size = 256
min_window_size = 4
num_windows = 1
knowledge_len = 300
min_vocab_freq=10

def train(args):
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 32

    output_path=data_path+base_output_path

    vocab2id, id2vocab, id2freq = load_vocab(data_path + 'holl_input_output.'+version+'.vocab', t=min_vocab_freq)

    if not os.path.exists(data_path+'glove.6B.300d.txt'+'.dat'):
        prepare_embeddings(data_path+'glove.6B.300d.txt')
    emb_matrix=load_embeddings(data_path+'glove.6B.300d.txt', id2vocab, embedding_size)

    if os.path.exists(data_path + 'holl-train.'+version+'.pkl'):
        train_dataset = torch.load(data_path + 'holl-train.'+version+'.pkl')
    else:
        train_dataset = GLKSDataset([data_path + 'holl-train.'+version+'.json'], vocab2id, min_window_size, num_windows, knowledge_len)

    model = GLKS(min_window_size, num_windows, embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1, emb_matrix=emb_matrix)
    init_params(model, escape='embedding')

    model_optimizer = optim.Adam(model.parameters())

    trainer = DefaultTrainer(model, args.local_rank)

    # for i in range(10):
    #     trainer.train_epoch('ds_train', train_dataset, collate_fn, batch_size, i, model_optimizer)

    for i in range(20):
        if i==5:
            train_embedding(model)
        trainer.train_epoch('ds_mle_mcc_train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        trainer.serialize(i, output_path=output_path)

def test(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 64

    output_path = data_path + base_output_path

    vocab2id, id2vocab, id2freq = load_vocab(data_path + 'holl_input_output.'+version+'.vocab', t=min_vocab_freq)

    if os.path.exists(data_path + 'holl-dev.'+version+'.pkl'):
        dev_dataset = torch.load(data_path + 'holl-dev.'+version+'.pkl')
    else:
        dev_dataset = GLKSDataset([data_path + 'holl-dev.'+version+'.json'], vocab2id, min_window_size, num_windows, knowledge_len)
    if os.path.exists(data_path + 'holl-test.'+version+'.pkl'):
        test_dataset = torch.load(data_path + 'holl-test.'+version+'.pkl')
    else:
        test_dataset = GLKSDataset([data_path + 'holl-test.'+version+'.json'], vocab2id, min_window_size, num_windows, knowledge_len)

    for i in range(20):
        print('epoch', i)
        file = output_path+'model/'+ str(i) + '.pkl'

        if os.path.exists(file):
            model = GLKS(min_window_size, num_windows, embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1)
            model.load_state_dict(torch.load(file))
            trainer = DefaultTrainer(model, None)
            trainer.test('test', dev_dataset, collate_fn, batch_size, i, output_path=output_path)
            trainer.test('test', test_dataset, collate_fn, batch_size, 100 + i, output_path=output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    if args.mode=='test':
        test(args)
    elif args.mode=='train':
        train(args)