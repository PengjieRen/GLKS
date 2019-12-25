from torch.utils.data import DataLoader
from modules.Generations import *
from Eval_Rouge import *
from torch.utils.data.distributed import DistributedSampler
from Utils import *

def train_embedding(model):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            print('requires_grad', name, param.size())
            param.requires_grad = True

def init_params(model, escape=None):
    for name, param in model.named_parameters():
        if escape is not None and escape in name:
            print('no_init', name, param.size())
            continue
        print('init', name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)

class DefaultTrainer(object):
    def __init__(self, model, local_rank):
        super(DefaultTrainer, self).__init__()
        self.local_rank=local_rank

        if local_rank is not None:
            torch.cuda.set_device(local_rank)

        if torch.cuda.is_available():
            self.model =model.cuda()
        else:
            self.model = model
        self.eval_model = self.model

        if torch.cuda.is_available() and local_rank is not None:
            print("GPU ", self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    def train_batch(self, epoch, data, method, optimizer):
        optimizer.zero_grad()
        loss = self.model(data, method=method)

        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [l.mean().cpu().item() for l in loss]
            # loss = torch.cat([l.mean().view(1) for l in loss]).sum()
            loss = torch.cat(loss, dim=-1).mean()
        else:
            loss = loss.mean()
            closs = [loss.cpu().item()]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
        optimizer.step()
        return closs

    def serialize(self,epoch, output_path):
        if self.local_rank!=0:
            return
        output_path = os.path.join(output_path, 'model/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(self.eval_model.state_dict(), os.path.join(output_path, '.'.join([str(epoch), 'pkl'])))

    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer):
        self.model.train()
        if torch.cuda.is_available():
            sampler = DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)

        start_time = time.time()
        count_batch=0
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            count_batch += 1

            bloss = self.train_batch(epoch, data, method=method, optimizer=optimizer)

            if j >= 0 and j%100==0:
                elapsed_time = time.time() - start_time
                print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time)
                sys.stdout.flush()
            del bloss

        # elapsed_time = time.time() - start_time
        # print(method + ' ', epoch, 'time ', elapsed_time)
        sys.stdout.flush()

    def predict(self,method, dataset, collate_fn, batch_size, epoch, output_path):
        self.eval_model.eval()

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                     num_workers=0)

            # srcs = []
            systems = []
            references = []
            for k, data in enumerate(test_loader, 0):
                if torch.cuda.is_available():
                    data_cuda=dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key]=value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                indices = self.eval_model(data, method=method)
                sents=self.eval_model.to_sentence(data,indices)

                remove_duplicate(sents)

                # srcs += [' '.join(dataset.input(id.item())) for id in data['id']]
                systems += [' '.join(s).replace(SEP_WORD, os.linesep).lower() for s in sents]
                for id in data['id']:
                    refs=dataset.output(id.item())
                    refs=[' '.join(ref).lower() for ref in refs]
                    references.append(refs)

            output_path = os.path.join(output_path, 'result/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # file = codecs.open(os.path.join(output_path, str(epoch)+'.txt'), "w", "utf-8")
            # for i in range(len(systems)):
            #     file.write(srcs[i]+ os.linesep+systems[i]+ os.linesep+os.linesep.join(references[i])+os.linesep+os.linesep)
            # file.close()
        return systems, references

    def test(self,method, dataset, collate_fn, batch_size, epoch, output_path):
        with torch.no_grad():
            systems,references=self.predict(method, dataset, collate_fn, batch_size, epoch, output_path)

        rouges= eval_rouge(systems, references)
        # bleus =eval_bleu(systems, references)
        bleus={}
        print({**rouges, **bleus})
        sys.stdout.flush()
        return rouges, bleus