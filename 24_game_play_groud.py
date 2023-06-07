"""
Trains a GPT (generative pre-trained transformer language model) to play 24 game.

Some backgroud info about 24 game:
Given 4 numbers ranging from [0, 9], use "+ - * /" 4 kinds of operators to try to get 24.

For example:
Given 4, 2, 5, 3
Some valid reasonings include (actually there are 34 possible reasonings):
[4, 2, 5, 3]: 5 - 2 = 3, 3 + 3 = 6, 6 * 4 = 24      
[4, 2, 5, 3]: 4 + 3 = 7, 5 + 7 = 12, 12 * 2 = 24    
......
[4, 2, 5, 3]: 5 + 3 = 8, 8 + 4 = 12, 12 * 2 = 24    

I trained a GPT model to do 24 game reasoning. 
For example, a prompt "[4, 2, 5, 3]: " is given, this GPT model could generate right reasonings, e.g.
GPT generates "5 + 3 = 8, 8 + 4 = 12, 12 * 2 = 24".

"""
import os

import torch
from dataset import DatasetOf24Game
from torch.utils.data.dataloader import DataLoader

from model import GPT
from utils import ConfigNode

def do_batch_testing(model):
    # construct all data dataset
    all_dataset = DatasetOf24Game(split='all')
    loader = DataLoader(all_dataset, batch_size=100, num_workers=0, shuffle=True, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # isolate the first two digits of the input sequence alone
        problem = x[:, :len(DatasetOf24Game.one_problem_sample)]
        # let the model sample the rest of the sequence
        problem_result_pred = model.generate(problem, len(DatasetOf24Game.one_result_sample), do_sample=False) # using greedy argmax, not sampling
        # isolate the last digit of the sampled sequence
        result_pred = problem_result_pred[:, -len(DatasetOf24Game.one_result_sample):]
        # evaluate the correctness of the results in this batch
        mistakes_printed_already = 0
        for i in range(x.size(0)):
            r = torch.cat((x[i, :len(DatasetOf24Game.one_problem_sample)], result_pred[i]), 0)
            r = r.tolist()
            r = "".join([DatasetOf24Game.itoc[i] for i in r])
            results.append(r)
            if r in DatasetOf24Game.all_data_set:
                right_results.append(r)
            if not r in DatasetOf24Game.all_data_set and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that " + r)
        print(f"score so far: {round(len(right_results)/len(results)*100, 2)}% [{len(right_results)} out of {len(results)}] are right predications.")
        # increase 2 to bigger number if you want to test more data from all data samples
        if b >= 2: 
            break

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # construct the model
    config = ConfigNode()
    config.model = GPT.get_default_config()
    config.model.model_type = 'gpt-mini'
    config.model.vocab_size = DatasetOf24Game.get_vocab_size()
    config.model.block_size = DatasetOf24Game.get_block_size()
    # print(config)
    model = GPT(config.model)
    model.to(device)
    # print(model)
    model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/out/get24/4train-6test-6layer-6head-384emb-20000steps.model.pt"))
    model.eval()
    results = []
    right_results = []
    
    # unblock below line if you want to do some testing, data from all data samples
    # do_batch_testing(model)
    
    while True:
        line = input("input 4 numbers seperated by space ' ', e.g. '4 8 9 3'. if you want to stop, type in 'stop'\n")
        if line == 'stop': break
        line = line.strip().split()
        if not len(line) == 4:
            print("input must be 4 single digits!")
            continue
        if not all([a in set("0 1 2 3 4 5 6 7 8 9".split()) for a in line]):
            print("any of 4 numbers must be [0-9]!")
            continue
        line = f"[{line[0]}, {line[1]}, {line[2]}, {line[3]}]: "
        idx = [DatasetOf24Game.ctoi[c] for c in line]
        problem = torch.tensor(idx, dtype=torch.long).to(device)
        problem = problem.view(1, -1)
        problem_result_pred = model.generate(problem, len(DatasetOf24Game.one_result_sample), do_sample=False) # using greedy argmax, not sampling
        # isolate the last digit of the sampled sequence
        result_pred = problem_result_pred[:, -len(DatasetOf24Game.one_result_sample):]
        r = torch.cat((problem.view(-1), result_pred[0]), 0)
        r = r.tolist()
        r = "".join([DatasetOf24Game.itoc[i] for i in r])
        print(r)
        
    


