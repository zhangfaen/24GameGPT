"""
Trains a GPT (generative pre-trained transformer language model) to play 24 game.

Some background info about 24 game:
Given 4 numbers ranging from [0, 9], use "+ - * /" 4 kinds of operators to try to get 24.
For example:
Given 4, 2, 5, 3
Some valid reasonings are (there are 34 possible reasonings):
[4, 2, 5, 3]: 5 - 2 = 3, 3 + 3 = 6, 6 * 4 = 24      
[4, 2, 5, 3]: 4 + 3 = 7, 5 + 7 = 12, 12 * 2 = 24    
......
[4, 2, 5, 3]: 5 + 3 = 8, 8 + 4 = 12, 12 * 2 = 24    

I trained a GPT model to predicate reasoning according to a 
prompt (in source code, I use 'problem' to represent a prompt).
The trained model takes a prompt "[3, 7, 5, 5]: " as input, we expect
the mode to output something like "7 + 5 = 12, 5 - 3 = 2, 2 * 12 = 24".

As we use a language model to solve game 24, we use
"[3, 7, 5, 5]: 7 + 5 = 12, 5 - 3 = 2, 2 * 12 = 24" and
"[3, 5, 5, 7]: 7 + 5 = 12, 5 - 3 = 2, 2 * 12 = 24" as different data samples.

We use 40% of all data samples (see 24_game_data_generator.py about how to generate all possible data samples) to train the model. 
The model runs on all data samples and achieves 99.1% accuracy.
"""

import os
import sys
import json

import torch
from torch.utils.data.dataloader import DataLoader

from model import GPT
from trainer import Trainer
from utils import set_seed, ConfigNode
from dataset import DatasetOf24Game

# -----------------------------------------------------------------------------

def get_config():
    C = ConfigNode()

    # system
    C.system = ConfigNode()
    C.system.seed = 3407
    C.system.work_dir = './out/get24'

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.max_iters = 20000

    return C

if __name__ == '__main__':
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    
    # create the work directory if it doesn't already exist
    os.makedirs(config.system.work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(config.system.work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(config.system.work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))
    
    # see random seeds for everywhere
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = DatasetOf24Game(split='train')
    test_dataset  = DatasetOf24Game(split='test')

    # construct the model
    config.model.vocab_size = DatasetOf24Game.get_vocab_size()
    config.model.block_size = DatasetOf24Game.get_block_size()
    print(config)
    model = GPT(config.model)
    print(model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        results = []
        right_results = []
        right_results_in_test = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            # isolate the first two digits of the input sequence alone
            problem = x[:, :len(DatasetOf24Game.one_problem_sample)]
            # let the model sample the rest of the sequence
            problem_result_pred = model.generate(problem, len(DatasetOf24Game.one_result_sample), do_sample=False) # using greedy argmax, not sampling
            # isolate the last digit of the sampled sequence
            result_pred = problem_result_pred[:, -len(DatasetOf24Game.one_result_sample):]
            # evaluate the correctness of the results in this batch
            for i in range(x.size(0)):
                r = torch.cat((x[i, :len(DatasetOf24Game.one_problem_sample)], result_pred[i]), 0)
                r = r.tolist()
                r = "".join([DatasetOf24Game.itoc[i] for i in r])
                # r = r.rstrip('\n').replace(" ", "")
                r = r.rstrip('\n')
                results.append(r)
                if r in DatasetOf24Game.all_data_set:
                    right_results.append(r)
                if r in DatasetOf24Game.all_test_data_set:
                    right_results_in_test.append(r)
                if not r in DatasetOf24Game.all_data_set and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that " + r)
            if max_batches is not None and b+1 >= max_batches:
                break
        print(f"{split} final score: {len(right_results)} out of {len(results)} are right predications, {len(right_results_in_test)} out of all {len(right_results)} right predications are not in training data.")
        return len(right_results)

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num > 0 and trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # train_score = eval_split(trainer, 'train', max_batches=5)
                train_score = 0
                test_score  = eval_split(trainer, 'test',  max_batches=10)
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
