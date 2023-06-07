"""
Trains a GPT (generative pre-trained transformer language model) to play 24 game.

Some background info about 24 game:
Given 4 numbers ranging from [0, 9], use "+ - * /" 4 kinds of operators to try to get 24.
For example:
Given 4, 2, 5, 3
Some valid reasonings include (there are 34 possible reasonings):
[4, 2, 5, 3]: 5 - 2 = 3, 3 + 3 = 6, 6 * 4 = 24      
[4, 2, 5, 3]: 4 + 3 = 7, 5 + 7 = 12, 12 * 2 = 24    
......
[4, 2, 5, 3]: 5 + 3 = 8, 8 + 4 = 12, 12 * 2 = 24    

This script generates all 323188 possibilities (i.e. data samples). 
For some combination of 4 numbers, there are dozens of valid reasonings. 
For example, 3, 7, 5, 5, we have 17 different reasonings
 [3, 7, 5, 5]: 5 - 3 = 2, 5 + 7 = 12, 12 * 2 = 24
 [3, 7, 5, 5]: 5 - 3 = 2, 5 + 7 = 12, 2 * 12 = 24
 [3, 7, 5, 5]: 5 - 3 = 2, 7 + 5 = 12, 12 * 2 = 24
 [3, 7, 5, 5]: 5 - 3 = 2, 7 + 5 = 12, 2 * 12 = 24
 [3, 7, 5, 5]: 5 / 5 = 1, 1 + 7 = 8, 3 * 8 = 24
 [3, 7, 5, 5]: 5 / 5 = 1, 1 + 7 = 8, 8 * 3 = 24
 [3, 7, 5, 5]: 5 / 5 = 1, 7 + 1 = 8, 3 * 8 = 24
 [3, 7, 5, 5]: 5 / 5 = 1, 7 + 1 = 8, 8 * 3 = 24
 [3, 7, 5, 5]: 5 + 7 = 12, 5 - 3 = 2, 12 * 2 = 24
 [3, 7, 5, 5]: 5 + 7 = 12, 5 - 3 = 2, 2 * 12 = 24
 [3, 7, 5, 5]: 7 + 5 = 12, 5 - 3 = 2, 12 * 2 = 24
 [3, 7, 5, 5]: 7 + 5 = 12, 5 - 3 = 2, 2 * 12 = 24

"""

import os
import random

all_valid_data_set = set()

def compute(nums: list[int], reasoning: str) -> None:
    if len(nums) == 1:
        if nums[0] == 24:
            all_valid_data_set.add(reasoning[:-2])
        return
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j: continue
            new_nums = [d for di, d in enumerate(nums) if di not in [i, j]]
            compute(new_nums[:] + [(nums[i] + nums[j])], reasoning = reasoning + str(nums[i]) + " + " + str(nums[j]) + " = " + str(nums[i] + nums[j]) + ", " )
            compute(new_nums[:] + [(nums[i] * nums[j])], reasoning = reasoning + str(nums[i]) + " * " + str(nums[j]) + " = " + str(nums[i] * nums[j]) + ", " )
            if nums[i] >= nums[j]:
                compute(new_nums[:] + [(nums[i] - nums[j])], reasoning = reasoning + str(nums[i]) + " - " + str(nums[j]) + " = " + str(nums[i] - nums[j]) + ", " )
            if nums[j] !=0 and nums[i] % nums[j] == 0:
                compute(new_nums[:] + [(nums[i] // nums[j])], reasoning = reasoning + str(nums[i]) + " / " + str(nums[j]) + " = " + str(nums[i] // nums[j]) + ", " )

if __name__ == '__main__':
    max_num = 10
    for a in range(max_num):
        for b in range(max_num):
            for c in range(max_num):
                for d in range(max_num):
                    compute([a, b, c, d], str([a, b, c, d]) + ": ")
    all_valid_data_list = list(all_valid_data_set)
    random.shuffle(all_valid_data_list)
    max_seq_len = max([len(a) for a in all_valid_data_list])
    all_valid_data_list = [a.ljust(max_seq_len, ' ') for a in all_valid_data_list]
    with open(os.path.dirname(os.path.realpath(__file__)) + "/24_game_all_data.txt", "w") as f:
        f.write('\n'.join(all_valid_data_list))

    print("totally " + str(len(all_valid_data_list)) + " lines wrote!")