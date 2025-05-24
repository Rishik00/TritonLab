import torch
import torch.nn as nn

lst = [i * 0.87 for i in range(8)]
input = torch.tensor(lst)
print(lst, input.sum())
# Take exponent of each element
exp_input = torch.exp(input)
m = nn.Sigmoid()
# Sum all exponentiated elements
sum_exp = exp_input.sum()

print("Exponentiated values:", exp_input)
print("Sum of exponentiated values:", sum_exp)
print("softmaxed: ", m(input))
