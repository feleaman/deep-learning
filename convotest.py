from torch import nn
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

conv1 = nn.Conv2d(3, 1, 3)

nn.init.kaiming_normal_(conv1.weight)


weight = conv1.weight.data.numpy()
plt.imshow(weight[0, ...])
plt.show()

