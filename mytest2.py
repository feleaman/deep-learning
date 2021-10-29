import torch
from sys import exit

class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

class FeedForward(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(FeedForward, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)
		self.relu = torch.nn.ReLU()
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		# h_relu = self.linear1(x).clamp(min=0)
		h = self.linear1(x)
		h_relu = self.relu(h)		
		y_pred = self.linear2(h_relu)
		y_pred = self.softmax(y_pred)
		return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
# model = TwoLayerNet(D_in, H, D_out)
model = FeedForward(D_in, H, D_out)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
	# Forward pass: Compute predicted y by passing x to the model
	y_pred = model(x)
	# print(y_pred)
	# print(type(y_pred))
	print(y.shape)
	exit()

	# Compute and print loss
	loss = criterion(y_pred, y)
	# print(loss)
	# print(loss.shape)
	# exit()
	if t % 100 == 99:
		print(t, loss.item())

	# Zero gradients, perform a backward pass, and update the weights.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()