from sklearn.datasets import load_digits
from sys import exit
digits = load_digits()
# print(digits.data.shape)

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(digits.images[0])

# plt.show()

x = digits.data
y = digits.target
print(len(y))

print('=================== Scikit =========================')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=19, shuffle=True, stratify=y)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(32), activation='relu', alpha=0.01, random_state=42, max_iter=10, solver='sgd', batch_size=16)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Test Scikit:', accuracy)

# print(clf.out_activation_)
# exit()



print('=================== Pytorch =========================')
import torch
from torch import nn

#+++++Hyperparametros
n_epochs = 1000
lr = 0.01

#+++++Device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

#+++++Datos a Tensor
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device=device, dtype=torch.long)

x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device=device, dtype=torch.long)
# print(y_train_tensor)
# print(y_train_tensor.shape)
# print(y_train_tensor.dtype)
# exit()
# N = 1797
# x_train_tensor = torch.randn(N, 64).float().to(device)
# y_train_tensor = torch.randn(N, 10).float().to(device)

# print(x_train_tensor.type())

#Loss Function
torch.manual_seed(42)
# loss_fn = nn.MSELoss(reduction='mean')
loss_fn = nn.CrossEntropyLoss()


# Define model
class FeedForward(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(FeedForward, self).__init__()
		self.input_size = input_size
		self.hidden_size  = hidden_size
		self.fc1 = nn.Linear(self.input_size, self.hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(self.hidden_size, 10)
		# self.softmax = nn.Softmax(dim=1)
	def forward(self, x):
		
		hidden = self.fc1(x)		
		relu = self.relu(hidden)		
		output = self.fc2(relu)
		# output = self.softmax(output)
		# return output.argmax(1)
		return output


model = FeedForward(64, 32).to(device=device)
# print(model.dtype)
# exit()

# for param in model.parameters():
	# print(type(param), param.size())
	# print(param.requires_grad)
# print(model.state_dict())

# exit()
# print('++++', model.requires_grad)
# for param in model.parameters():
    # param.requires_grad = True

#Optimizador
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=lr)


#Entrenamiento
for epoch in range(n_epochs):
	model.train()
	
	#Prediccion
	yhat = model(x_train_tensor)
	# print(yhat.shape)
	# print(y_train_tensor.shape)
	# exit()
	
	#Calculo de gradientes
	# loss = loss_fn(y_train_tensor, yhat)
	loss = loss_fn(yhat, y_train_tensor)
	# print(loss)
	# print(loss.shape)
	# exit()
	loss.backward()
	
	#Contador
	# print(epoch, loss.item())
	
	#Update
	optimizer.step()
	
	#Resetear gradientes
	optimizer.zero_grad()

model.eval()
with torch.no_grad():
	pred = model(x_test_tensor)
	pred = pred.argmax(1)
	# predicted, actual = classes[pred[0].argmax(0)], classes[y]
	# print(pred.numpy())

accuracy = accuracy_score(y_test_tensor, pred)
print('Accuracy Test Pytorch:', accuracy)