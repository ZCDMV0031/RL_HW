import numpy as np
from arguments import get_args
import torch
from abc import abstractmethod
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_args()

class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=5)

		self.fc1 = nn.Linear(in_features=6 * 53 * 53, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=64)
		self.out = nn.Linear(in_features=64, out_features=13)

	def forward(self, x):

		t = self.conv1(x)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		# print("t.shape:",t.shape)
		t = t.reshape(-1, 6 * 53 * 53)
		t = self.fc1(t)
		t = F.relu(t)
		t = self.fc2(t)
		t = F.relu(t)
		t = self.out(t)

		return t



# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = None

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		self.model.train(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict(data_batch)
		return label_predict


class MyDagger(DaggerAgent):
	def __init__(self,init_parameters = None):
		super(DaggerAgent, self).__init__()

		self.batch_size = args.num_steps
		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.23,0.22,0.24])
		])

		self.epochs = 4
		self.numclass = 13		#一共8个动作
		self.lr = 0.001
		self.loss_fn = nn.CrossEntropyLoss()
		self.model = CNN().to(device=device)
		self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)

	def update(self,data_batch, label_batch):
		data_batch = np.array(data_batch)
		# print("shape:")
		label_batch=np.array(label_batch)
		label_batch = torch.from_numpy(label_batch)
		label_batch=label_batch.to(device)

		for epoch in range(self.epochs):
			total_loss = 0

			for i in range(0,len(data_batch), self.batch_size):
				inputs = torch.stack([self.transform(img) for img in data_batch[i:i+self.batch_size]])

				inputs = inputs.to(device)
				# print("input:",inputs.shape)
				labels = label_batch[i:i+self.batch_size]
				self.optimizer.zero_grad()
				outputs = self.model(inputs)
				# print("shape_pad:",outputs.shape,labels)
				loss = self.loss_fn(outputs, labels.long())
				total_loss += loss.item()
				loss.backward()
				self.optimizer.step()

			print('epoch:{}, loss: {}'.format(epoch, total_loss))

	def select_action(self, obs):
		self.model.eval()
		# data_batch = Image.fromarray(data_batch)
		# print("lenobs:",len(obs))
		if len(obs)==2:
			data_batch,state = obs
		else:
			data_batch = obs
		inputs = self.transform(data_batch)
		# print("shape:",inputs.shape)
		inputs = torch.unsqueeze(inputs,dim=0)
		inputs = inputs.to(device)

		y_pre = self.model(inputs)
		y_pre = torch.squeeze(y_pre,dim=0)
		y_pre = y_pre.cpu().detach().numpy()
		y_pre = np.argmax(y_pre,axis=0)

		return y_pre

