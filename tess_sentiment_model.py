import torchaudio
import torch
import download
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from time import sleep
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu" 

def get_file_list(file_path = "dataverse_files/"):
	arr = os.listdir(file_path)
	return arr

def get_emotion_labels(file_list):
	encoding = {"angry":0, "disgust":1,"fear":2, "happy":3,"ps":4, "sad": 5,"neutral":6}
	labels = []
	for file_name in file_list:
		components = file_name.split("_")
		emotion = components[2].split(".")[0]
		if ' ' in emotion:
			emotion = emotion.split(" ")[0]
		labels.append(encoding[emotion])

	return torch.LongTensor(labels)

def generate_sequence(x,y,size=64):
	combined = []
	for i,datum in enumerate(x):
		label = y[i]
		combined.append((datum,label))

	return DataLoader(combined,batch_size=size)

def generate_MFCC(file_list, folder="dataverse_files/",padding=730):
	max = 0
	audio_features = []
	for file in file_list:
		path = folder + file
		audio, sample_rate = torchaudio.load(path)
		m = torchaudio.transforms.MFCC(sample_rate, n_mfcc=96)
		feat_vec = m(audio)
		if feat_vec.shape[2] > max:
			max = feat_vec.shape[2]
		feat_vec = torch.nn.functional.pad(feat_vec, (0, padding-feat_vec.shape[2]))
		audio_features.append(feat_vec)

	audio_features = torch.stack(audio_features)
	return audio_features

class FCN(nn.Module):
	def __init__(self, in_channels=1):
		super(FCN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels,128,3)
		self.batchNorm1 = nn.BatchNorm2d(128)
		self.conv2 = nn.Conv2d(128,384,3)
		self.batchNorm2 = nn.BatchNorm2d(384)
		self.conv3 = nn.Conv2d(384,768,3)
		self.batchNorm3= nn.BatchNorm2d(768)
		self.conv4 = nn.Conv2d(768,2048,3)
		self.batchNorm4= nn.BatchNorm2d(2048)
		self.fc = nn.Linear(2048,7)
		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.mp1 = nn.MaxPool2d(kernel_size=(2,4),padding=(1,2))
		self.mp4 = nn.MaxPool2d(kernel_size=(10,4))
		self.drop = nn.Dropout2d(.5)

	def forward(self,x):
		x = self.conv1(x)
		x = self.batchNorm1(x)
		x = self.relu(x)
		x = self.mp1(x)
		x = self.drop(x)

		x = self.conv2(x)
		x = self.batchNorm2(x)
		x = self.relu(x)
		x = self.mp1(x)
		x = self.drop(x)

		x = self.conv3(x)
		x = self.batchNorm3(x)
		x = self.relu(x)
		x = self.mp1(x)
		x = self.drop(x)

		x = self.conv4(x)
		x = self.batchNorm4(x)
		x = self.relu(x)
		x = self.mp4(x)
		x = self.drop(x)

		x = x.view(-1,2048)
		x = self.fc(x)
		x = self.sig(x)

		return x

def training(model,loader,learning_rate,decay,num_epochs=20):
	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
	for epoch in range(num_epochs):
		running_loss = []
		for inputs, labels in tqdm(loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			out = model(inputs)
			loss = criterion(out,labels)
			loss.backward()
			optimizer.step()
			running_loss.append(loss.item())
		print("Epoch {} loss:{}".format(epoch+1,np.mean(running_loss)))
		sleep(.25)
	print('Finished Training')

def evaluate(model, loader): # Evaluate accuracy on validation / test set
	model.eval() # Set the model to evaluation mode
	correct = 0
	with torch.no_grad(): # Do not calculate grident to speed up computation
		for batch, label in tqdm(loader):
			batch = batch.to(device)
			label = label.to(device)
			pred = model(batch)
			print(torch.argmax(pred,dim=1))
			correct += (torch.argmax(pred,dim=1)==label).sum().item()
	acc = correct/len(loader.dataset)
	print("Evaluation accuracy: {}".format(acc))
	return acc

if __name__ == "__main__":
	file_list = get_file_list() 
	# train_len = int(len(file_list)*.7)
	# test_len = len(file_list) - train_len
	# train_subset, test_subset = torch.utils.data.dataset.random_split(file_list,(train_len,test_len))
	# mfcc_features_train = generate_MFCC(train_subset)
	# mfcc_features_test = generate_MFCC(test_subset)
	# training_labels = get_emotion_labels(train_subset)
	# testing_labels = get_emotion_labels(test_subset)
	# trainloader = generate_sequence(mfcc_features_train, training_labels)
	# testloader = generate_sequence(mfcc_features_test,testing_labels)
	# total_labels = torch.cat((training_labels, testing_labels), 0)

	mfcc_features = generate_MFCC(file_list)
	labels = get_emotion_labels(file_list)
	x_train,x_test,y_train,y_test = train_test_split(mfcc_features,labels,train_size=.7, stratify=labels)

	trainloader = generate_sequence(x_train,y_train)
	testloader = generate_sequence(x_test,y_test)
	print(x_train.shape,x_test.shape)
	model = FCN().to(device)
	model.load_state_dict(torch.load("sentiment_model.dms",map_location=torch.device('cpu')))
	print("Starting Training")
	sleep(.25)
	training(model,trainloader,learning_rate=.0001,decay=1e-6,num_epochs=500)
	print("Saving")
	torch.save(model.state_dict(), "sentiment_model")
	evaluate(model, testloader)