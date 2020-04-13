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
		self.conv1 = nn.Conv2d(in_channels,32,1)
		self.batchNorm1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32,32,3)
		self.batchNorm2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,64,1)
		self.batchNorm3= nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64,64,3)
		self.batchNorm4= nn.BatchNorm2d(64)
		self.fc1 = nn.Linear(64*1*5,512)
		self.fc2 = nn.Linear(512,8)
		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.mp1 = nn.MaxPool2d(kernel_size=(2,2))
		self.drop = nn.Dropout()

	def forward(self,x):
		x = self.conv1(x)
		# x = self.batchNorm1(x)
		x = self.relu(x)
		x = self.conv2(x)
		# x = self.batchNorm2(x)
		x = self.relu(x)
		x = self.mp1(x)
		x = self.drop(x)

		x = self.conv3(x)
		# x = self.batchNorm3(x)
		x = self.relu(x)
		x = self.conv4(x)
		# x = self.batchNorm4(x)
		x = self.relu(x)
		x = self.mp1(x)
		x = self.drop(x)

		x = x.view(-1,64*1*5)
		x = self.fc1(x)
		x = self.drop(x)
		x = self.fc2(x)
		return x

def training(model,loader,val,learning_rate,decay,num_epochs=20):
	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
	val_acc = []
	train_acc = []
	train_loss = []
	val_loss = []
	for epoch in range(num_epochs):
		running_train_loss = []
		running_val_loss = []
		val_correct = 0
		val_total = 0
		train_correct = 0
		train_total = 0
		for train, val_set in tqdm(zip(loader,val),total=len(val)):
			inputs = train[0].to(device)
			labels = train[1].to(device)
			optimizer.zero_grad()
			out = model(inputs)
			loss = criterion(out,labels)
			loss.backward()
			optimizer.step()
			running_train_loss.append(loss.item())
			val_inputs = val_set[0].to(device)
			val_labels = val_set[1].to(device)
			val_out = model(val_inputs)

			train_correct += (torch.argmax(out,dim=1)==labels).sum().item()
			train_total += len(labels)

			val_correct += (torch.argmax(val_out,dim=1)==val_labels).sum().item()
			val_total += len(val_labels)
			running_val_loss.append(criterion(val_out,val_labels).item())

			del inputs
			del labels
			del out
			del loss
			del val_inputs
			del val_labels

		val_acc.append(float(val_correct)/val_total)
		train_acc.append(float(train_correct)/train_total)
		train_loss.append(np.mean(running_train_loss))
		val_loss.append(np.mean(running_val_loss))

		print("Epoch train {} loss:{}".format(epoch+1,np.mean(running_train_loss)))
		print("Epoch train accuracy: ", train_acc[len(train_acc)-1])
		print("Epoch val {} loss:{}".format(epoch+1,np.mean(running_val_loss)))
		print("Epoch val accuracy: ", val_acc[len(val_acc)-1])
		sleep(.1)
	print('Finished Training')
	return val_acc,val_loss,train_acc,train_loss

def evaluate(model, loader): # Evaluate accuracy on validation / test set
	model.eval() # Set the model to evaluation mode
	correct = 0
	with torch.no_grad(): # Do not calculate grident to speed up computation
		for batch, label in tqdm(loader):
			batch = batch.to(device)
			label = label.to(device)
			pred = model(batch)
			correct += (torch.argmax(pred,dim=1)==label).sum().item()
			del batch
			del label
			del pred
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

	mfcc_features = generate_MFCC(file_list,padding=730)
	labels = get_emotion_labels(file_list)
	x_train,x_test,y_train,y_test = train_test_split(mfcc_features,labels,train_size=.7, stratify=labels)

	trainloader = generate_sequence(x_train,y_train)
	testloader = generate_sequence(x_test,y_test)
	print(x_train.shape,x_test.shape)
	model = FCN().to(device)
	# model.load_state_dict(torch.load("sentiment_model.dms",map_location=torch.device('cpu')))
	print("Starting Training")
	sleep(.25)
	training(model,trainloader,learning_rate=.0001,decay=1e-6,num_epochs=500)
	print("Saving")
	torch.save(model.state_dict(), "sentiment_model")
	evaluate(model, testloader)