import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import preprocess


class CNN(nn.Module):
	def __init__(self, in_frames=5):
		super(CNN, self).__init__()
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channels=in_frames, out_channels=96, kernel_size=3, stride=1)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.fc6 = nn.Linear(in_features=512*6*6, out_features=1)

	def forward(self, x):

		conv1_out = self.conv1(x)
		# print("conv1", conv1_out.shape)
		pool1_out = self.pool1(self.relu(conv1_out))
		# print("pool1", pool1_out.shape)
		conv2_out = self.conv2(pool1_out)
		# print("conv2_out", conv2_out.shape)
		pool2_out = self.pool2(self.relu(conv2_out))
		# print("pool2_out", pool2_out.shape)
		conv3_out = self.conv3(pool2_out)
		# print("conv3_out", conv3_out.shape)
		conv4_out = self.conv4(self.relu(conv3_out))
		# print("conv4_out", conv4_out.shape)
		conv5_out = self.conv5(self.relu(conv4_out))
		# print("conv5_out", conv5_out.shape)
		pool5_out = self.pool5(self.relu(conv5_out))
		# print("pool5_out", pool5_out.shape)
		pool5_out = pool5_out.flatten()
		o = self.fc6(pool5_out)
		o = o.view(1,1)
		# print("o", o.shape)
		return o


class LSTM(nn.Module):
	def __init__(self, input_size=1, hidden_size=256, num_layers=3):
		super(LSTM, self).__init__()
		self.layers = nn.LSTM(input_size, hidden_size, num_layers)

	def forward(self, x, h0, c0):
		out, (h_n, c_n) = self.layers(x, (h0, c0))

		return out, h_n, c_n


class Watch(nn.Module):
	def __init__(self, in_frames=5, lstm_input_size=1, lstm_hidden_size=256, lstm_num_layers=3):
		super(Watch, self).__init__()
		self.cnn = CNN(in_frames)
		self.lstm = LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
		self.hidden_size = lstm_hidden_size
		self.num_layers = lstm_num_layers


	def forward(self, x):
		# x is 5 frames worth of data
		condensed_frames = self.cnn(x)
		h0 = torch.zeros(self.hidden_size)
		c0 = torch.zeros(self.hidden_size)
		print("o", condensed_frames.shape)
		out, h_n, c_n = self.lstm(condensed_frames, h0, c0)

		return out

if __name__ == "__main__":
	# model = Watch()
	#
	# model.train()
	ids = preprocess.get_ids()
	# video_features = preprocess.load_video(ids)
	# input = video_features[:,0:5,:,:]
	# print(input.shape, input.dtype)
	#
	# o = model(input)

	audio_features = preprocess.load_audio(ids)

	# model(video_features[0:5])
