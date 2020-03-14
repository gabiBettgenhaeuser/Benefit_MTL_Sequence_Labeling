import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_tasks, batch_size):
		super(Net, self).__init__()

		# Class attributes
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_tasks = num_tasks
		self.batch_size = batch_size

		self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
		self.fc1 = nn.Linear(hidden_size * 2, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.ModuleList([nn.Linear(64, output_size) for i in range(num_tasks)])

	def forward(self, x):
		# Initialize hidden state with zeros
		# (num_layers * num_directions, batch, hidden_size
		h0 = torch.zeros(2, x.size(0), self.hidden_size, device = DEVICE).requires_grad_()

		# Initialize cell state
		c0 = torch.zeros(2, x.size(0), self.hidden_size,device = DEVICE).requires_grad_()

		# Seq_len time steps
		# We need to detach as we are doing truncated backpropagation through time (BPTT)
		# If we don't, we'll backprop all the way to the start even after going through another batch
		out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

		# Index hidden state of last time step
		# out.size() --> batch_size, seq_len, emb_dim
		# out[:, -1, :] -> just want last time step hidden states! - last sentence
		out = self.fc1(out[:, -1, :])
		out = self.fc2(out)
		out = self.fc3(out)
		outputs = torch.zeros(self.num_tasks, self.batch_size, self.output_size).to(DEVICE)
		for i in range(self.num_tasks):
			outputs[i] = self.fc4[i](out)
		return outputs

 

