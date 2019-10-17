import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torch.quantization import QuantStub, DeQuantStub
import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = nn.Sequential(
            nn.Conv2d(       #(1*28*28)
                in_channels=1,
                out_channels=128,
                kernel_size=5,
                stride=1, #步长
                padding=2,
            ),    #(16*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(16*14*14)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(128,512,5,1,2),  #32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32*7*7
        )
        self.out = nn.Linear(512*7*7,10)  #全连接

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.conv2(x)   #(batch,32,7,7)
        # x = x.view(x.size(0),-1) #(batch,32*7*7)
        x = x.reshape(x.size(0),-1) #(batch,32*7*7)
        output = self.out(x)
        output = self.dequant(output)
        return output

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False
)

test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

model = CNN()
model.load_state_dict(torch.load('./models/mnist_cnn.pth'))
model.eval()
print_size_of_model(model)

print("Float CPU test:")
tic = time.time()
for _ in range(100):
    test_output = model(test_x[:10])
    pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
toc = time.time()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
print("Cost time {} seconds.".format((toc-tic)/100))

# model.fuse_model()
print("Float GPU  Test")
model = model.cuda()
test_x_g, test_y_g = test_x.cuda(), test_y.cuda()
tic = time.time()
for _ in range(100):
    test_output = model(test_x_g[:10])
    pred_y = torch.max(test_output,1)[1].cpu().data.numpy().squeeze()
toc = time.time()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
print("Cost time {} seconds.".format((toc-tic)/100))

print("Quant CPU test:")
model = model.cpu()
model.qconfig = torch.quantization.default_qconfig
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

print_size_of_model(model)

tic = time.time()
for _ in range(100):
    test_output = model(test_x[:10])
    pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
toc = time.time()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
print("Cost time {} seconds.".format((toc-tic)/100))

print("Perchannel Quant CPU test:")
del model
model = CNN()
model.load_state_dict(torch.load('./models/mnist_cnn.pth'))
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

print_size_of_model(model)
tic = time.time()
for _ in range(100):
    test_output = model(test_x[:10])
    pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
toc = time.time()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
print("Cost time {} seconds.".format((toc-tic)/100))
torch.jit.save(torch.jit.script(model), './models/mnist_cnn_per_channel_quant.pth')
