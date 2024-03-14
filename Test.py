import torch.optim as optim
import torchvision
from AAM_Softmax import *
from Dataloader import *

from torch.utils.data import DataLoader

trainset = train_loader("./data/VoxCeleb1/train_list.txt","./data/VoxCeleb1/train",None,None,300)
trainloader = DataLoader(trainset, batch_size = 32, shuffle = True,num_workers = 10, drop_last = True)


# Initialize ResNet-18 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestingModel(nn.Module):
    def __init__(self):
            super(TestingModel, self).__init__()
            self.Reader = nn.Conv2d(1,3,kernel_size =1,stride=1,padding=0)
            self.aa = torchvision.models.resnet18(pretrained=True)
            self.Classifier = nn.Sequential(nn.Linear(1000, 192))

    def forward(self, input):
            x =self.Reader(input.reshape(-1,1,40,13))
            x = self.aa(x)
            return self.Classifier(x)

net = TestingModel().cuda()

# Define loss function and optimizer
criterion = AAMsoftmax(n_class = 1211,m=0.2,s=30).cuda()
optimizer = optim.AdamW(net.parameters(), lr=0.02,  weight_decay=5e-4)

# Training the model
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader,start=1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss,prec = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Epoch: {epoch} | Loss: {train_loss/len(trainloader):.3f} | Acc: {100.*correct/total:.3f}')

# Main training loop
num_epochs = 10
for epoch in range(num_epochs):
    train(epoch)


'''
    

testset = train_loader("./data/VoxCeleb1/train_list.txt","./data/VoxCeleb1/test",None,None,300)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
'''