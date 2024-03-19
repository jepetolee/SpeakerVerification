import torch.optim as optim
from AAM_Softmax import AAMsoftmax
from Dataloader import train_loader
from torch.utils.data import DataLoader

# Training the model
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader,start=1):
        inputs, targets = inputs.cuda(), targets.cuda()

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


net = TestingModel().cuda()

# Define loss function and optimizer
criterion = AAMsoftmax(n_class = 1211,m=0.2,s=30).cuda()
optimizer = optim.AdamW(net.parameters(), lr=0.02,  weight_decay=5e-4)


trainset = train_loader("./data/VoxCeleb1/train_list.txt","./data/VoxCeleb1/train",None,None,300)
trainloader = DataLoader(trainset, batch_size = 32, shuffle = True,num_workers = 10, drop_last = True)



'''
testset = train_loader("./data/VoxCeleb1/train_list.txt","./data/VoxCeleb1/test",None,None,300)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
'''