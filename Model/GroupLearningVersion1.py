from Model.Model import  ResNet34AveragePooling
import torch.nn as  nn
import torch
import torch.optim as optim
import torch.nn.functional  as F
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
class GroupLearning(nn.Module):
    def __init__(self,model1,model2,freezed_model):
        super(GroupLearning, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.freezed_model = freezed_model
        for para in self.freezed_model.parameters():
            para.requires_grad = False
        self.optimizer  = optim.Adam(self.parameters(), lr=1e-5)
        self.layer_norm = nn.LayerNorm(512)
    def forward(self,x):
        return self.model1(x)

    def TrainLoss(self, input_tensor):
        # Getting Image and Text Features
        feats1 = self.model1(input_tensor[0])
        feature1 = self.layer_norm(feats1)
        #freezed1 = F.softmax(self.freezed_model(input_tensor[0]),dim=-1)
        #divergence1 = (freezed1 * (log(freezed1) - log(F.softmax(feats1,dim=-1)))).sum(dim=-1)

        feats2 = self.model2(input_tensor[1])
        feature2 = self.layer_norm (feats2)
        #freezed2 = F.softmax(self.freezed_model(input_tensor[1]),dim=-1)
        #divergence2 = (freezed2 * (log(freezed2) - log(F.softmax(feats2,dim=-1)))).sum(dim=-1)

        # Calculating the Loss
        logits = (feature1 @ feature2.T) / 1.0
        feature1_similarity = feature1 @ feature1.T
        feature2_similarity = feature2 @ feature2.T
        targets = F.softmax((feature1_similarity + feature2_similarity) / 2 * 1.0, dim=-1)

        self.optimizer.zero_grad()
        feature1_loss = cross_entropy(logits, targets, reduction='none')
        feature2_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = feature2_loss + feature1_loss# +divergence1 +divergence2) # shape: (batch_size)
        loss.mean().backward()
        self.optimizer.step()
        return loss.mean()

