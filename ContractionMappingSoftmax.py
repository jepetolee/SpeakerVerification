import math
import torch.nn as nn
import torch.nn.functional as F
import torch


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class CMSoftmax(nn.Module):
    def __init__(self,
                 margin=0.0,
                 margin2=0.0,
                 scale=1.0,
                 n_class=1000,
                 easy_margin=False):
        super(CMSoftmax, self).__init__()
        self.easy_margin = easy_margin
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_class, 512), requires_grad=True)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.nll_loss = nn.NLLLoss()
        self.n_classes = n_class
        self.margin2 = margin2

    def forward(self, logits, targets):
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))

        sine = torch.sqrt(1.0 - torch.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        target_one_hot = F.one_hot(targets, self.n_classes)
        outputs = (target_one_hot * phi) + (
                (1.0 - target_one_hot) * cosine) - target_one_hot * self.margin2
        outputs = self.scale * outputs
        pred = F.log_softmax(outputs, dim=-1)
        prec1 = accuracy(outputs.detach(), targets.detach(), topk=(1,))[0]
        return self.nll_loss(pred, targets), prec1[0]
