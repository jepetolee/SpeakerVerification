import torch
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from RunMel import *
from bayes_opt import *

def blackBoxFunctionForLearningRateAndBatchSize(batch_size, learning_rate,loss_weight_decay, model_weight_decay):
    if 0 <= batch_size < 1:
       batch_size = 4
    elif 1 <= batch_size < 2:
       batch_size = 16
    elif 2 <= batch_size < 3:
        batch_size = 64
    else:
        batch_size = 256

    return 100 - RunWithBatchArguments(batch_size=batch_size, lr=learning_rate,num_epochs=2, loss_weight_decay=loss_weight_decay, model_weight_decay=model_weight_decay)


def blackBoxFunctionForEverything(#batch_size, learning_rate,loss_weight_decay, model_weight_decay,
                                  window_size, hop_size, window_fn, n_mel,
                                  margin,scale):
  #  if 0 <= batch_size < 1:
 #      batch_size = 16
   # elif 1 <= batch_size < 2:
   #    batch_size = 32
   # elif 2 <= batch_size < 3:
   #     batch_size = 64
   # else:
   #     batch_size = 128

    if 0 <= window_fn < 1:
       window_fn = torch.hamming_window
    elif 1 <= window_fn < 2:
       window_fn = torch.hann_window
    elif 2 <= window_fn < 3:
       window_fn = torch.kaiser_window
    elif 3 <= window_fn < 4:
        window_fn = torch.bartlett_window


    if 0 <= n_mel < 1:
        n_mel = 64
    elif 1 <= n_mel < 2:
        n_mel = 80
    else:
        n_mel = 96

    return 100 - RunWithBatchArguments(# batch_size=batch_size, lr=learning_rate,num_epochs=2, loss_weight_decay=loss_weight_decay, model_weight_decay=model_weight_decay,
                                       window_size=int(window_size), hop_size=int(hop_size),window_fn=window_fn,n_mel=n_mel,
                                       margin=margin,scale=int(scale))


#LrAndBatchBound = {'batch_size': (0, 4), 'learning_rate': (1e-7, 1e-3),'loss_weight_decay': (1e-6, 1e-3),"model_weight_decay": (1e-6, 1e-3)}

EverythingBound = {'window_size': (80, 2000), 'hop_size': (80, 500),'window_fn': (0, 4),"n_mel": (0, 3),
                   #'batch_size': (0, 4), 'learning_rate': (1e-7, 1e-3),'loss_weight_decay': (1e-6, 1e-3),"model_weight_decay": (1e-6, 1e-3),
                   "margin":(0,1),"scale":(1,60)
                   }

optimizer = BayesianOptimization(f=blackBoxFunctionForEverything,
                                 pbounds=EverythingBound,
                                 random_state=8)
logger = JSONLogger(path="./logs.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(init_points=10,
                   n_iter=100)