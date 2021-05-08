from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import load_data,normalize,toy_data
from models import GNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=426, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--ndim', type=int, default=2,
                    help='Embeddings dimension.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj = load_data()
adj = toy_data()
adj_norm = normalize(adj)
features = np.identity(adj.shape[0])

adj = torch.FloatTensor(np.array(adj))
features = torch.FloatTensor(np.array(features))
adj_norm = torch.FloatTensor(np.array(adj_norm))


# Model and optimizer
model = GNN(nfeat=features.shape[1],
            nhid=args.hidden,
            ndim=2*args.ndim,
            dropout=args.dropout)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.reconstructions.register_forward_hook(get_activation('reconstructions'))

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_norm = adj_norm.cuda()


# Train model
t_total = time.time()

for epoch in range(args.epochs):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_norm)
    criterion = torch.nn.MSELoss()
    loss = criterion(torch.flatten(output), torch.flatten(adj_norm))
    loss.backward()
    optimizer.step()


    print('Epoch: {:04d}'.format(epoch+1),
          'loss: {:.4f}'.format(loss.item()),
          'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))