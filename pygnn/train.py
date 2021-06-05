from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import load_data,normalize,toy_data,norm_embed,nmi_score
from models import GNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=426, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=10e-8,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--ndim', type=int, default=2,
                    help='Embeddings dimension.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj = load_data(daily=True)
#adj = toy_data()

adj_norm = normalize(adj)

adj = torch.FloatTensor(np.array(adj))
adj_norm = torch.FloatTensor(np.array(adj_norm))

# Model and optimizer
model = GNN(batch_size=adj_norm.shape[0],
            nfeat=adj_norm.shape[1],
            nhid=args.hidden,
            ndim=2*args.ndim)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.embeddings.register_forward_hook(get_activation('embeddings'))
model.reconstructions.register_forward_hook(get_activation('reconstructions'))


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



features = torch.FloatTensor(torch.eye(adj.shape[1]))
features = features.reshape((1,adj.shape[1],adj.shape[1]))
features = features.repeat(adj.shape[0], 1, 1)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_norm = adj_norm.cuda()


# Train model
t_total = time.time()
A2norm = (adj_norm ** 2).mean()
#A2norm = (adj_norm ** 2).view(adj_norm.shape[0],-1).mean(axis=1)

for epoch in range(args.epochs):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_norm)

    embed = activation['embeddings']
    pred = activation['reconstructions']

    embedd = norm_embed(embed)
    embedx, embedy = torch.chunk(embedd, chunks=2, dim=2)

    # loss function
    criterion = torch.nn.MSELoss()
    # regularization
    reg_criterion = torch.nn.L1Loss()
    reg_loss = reg_criterion((embedx ** 2).sum(axis=1), (embedy ** 2).sum(axis=1))

    loss = criterion(torch.flatten(output).reshape(365,-1), torch.flatten(adj_norm).reshape(365,-1)) / A2norm
    loss.backward()
    optimizer.step()

    if epoch == 0:
        best_loss = loss
        best_rl = reg_loss
        best_embed = embedd
        best_pred = pred
    else:
        if loss < best_loss:
            best_loss = loss
            best_embed = embedd
            best_pred = pred
            best_rl = reg_loss
        elif loss == best_loss and reg_loss < best_rl:
            best_loss = loss
            best_embed = embedd
            best_pred = pred
            best_rl = reg_loss

    if epoch % 100 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.8f}'.format(best_loss.item()),
              'reg_loss: {:.8f}'.format(best_rl.item()),
              'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


