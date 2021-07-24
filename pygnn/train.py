from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import load_data,normalize,toy_data,norm_embed,nmi_score,svdApprox
from models import GNN

torch.set_printoptions(sci_mode=False)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=426, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
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

fixed=False

# Load data
adj = load_data(daily=False)
#adj = toy_data()

adj_norm = normalize(adj)

adj = torch.FloatTensor(np.array(adj))
adj_norm = torch.FloatTensor(np.array(adj_norm))

# loss function
criterion = torch.nn.GaussianNLLLoss()

# features = torch.FloatTensor(torch.eye(adj.shape[1]))
# features = features.reshape((1,adj.shape[1],adj.shape[1]))
# features = features.repeat(adj.shape[0], 1, 1)

# NULL Model
mu0 = adj.mean() * torch.ones(adj.shape[1:])
sigma0 = adj.std() * torch.ones(adj.shape[1:])
with torch.no_grad():
    loss0 = criterion(torch.flatten(adj), torch.flatten(mu0), torch.flatten(torch.square(sigma0)))

#svd features
svd_mu,svd_loss,svdembedx,svdembedy = svdApprox(adj=adj,dim=dim)
features = torch.cat((svdembedx,svdembedy),dim=1)
if not fixed:
    mse = torch.nn.MSELoss()
    mseloss = mse(torch.flatten(svd_mu),torch.flatten(adj))
    sig = torch.sqrt(mseloss)/args.ndim
    ones = torch.ones(features.size())*torch.sqrt(sig)
    features = torch.cat((features,ones),dim=1)
features = features.unsqueeze(dim=0)

# Model and optimizer
"""model = GNN(batch_size=adj.shape[0],
            nfeat=features.shape[2],
            nhid=args.hidden*args.ndim,
            ndim=args.ndim,
            mu0=adj.mean(),
            sigma0=adj.std(),
            fixed=fixed)
"""

model = GNN(batch_size=adj.shape[0],
            nfeat=adj.shape[1],
            nhid=adj.shape[1],
            ndim=args.ndim,
            mu0=adj.mean(),
            sigma0=adj.std(),
            fixed=fixed)

"""activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.embeddings.register_forward_hook(get_activation('embeddings'))"""

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_norm = adj_norm.cuda()

# Train model
t_total = time.time()

# NULL Model
mu0 = adj.mean() * torch.ones(adj.shape[1:])
sigma0 = adj.std() * torch.ones(adj.shape[1:])
with torch.no_grad():
    loss0 = criterion(torch.flatten(adj), torch.flatten(mu0), torch.flatten(torch.square(sigma0)))

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


for epoch in range(args.epochs):

    t = time.time()
    model.train()
    optimizer.zero_grad()

    if fixed:
        mu = model(features, adj_norm)
        with torch.no_grad():
            mse = torch.nn.MSELoss()
            mseloss = mse(torch.flatten(mu), torch.flatten(adj))
            sig = torch.sqrt(mseloss)
        sigma = sig * torch.ones(adj.shape, requires_grad=True)
    else:
        mu, sigma = model(features, adj_norm)

    loss = criterion(torch.flatten(adj), torch.flatten(mu), torch.flatten(torch.square(sigma)))
    loss.backward()

    optimizer.step()

    if epoch == 0:
        best_loss = loss
    else:
        if loss < best_loss:
            best_loss = loss

    if epoch % 100 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.8f}'.format(best_loss.item()),
              'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
