import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
from model import *
from train import *
from attack import *
from torch.utils.data import ConcatDataset

# def get_entropy(logits):
#     probs = torch.nn.functional.softmax(logits.detach(), dim=1)
#     return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().detach()

def get_entropy(logits):
    probs = torch.nn.functional.softmax(logits.detach(), dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).detach()  

def compute_entropies_batched(model, loader, batch_size=256, device="cuda"):
    model.eval()
    all_entropies = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            entropies = get_entropy(logits)
            # print(entropies.shape)
            all_entropies.append(entropies.cpu())
    return torch.cat(all_entropies).numpy()

def count_a_in_top_c(a, b, c):
    a = np.array(a)
    b = np.array(b)

    combined = np.concatenate([a, b])

    # Get indices of top-c values
    top_c_indices = np.argpartition(combined, -c)[-c:]

    # Count how many indices belong to 'a'
    count = np.sum(top_c_indices < len(a))

    return count

parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--proxy_model')
parser.add_argument('--victim_model')
parser.add_argument('--total_dataset_size',type=int)
parser.add_argument('--poison_dataset_size',type=int)
# parser.add_argument('--coreset_size',type=int)
parser.add_argument('--attack_mode',default="replace")
parser.add_argument('--proxy_epochs',type=int)
parser.add_argument('--victim_epochs',type=int)
parser.add_argument('--attack_name')
parser.add_argument('--max_perturbation')
parser.add_argument('--batch_size',type=int,default=32)
args = parser.parse_args()
device='cuda'
is_proxy_simple=True
is_victim_simple=True
if args.dataset=="MNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
elif args.dataset=="FashionMNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
elif args.dataset=="CIFAR10":
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)


if args.proxy_model=="SimpleNN":
    proxy_model=SimpleNN((3,32,32)).to(device)
elif args.proxy_model=="ResNet":
    proxy_model=ResNet18(10,3).to(device)
    is_proxy_simple=False


if args.victim_model=="SimpleNN":
    victim_model=SimpleNN((3,32,32)).to(device)
elif args.victim_model=="ResNet":
    victim_model=ResNet18(10,3).to(device)
    is_victim_simple=False


total_pool,_ = torch.utils.data.random_split(trainset, [args.total_dataset_size, len(trainset)-args.total_dataset_size],generator=torch.Generator().manual_seed(args.seed))

if args.attack_mode=="replace":
    attack_pool,rest=torch.utils.data.random_split(total_pool, [args.poison_dataset_size, len(total_pool)-args.poison_dataset_size],generator=torch.Generator().manual_seed(args.seed))
elif args.attack_mode=="insert":
    pass

print(len(attack_pool),len(total_pool),len(rest))

pool_dataloader=torch.utils.data.DataLoader(total_pool, batch_size=args.batch_size, shuffle=True)
proxy_model=train_model(proxy_model,pool_dataloader,args.proxy_epochs)
print(eval_model(proxy_model,testloader))
attack_pool_dataloader=torch.utils.data.DataLoader(attack_pool, batch_size=args.batch_size, shuffle=True)
rest_dataloader=torch.utils.data.DataLoader(rest, batch_size=args.batch_size, shuffle=True)
before_entropies_attack_pool=compute_entropies_batched(proxy_model,attack_pool_dataloader)
before_entropies_rest_pool=compute_entropies_batched(proxy_model,rest_dataloader)
percentages=[1,5,10,20,40,60]
for percentage in percentages:
    print(100*count_a_in_top_c(before_entropies_attack_pool,before_entropies_rest_pool,int(args.total_dataset_size*percentage/100))/args.poison_dataset_size,end=", ")

print("\n*********")
def resolve_index(subsett, ii):
    idx = ii
    while isinstance(subsett, torch.utils.data.Subset):
        idx = subsett.indices[idx]
        subsett = subsett.dataset
    return subsett, idx  # base dataset + final index

# for i in range(len(attack_pool)):
#     plt.imsave(f"b{i}.png",attack_pool[i][0].squeeze().cpu().detach().numpy())

for i in range(len(attack_pool)):
    x_adv=pickme_attack(proxy_model,attack_pool[i][0].cuda(),is_proxy_simple)
    orig_dataset, final_idx = resolve_index(attack_pool, i)
    # plt.imsave("b-cifar.png",orig_dataset.data[final_idx])
    orig_dataset.data[final_idx] = (x_adv.squeeze()*255).to(torch.uint8).detach().permute(1, 2, 0).cpu()
    # plt.imsave("a-cifar.png",orig_dataset.data[final_idx])

poisoned_pool=ConcatDataset([attack_pool,rest])
poisoned_pool_dataloader=torch.utils.data.DataLoader(poisoned_pool, batch_size=args.batch_size, shuffle=True)
victim_model=train_model(victim_model,poisoned_pool_dataloader,args.victim_epochs)

attack_pool_dataloader=torch.utils.data.DataLoader(attack_pool, batch_size=args.batch_size, shuffle=True)
after_entropies_attack_pool=compute_entropies_batched(victim_model,attack_pool_dataloader)
after_entropies_rest_pool=compute_entropies_batched(victim_model,rest_dataloader)
# print(after_entropies_attack_pool,before_entropies_attack_pool)

percentages=[1,5,10,20,40,60]
for percentage in percentages:
    print(100*count_a_in_top_c(after_entropies_attack_pool,after_entropies_rest_pool,int(args.total_dataset_size*percentage/100))/args.poison_dataset_size,end=", ")
print("\n---------------------------------")