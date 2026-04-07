from tqdm import tqdm
import torch
import torch.nn.functional as F

def pickme_attack(model, x, is_simple=True, epsilon=0.3, alpha=0.01, iters=100):
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(iters):
        logits=None
        if is_simple:
            logits = model(x_adv)
        else:
            logits=model (x_adv.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()  # Compute entropy

        # Compute gradient
        entropy.backward()
        grad = x_adv.grad.data

        # Perform PGD step
        x_adv = x_adv + alpha * grad.sign()

        # Project back into the epsilon-ball
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid pixel range [0,1]

        # Detach and re-enable gradient tracking for next step
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()