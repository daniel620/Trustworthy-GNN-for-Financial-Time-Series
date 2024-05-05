import torch

def integrated_gradients(input, edge_index, baseline, model, steps=50, node_index=0):
    input = input.clone().detach().requires_grad_(True)
    baseline = baseline.clone().detach()

    interpolated = [baseline + (step / steps) * (input - baseline) for step in range(0, steps + 1)]

    grads = []
    for i, inp in enumerate(interpolated):
        if inp.grad is not None:
            inp.grad.zero_()
        inp.requires_grad_(True)
        with torch.autograd.set_grad_enabled(True):
            output = model(inp, edge_index)
            target_score = output[node_index] # target score at the node of interest
            grads.append(torch.autograd.grad(target_score, inp)[0])

    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grad = (input - baseline)[node_index] * avg_grads  # element-wise multiplication
    return integrated_grad
