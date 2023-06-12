import torch

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.count_nonzero(preds == labels) / preds.shape[0]


def get_model_gpu_memory(model):
    dummy_input = torch.randn(10, 3, 512, 512, dtype=torch.half).cuda()
    torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    memory_cached = torch.cuda.memory_cached() / 1024 ** 2
    return memory_allocated, memory_cached