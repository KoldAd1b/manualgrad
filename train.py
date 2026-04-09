"""
Training a tiny GPT type model very close to the MinGPT implementation from Karpathy!
https://github.com/karpathy/minGPT
"""
import cupy as cp
import numpy as np
from tqdm import tqdm
import requests
import pickle
import os
import threading

from model import GPT2Config, get_gpt2
import nn
import optim

### Detect GPUs ###
num_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Found {num_gpus} GPU(s) — using all of them!")

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Dataset length: {len(text)} characters")

### Create A Tokenizer ###
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

### Save tokenizers ###
os.makedirs("work_dir", exist_ok=True)
with open("work_dir/char_to_idx.pkl", "wb") as f:
    pickle.dump(char_to_idx, f)
with open("work_dir/idx_to_char.pkl", "wb") as f:
    pickle.dump(idx_to_char, f)

### Tokenize Data — keep a copy on each GPU ###
data_shards = []
for gpu_idx in range(num_gpus):
    with cp.cuda.Device(gpu_idx):
        data_shards.append(cp.array([char_to_idx[ch] for ch in text]))

### Get Batches ###
def get_batch(data, batch_size, seq_len):
    idx = cp.random.randint(0, len(data) - seq_len - 1, batch_size).tolist()
    inputs = cp.stack([data[i:i+seq_len] for i in idx])
    targets = cp.stack([data[i+1:i+seq_len+1] for i in idx])
    return inputs, targets.flatten()

### Create one model replica per GPU ###
config = GPT2Config()
models = []
loss_fns = []
causal_masks = []
for gpu_idx in range(num_gpus):
    with cp.cuda.Device(gpu_idx):
        models.append(get_gpt2(config))
        loss_fns.append(nn.CrossEntropyLoss())
        causal_masks.append(
            cp.triu(cp.ones((1, 1, config.max_seq_len, config.max_seq_len)) * -cp.inf, k=1)
        )

### Optimizer lives on GPU 0 ###
optimizer = optim.AdamOptimizer(models[0].parameters(), lr=0.0001)

### Sync params from GPU 0 → all replicas ###
def sync_params():
    src_params = []
    for p in models[0].parameters():
        arr = p.params.get()
        pinned = cp.cuda.alloc_pinned_memory(arr.nbytes)
        buf = np.frombuffer(pinned, dtype=arr.dtype)[:arr.size].reshape(arr.shape)
        buf[:] = arr
        src_params.append(buf)
    for gpu_idx in range(1, num_gpus):
        with cp.cuda.Device(gpu_idx):
            for i, param in enumerate(models[gpu_idx].parameters()):
                param.params[:] = cp.asarray(src_params[i])

### Zero grads on all replicas ###
def zero_all_grads():
    for gpu_idx in range(num_gpus):
        for param in models[gpu_idx].parameters():
            param._zero_grad()

### All-reduce: accumulate grads from replicas onto GPU 0, then average ###
def allreduce_grads():
    src = models[0].parameters()
    for gpu_idx in range(1, num_gpus):
        for i, param in enumerate(models[gpu_idx].parameters()):
            arr = param.grad.get()
            pinned = cp.cuda.alloc_pinned_memory(arr.nbytes)
            buf = np.frombuffer(pinned, dtype=arr.dtype)[:arr.size].reshape(arr.shape)
            buf[:] = arr
            with cp.cuda.Device(0):
                src[i].grad += cp.asarray(buf)
    with cp.cuda.Device(0):
        for param in src:
            param.grad /= num_gpus

### Per-GPU forward + backward worker ###
results = [None] * num_gpus

def run_on_gpu(gpu_idx, inputs, targets):
    with cp.cuda.Device(gpu_idx):
        logits = models[gpu_idx].forward(inputs, causal_masks[gpu_idx])
        loss = loss_fns[gpu_idx].forward(y_true=targets, logits=logits)
        loss_grad = loss_fns[gpu_idx].backward()
        models[gpu_idx].backward(loss_grad)
        results[gpu_idx] = (float(loss), logits)

### Training ###
for m in models:
    m.train()

train_iterations = 40000
batch_size_per_gpu = 64
total_batch_size = batch_size_per_gpu * num_gpus
print(f"Effective batch size: {total_batch_size} ({batch_size_per_gpu} per GPU)")

for epoch in tqdm(range(train_iterations)):

    ### Sync params to all replicas ###
    sync_params()
    zero_all_grads()

    ### Get full batch on GPU 0, split into per-GPU shards ###
    inputs, targets = get_batch(data_shards[0], total_batch_size, config.max_seq_len)

    ### Launch one thread per GPU ###
    threads = []
    for gpu_idx in range(num_gpus):
        start = gpu_idx * batch_size_per_gpu
        end   = start + batch_size_per_gpu
        with cp.cuda.Device(gpu_idx):
            inp_shard = cp.asarray(inputs[start:end])
            tgt_shard = cp.asarray(targets[start * config.max_seq_len : end * config.max_seq_len])
        t = threading.Thread(target=run_on_gpu, args=(gpu_idx, inp_shard, tgt_shard))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    ### All-reduce grads onto GPU 0 ###
    allreduce_grads()

    ### Gradient Clipping ###
    for param in models[0].parameters():
        cp.clip(param.grad, -1.0, 1.0, out=param.grad)

    ### Update GPU 0 model ###
    optimizer.step()

    if epoch % 500 == 0:
        loss_avg = sum(r[0] for r in results) / num_gpus
        logits0  = results[0][1]
        targets0 = targets[:batch_size_per_gpu * config.max_seq_len]
        preds    = cp.argmax(logits0, axis=-1)
        accuracy = cp.mean(preds == targets0) * 100

        print(f"\nEpoch {epoch} | Loss: {loss_avg:.4f} | Accuracy: {accuracy:.2f}%")

        models[0].eval()
        with cp.cuda.Device(0):
            seed = cp.array([[char_to_idx['h']]])
            generated = [seed.item()]
            mask0 = causal_masks[0]

            for _ in range(config.max_seq_len):
                curr_len = seed.shape[1]
                logits = models[0].forward(seed, mask0[:, :, :curr_len, :curr_len])
                last_logits = logits[-1]
                probs = cp.exp(last_logits - cp.max(last_logits))
                probs /= cp.sum(probs)
                next_token = cp.random.choice(vocab_size, size=1, p=probs)[0].get().item()
                generated.append(next_token)
                seed = cp.array(generated).reshape(1, -1)

        print("".join([idx_to_char[i] for i in generated]))
        models[0].train()

models[0].save("work_dir/character_transformer.npz")
