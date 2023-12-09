#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from contextlib import contextmanager
from llama import Llama
import torch
@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring

with suspend_nn_inits():
    generator = Llama.build(
            ckpt_dir="/raid/sourab/mixtral-8x7b-32kseqlen/",
            tokenizer_path="/raid/sourab/mixtral-8x7b-32kseqlen/tokenizer.model",
            max_seq_len=128,
            max_batch_size=4,
            num_gpus=2,
        )


# In[2]:


params = 0
for p in generator.model.parameters():
    params += p.numel()

params


# In[3]:


model = generator.model


# In[4]:


model.tok_embeddings.weight


# In[5]:


import bitsandbytes as bnb
from llama.model import MoE
import torch


def replace_remaining_with_4bit(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, bnb.nn.modules.LinearSparse) and "output" not in name and "gate" not in name:
            weight = module.weight.data
            in_features = module.in_features
            out_features = module.out_features
            if module.bias is not None:
                    bias = module.bias
            module = bnb.nn.Linear4bit(
                            in_features,
                            out_features,
                            module.bias is not None,
                        )
            module.weight = bnb.nn.Params4bit(weight, requires_grad=False)
            if module.bias is not None:
                    module.bias = bias
            model._modules[name] = module
            
        if len(list(module.children())) > 0:
            replace_remaining_with_4bit(module)

    
    

def replace_moe_with_sparse_linear_and_remaining_with_4bit(model):
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            for expert in module.experts:
                for name, layer in expert.named_children():
                    weight = layer.weight.data
                    if layer.bias is not None:
                        bias = layer.bias
                    in_features = layer.in_features
                    out_features = layer.out_features
                    layer = bnb.nn.modules.LinearSparse(in_features, out_features, layer.bias is not None, sparsity_level=0.97)
                    layer.weight = bnb.nn.modules.ParamsSparse(weight, sparsity_level=0.97)
                    if layer.bias is not None:
                        layer.bias = bias
                    setattr(expert, name, layer)
    replace_remaining_with_4bit(model)
                


# In[6]:


get_ipython().run_cell_magic('time', '', 'with suspend_nn_inits():\n    replace_moe_with_sparse_linear_and_remaining_with_4bit(model)\n')


# In[7]:


model


# In[8]:


model.to("cuda")
model.eval()


# In[9]:


prompts = [
# For these prompts, the expected answer is the natural continuation of the prompt
"Mistral.ai is a company that",
"Simply put, the theory of relativity states that ",
"""A brief message congratulating the team on the launch:

Hi everyone,

I just """,
# Few shot prompt (providing a few examples before asking model to complete more);
"""Translate English to French:

sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>""",
]
with torch.autocast(dtype=torch.float16, device_type="cuda"):
    results = generator.text_completion(
    prompts,
    max_gen_len=64,
    temperature=0.2,
    top_p=0.95,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


# In[ ]:




