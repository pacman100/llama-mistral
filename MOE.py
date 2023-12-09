# coding: utf-8

# In[1]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from contextlib import contextmanager
from llama import Llama
import torch


@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None
    saved_inits = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )  # saving
    torch.nn.init.kaiming_uniform_ = (
        torch.nn.init.uniform_
    ) = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = saved_inits  # restoring


with suspend_nn_inits():
    generator = Llama.build(
        ckpt_dir="/raid/sourab/mixtral-8x7b-32kseqlen/",
        tokenizer_path="/raid/sourab/mixtral-8x7b-32kseqlen/tokenizer.model",
        max_seq_len=128,
        max_batch_size=4,
        num_gpus=2,
        on_cpu=False,
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
