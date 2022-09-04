from transformers import GPT2LMHeadModel2,GPT2TokenizerFast,GPT2LMHeadModel
import faiss
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import time
import scipy
import argparse
import sys
import torch
import faiss
import math
import numpy as np
import time
sys.path.insert(2,"./")
from transformers.models.knngpt2.knnlm import KNN_Dstore
from transformers.models.knngpt2.setparser import get_parser
import pdb

args=get_parser()
subset=args.gen_subset
dstore=args.save_knnlm_dstore
knnlm=args.knnlm
device = "cuda"
model_id = "gpt2"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if knnlm:
    model = GPT2LMHeadModel2.from_pretrained(model_id).to(device)
else:
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
device = "cuda"
subset= load_dataset("wikitext", "wikitext-103-v1", split=str(subset))
encodings = tokenizer("\n\n".join(subset["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 512
nlls = []

if dstore:
    dstore_keys = np.memmap( args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+',shape=(args.dstore_size, args.decoder_embed_dim))
    dstore_vals = np.memmap( args.dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))
#pdb.set_trace()
index=0
for i in tqdm(range(0, min(encodings.input_ids.size(1), args.dstore_size), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1)-1, args.dstore_size)
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        #pdb.set_trace()
        neg_log_likelihood = outputs[0] * trg_len
        neg_log_likelihood=torch.tensor(neg_log_likelihood, dtype=torch.float32)

    nlls.append(neg_log_likelihood)

    if dstore:
        dstore_keys[index:index+trg_len] = outputs.what_i_need[:,-trg_len:].view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        target_ids = encodings.input_ids[:, min(begin_loc+1,encodings.input_ids.size(1)):min(end_loc+1,encodings.input_ids.size(1))].to(device)
        dstore_vals[index:index+trg_len] = target_ids[:,-trg_len:].view(-1, 1).cpu().numpy().astype(np.int)
    #pdb.set_trace()
    index+=trg_len
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
pdb.set_trace()
print(ppl)
