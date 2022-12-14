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
for i in tqdm(range(0, min(encodings.input_ids.size(1), args.dstore_size), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1), args.dstore_size)
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        if knnlm:
            knnlm_dstore= KNN_Dstore(args)
            #pdb.set_trace()
            outputs= model(input_ids,labels=target_ids,knnlm=knnlm_dstore,fp16=args.fp16,labda=args.labda,tok=tokenizer)
            #pdb.set_trace()
        else:
            outputs = model(input_ids, labels=target_ids)
        #pdb.set_trace()
        neg_log_likelihood = outputs[0] * trg_len
        neg_log_likelihood=torch.tensor(neg_log_likelihood, dtype=torch.float32)

    nlls.append(neg_log_likelihood)
    #pdb.set_trace()
   
       
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
#pdb.set_trace()
print(ppl)
