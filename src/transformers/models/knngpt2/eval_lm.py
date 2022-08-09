from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import faiss
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
from .setparser import get_parser
import time
import scipy

args=get_parser
subset=args.gen_subset
dstore=args.save_knnlm_dstore
device = "cuda"
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
device = "cuda"
subset= load_dataset("wikitext", "wikitext-103-v1", split=str(subset))
encodings = tokenizer("\n\n".join(subset["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 512
nlls = []

if dstore:
    dstore_keys = np.memmap(args.dstore_mmap + '_keys.npy', dtype=np.float16, mode='w+',
                            shape=(args.dstore_size, args.decoder_embed_dim))
    dstore_vals = np.memmap(args.dstore_mmap + '_vals.npy', dtype=np.int16, mode='w+', shape=(args.dstore_size, 1))

for i in tqdm(range(0, min(encodings.input_ids.size(1), args.dstore_size), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1), args.dstore_size)
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)

    if dstore:

    #dstore_keys[begin_loc:end_loc] = outputs.logits[0].view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        dstore_keys[begin_loc:end_loc] = outputs.what_i_needs.view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)

        dstore_vals[begin_loc:end_loc] = target_ids.view(
            -1, 1).cpu().numpy().astype(np.int16)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)