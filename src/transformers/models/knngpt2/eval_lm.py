from transformers import GPT2LMHeadModel2,GPT2TokenizerFast
import faiss
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import time
import scipy
import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--probe', default=8, type=int,
                        help='for FAISS, the number of lists to query')
    parser.add_argument('--k', default=32, type=int,
                        help='number of nearest neighbors to retrieve')
    parser.add_argument('--dstore_size', default=103227021, type=int,
                        help='number of items in the knnlm datastore')
    parser.add_argument('--dstore_filename', type=str, default=None,
                        help='File where the knnlm datastore is saved')
    parser.add_argument('--indexfile', type=str, default=None,
                        help='File containing the index built using faiss for knn')
    parser.add_argument('--lmbda', default=0.0, type=float,
                        help='controls interpolation with knn, 0.0 = no knn')
    parser.add_argument('--knn_sim_func', default=None, type=str,
                        help='similarity function to use for knns')
    parser.add_argument('--faiss_metric_type', default='l2', type=str,
                        help='the distance metric for faiss')
    parser.add_argument('--no_load_keys', default=False, action='store_true',
                        help='do not load keys')
    parser.add_argument('--dstore_fp16', default=True, action='store_true',
                        help='if true, datastore items are saved in fp16 and int16')
    parser.add_argument('--move_dstore_to-mem', default=False, action='store_true',
                        help='move the keys and values for knn to memory')
    parser.add_argument('--decoder_embed_dim', default=1024, action='store_true',
                        help='move the keys and values for knn to memory')
    parser.add_argument('--gen_subset', default='test', metavar='SPLIT',
                       help='data subset to generate (train, valid, test)')
    parser.add_argument('--output_word_probs', action='store_true',
                       help='if set, outputs words and their predicted log probabilities to standard output')
    parser.add_argument('--output_word_stats', action='store_true',
                       help='if set, outputs word statistics such as word count, average probability, etc')
    parser.add_argument('--context_window', default=0, type=int, metavar='N',
                       help='ensures that every evaluated token has access to a context of at least this size,'
                            ' if possible')
    parser.add_argument('--softmax_batch', default=sys.maxsize, type=int, metavar='N',
                       help='if BxT is more than this, will batch the softmax over vocab to this amount of tokens'
                            ' in order to fit into GPU memory')
    parser.add_argument('--lm_eval', default=True, action='store_true',
                       help='helpful for certain ops that are only used during eval')
    parser.add_argument('--knnlm', action='store_true',
                       help='use the k-nearest neighbors language model')
    parser.add_argument('--save_knnlm_dstore', action='store_true',
                       help='save keys for the knnlm datastore')
    parser.add_argument('--dstore_mmap', default=None, type=str,
                       help='If saving knnlm dstore, save keys and values to this file')
    parser.add_argument('--fp16',default=True, action='store_true',
                        help='save keys for the knnlm datastore')
    parser.add_argument('--labda', default=0.25, action='store_true',
                        help='save keys for the knnlm datastore')
    args = parser.parse_args()
    return args

args=get_parser()
subset=args.gen_subset
dstore=args.save_knnlm_dstore
knnlm=args.knnlm
device = "cuda"
model_id = "gpt2"
#model = GPT2LMHeadModel2.from_pretrained(model_id).to(device)
model = GPT2LMHeadModel2.from_pretrained(model_id).to(device)
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
        if knnlm:
            outputs=model(input_ids,labels=target_ids,args,fp16=args.fp16,labda=args.labda)
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)

    if dstore:

    #dstore_keys[begin_loc:end_loc] = outputs.logits[0].view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        dstore_keys[begin_loc:end_loc] = outputs.what_i_need.view( -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)

        dstore_vals[begin_loc:end_loc] = target_ids.view(
            -1, 1).cpu().numpy().astype(np.int16)
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl)