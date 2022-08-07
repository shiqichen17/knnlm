from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import faiss
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
from setparser import get_parser
import time
import scipy

def save_key_val(model,tokenizer,args):
    device = "cuda"

    train = load_dataset("wikitext", "wikitext-103-v1", split="train")
    encodings = tokenizer("\n\n".join(train["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    nlls = []
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

        dstore_keys[begin_loc:end_loc] = outputs.logits[0].view(
            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        dstore_vals[begin_loc:end_loc] = target_ids.view(
            -1, 1).cpu().numpy().astype(np.int16)
    #ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

def test(model,tokenizer,args):
    device = "cuda"

    test = load_dataset("wikitext", "wikitext-103-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    nlls = []

    query=np.empty(encodings.input_ids.size(1),args.decoder_embed_dim)
    prob=np.empty(encodings.input_ids.size(1),args.decoder_embed_dim)
    tgt=np.empty(encodings.input_ids.size(1),1)


    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

        query[begin_loc:end_loc] = outputs.logits[0].view(
            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        prob[begin_loc:end_loc] = torch.softmax(outputs.logits[0],-1).view(
            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
        tgt[begin_loc:end_loc] = target_ids.view(
            -1, 1).cpu().numpy().astype(np.int16)

    return query,prob,tgt



    #ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    return curr_prob

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index = self.setup_faiss(args)


    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        print('Keys are fp16 and vals are int16')

        self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, 1))

        return index


    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt):
        def dist_func(d, k, q, function=None):
            if not function:
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                    return -1 * l2
                return d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx])
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries, function=self.sim_func)
        probs = utils.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

def recal(args,prob):
    device = "cuda"
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    device = "cuda"

    test = load_dataset("wikitext", "wikitext-103-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

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
            #outputs = model(input_ids, labels=target_ids)
            outputs=logit(prob)
            neg_log_likelihood = outputs * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl
def main(args):
    device = "cuda"
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    save_key_val(model,tokenizer,args)
    knn_dstore=KNN_Dstore(args)
    query,prob,tgt=test(model,tokenizer,args)
    yhat_knn_prob = knn_dstore.get_knn_log_prob(query,tgt)
    #yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
    if args.fp16:
        yhat_knn_prob = yhat_knn_prob.half()
        prob = prob.half()

    prob = combine_knn_and_vocab_probs(yhat_knn_prob, prob, args.lmbda)
    ppl=recal(args,prob)
    print(ppl)

def cli_main():
    args = get_parser()
    main(args)


if __name__ == '__main__':
    cli_main()