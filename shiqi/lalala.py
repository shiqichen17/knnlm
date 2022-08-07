import numpy as np

dstore_keys = np.memmap('_keys.npy', dtype=np.float16, mode='w+', shape=(1000, 50257))
dstore_vals = np.memmap('_vals.npy', dtype=np.int16, mode='w+', shape=(1000, 1))
dstore_idx = 0
for i in tqdm(range(0, min(encodings.input_ids.size(1), 1000), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1), 1000)
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)
    '''
    hypos.append([{
        'keys': outputs.logits[0],
        'vals':target_ids

    }])
   '''
    dstore_keys[begin_loc:end_loc] = outputs.logits[0].view(
        -1, 50257).cpu().numpy().astype(np.float16)
    dstore_vals[begin_loc:end_loc] = target_ids.view(
        -1, 1).cpu().numpy().astype(np.int16)

