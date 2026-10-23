import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MAX_TOK = 510
STATE = {"oom": 0}
log = print

def spaced(hexstr): return " ".join(hexstr[i:i + 2] for i in range(0, len(hexstr), 2))


def word_ids_array(wid):
    return np.fromiter((-1 if x is None else x for x in wid), dtype=np.int32, count=len(wid))


# Select the longest whole-byte prefix that fits the token budget.
def cut_by_tokens(counts, budget):

    csum = np.cumsum(counts)
    k = int(np.searchsorted(csum, budget, side="right"))
    return k, (int(csum[k - 1]) if k > 0 else 0)


# Build token and byte pooling weights while excluding special tokens and padding.
def pool_weights(wid):

    w = word_ids_array(wid)
    valid = w >= 0
    n_tok = int(valid.sum())
    wt = np.zeros(len(w), np.float32); wb = np.zeros(len(w), np.float32)
    if n_tok == 0: return wt, wb
    wt[valid] = 1.0 / n_tok
    cnt = np.bincount(w[valid])
    n_bytes = int((cnt > 0).sum())
    wb[valid] = 1.0 / (cnt[w[valid]] * n_bytes)
    return wt, wb


# Group sessions by length and attention mode within the batch token budget.
def make_batches(seq_lens, order, budget, max_batch, thr, blk=0):

    def key(L): return 0 if L <= thr else (-(-L // blk) if blk else 1)
    batches, cur, cur_max, cur_key = [], [], 0, None
    for i in order:
        L = seq_lens[i]
        if cur:
            if key(L) != cur_key or max(cur_max, L) * (len(cur) + 1) > budget or len(cur) >= max_batch:
                batches.append(cur); cur, cur_max = [], 0
        if not cur: cur_key = key(L)
        cur.append(i); cur_max = max(cur_max, L)
    if cur: batches.append(cur)
    return batches

# Record the bytes and tokens retained from each session payload.
def build_index(tok, hexes, budget, thr, log, name):
    n_s = len(hexes)
    rows = []; hist = np.zeros((n_s, 256), np.float32)
    for s in range(0, n_s, 200):
        part = hexes[s:s + 200]
        enc = tok([spaced(h) for h in part], add_special_tokens=False, truncation=False, verbose=False)
        for j, h in enumerate(part):
            n = len(h) // 2
            wid = word_ids_array(enc.word_ids(j))
            cnt = np.bincount(wid[wid >= 0], minlength=n)
            if len(cnt) != n or int(cnt.sum()) != len(enc["input_ids"][j]):
                raise SystemExit("token bookkeeping mismatch in %s session row %d" % (name, s + j))
            k, tt = cut_by_tokens(cnt, budget)
            b = np.frombuffer(bytes.fromhex(h[:2 * k]), dtype=np.uint8)
            hist[s + j] = np.bincount(b, minlength=256) / max(k, 1)
            rows.append((n, k, tt, tt + 2))
        if (s // 200) % 10 == 0: log("%s: %d / %d sessions tokenized" % (name, min(s + 200, n_s), n_s))
    idx = pd.DataFrame(rows, columns=["n_bytes_in", "bytes_used", "tokens_used", "seq_len"])
    idx["cut"] = (idx.bytes_used < idx.n_bytes_in).astype(int)
    idx["attn"] = np.where(idx.seq_len <= thr, "full", "sparse")
    return idx, hist

# Embed a batch and pool its token vectors into session features.
def run_batch(texts, exp_len):
    enc = tok(texts, padding=True, truncation=False, return_tensors="pt", verbose=False)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    B, L = ids.shape
    got = mask.sum(1).tolist()
    if got != list(exp_len): raise SystemExit("token count differs from the index: %s vs %s" % (got[:5], list(exp_len)[:5]))
    Wt = np.zeros((B, L), np.float32); Wb = np.zeros((B, L), np.float32)
    for i in range(B):
        wt, wb = pool_weights(enc.word_ids(i)); Wt[i, :len(wt)] = wt; Wb[i, :len(wb)] = wb
    sparse = L > THR
    assert (min(exp_len) > THR) == sparse and (max(exp_len) <= THR) == (not sparse), "batch straddles the attention threshold"
    if sparse: assert len({-(-x // BLK) for x in exp_len}) == 1, "sparse batch mixes different block counts"
    model = m_sparse if sparse else m_full
    with torch.inference_mode():
        H = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
        cls = H[:, 0]
        tokm = torch.einsum("bl,bld->bd", torch.from_numpy(Wt).to(dev), H)
        bytem = torch.einsum("bl,bld->bd", torch.from_numpy(Wb).to(dev), H)
        res = [x.float().cpu().numpy() for x in (cls, tokm, bytem)]
    return res

# Retry a batch with smaller groups when device memory is insufficient.
def run_safe(texts, exp_len):
    try:
        return run_batch(texts, exp_len)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); STATE["oom"] += 1
        if len(texts) == 1: raise
        h = len(texts) // 2
        log("out of memory with %d sessions - splitting" % len(texts))
        a = run_safe(texts[:h], exp_len[:h]); b = run_safe(texts[h:], exp_len[h:])
        return [np.vstack([x, y]) for x, y in zip(a, b)]

def token_counts_per_byte(word_ids, n_bytes):

    w = np.asarray(word_ids, dtype=np.int64)
    if len(w) == 0 or w.min() < 0: raise ValueError("token without a byte")
    cnt = np.bincount(w, minlength=n_bytes)
    if len(cnt) != n_bytes or (cnt == 0).any(): raise ValueError("token-to-byte mapping is not one word per byte")
    if (np.diff(w) < 0).any(): raise ValueError("word ids not monotone")
    return cnt


# Split a payload at whole-byte boundaries to fit the BERT token limit.
def make_chunks(counts, budget=MAX_TOK):

    csum = np.cumsum(counts); n = len(counts); out = []; b0 = 0; t0 = 0
    while b0 < n:
        b1 = int(np.searchsorted(csum, t0 + budget, side="right"))
        if b1 <= b0: raise ValueError("a single byte needs more than %d tokens" % budget)
        t1 = int(csum[b1 - 1]); out.append((b0, b1, t0, t1)); b0, t0 = b1, t1
    return out


# Check that chunking covers each retained byte exactly once.
def check_chunks(chunks, n_bytes, n_tokens, budget=MAX_TOK):
    ok = chunks[0][0] == 0 and chunks[-1][1] == n_bytes and chunks[0][2] == 0 and chunks[-1][3] == n_tokens
    ok = ok and all(chunks[i][1] == chunks[i + 1][0] and chunks[i][3] == chunks[i + 1][2] for i in range(len(chunks) - 1))
    ok = ok and all(0 < c[3] - c[2] <= budget and c[1] > c[0] for c in chunks)
    if not ok: raise ValueError("chunks do not tile the sequence")


class Embedder:
    def __init__(self, ckpt, dev, torch, tok, model, budget_tokens, max_batch, log):
        self.ckpt, self.dev, self.torch, self.tok, self.model = ckpt, dev, torch, tok, model
        self.budget, self.max_batch, self.log = budget_tokens, max_batch, log
        self.cls, self.sep, self.pad, self.unk = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.unk_token_id
        self.oom = 0

    # Tokenize and partition a session payload for BERT inference.
    def prepare(self, hexstr, n_bytes):

        h = hexstr[:2 * n_bytes].lower()
        if len(h) != 2 * n_bytes: raise SystemExit("payload shorter than bytes_used")
        text = spaced(h)
        enc = self.tok(text, add_special_tokens=False, truncation=False, return_attention_mask=False, verbose=False)
        ids = enc["input_ids"]; wid = enc.word_ids()
        counts = token_counts_per_byte(wid, n_bytes)
        chunks = make_chunks(counts); check_chunks(chunks, n_bytes, len(ids))
        if self.unk in ids: raise SystemExit("unknown token in a payload: %s" % text[:80])
        return np.asarray(ids, np.int64), np.asarray(wid, np.int64), chunks, h

    # Embed a batch and pool its token vectors into session features.
    def run_batch(self, id_lists, w_lists):

        torch = self.torch
        L = max(len(x) for x in id_lists) + 2; B = len(id_lists)
        ids = np.full((B, L), self.pad, np.int64); mask = np.zeros((B, L), np.int64); W = np.zeros((B, L), np.float32)
        for i, (x, w) in enumerate(zip(id_lists, w_lists)):
            n = len(x); ids[i, 0] = self.cls; ids[i, 1:n + 1] = x; ids[i, n + 1] = self.sep; mask[i, :n + 2] = 1
            wid = [None] + w.tolist() + [None]
            _, wb = pool_weights(wid); W[i, :len(wb)] = wb
        with torch.inference_mode():
            H = self.model(input_ids=torch.from_numpy(ids).to(self.dev), attention_mask=torch.from_numpy(mask).to(self.dev)).last_hidden_state
            v = torch.einsum("bl,bld->bd", torch.from_numpy(W).to(self.dev), H).float().cpu().numpy()
        return v

    # Retry a batch with smaller groups when device memory is insufficient.
    def run_safe(self, id_lists, w_lists):
        try:
            return self.run_batch(id_lists, w_lists)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache(); self.oom += 1
            if len(id_lists) == 1: raise
            h = len(id_lists) // 2
            self.log("out of memory with %d chunks - splitting" % len(id_lists))
            return np.vstack([self.run_safe(id_lists[:h], w_lists[:h]), self.run_safe(id_lists[h:], w_lists[h:])])

    # Combine chunk embeddings into one vector per session.
    def embed_sessions(self, hexes, nbytes, rows):

        items = []; stats = {}
        for r in rows:
            ids, wid, chunks, h = self.prepare(hexes[r], nbytes[r])
            stats[r] = dict(row=r, bytes=int(nbytes[r]), tokens=int(len(ids)), chunks=len(chunks),
                            bytes_per_chunk=";".join(str(c[1] - c[0]) for c in chunks), tokens_per_chunk=";".join(str(c[3] - c[2]) for c in chunks))
            if "".join(h[2 * c[0]:2 * c[1]] for c in chunks) != h: raise SystemExit("chunk concatenation does not reproduce the payload (row %d)" % r)
            for k, (b0, b1, t0, t1) in enumerate(chunks):
                items.append((t1 - t0, r, k, b1 - b0, ids[t0:t1], wid[t0:t1] - b0))
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        acc = {r: np.zeros(768, np.float64) for r in rows}
        i = 0
        while i < len(items):
            j = i
            while j < len(items) and (j - i) < self.max_batch and (items[j][0] + 2) * (j - i + 1) <= self.budget: j += 1
            j = max(j, i + 1)
            batch = items[i:j]
            v = self.run_safe([b[4] for b in batch], [b[5] for b in batch])
            for (nt, r, k, nb, _, _), vec in zip(batch, v): acc[r] += vec.astype(np.float64) * (nb / nbytes[r])
            i = j
        return np.stack([acc[r] for r in rows]).astype(np.float32), [stats[r] for r in rows]


# Download pinned checkpoints, or use an explicitly supplied local copy.
def checkpoint(name, settings, local=None):
    if local:
        return Path(local).resolve()
    from huggingface_hub import snapshot_download
    spec = settings['models'][name]
    patterns = list(spec.get('files_sha256', {})) or [
        'config.json', 'pytorch_model.bin', 'spiece.model',
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    folder = ROOT / spec['local_dir']
    snapshot_download(repo_id=spec['repo'], revision=spec['revision'],
                      local_dir=str(folder), allow_patterns=patterns)
    for filename, expected in spec.get('files_sha256', {}).items():
        h = hashlib.sha256()
        with (folder / filename).open('rb') as f:
            for block in iter(lambda: f.read(4 << 20), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('Checkpoint hash mismatch: ' + filename)
    return folder


def archive_frame(path, member):
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        return pd.read_csv(f, dtype={'f_hex':str, 'a_cap_hex':str,
                                    'client_ip':str, 'server_ip':str})


# Embed one sample while preserving its row order and whole-byte token boundaries.
def main():
    ap = argparse.ArgumentParser(description='Create F/A embeddings and H metadata for one sample.')
    ap.add_argument('--dataset', choices=['unsw_nb15','cicids2017'], required=True)
    ap.add_argument('--sample', choices=['W1','W2','W3','W4'], required=True)
    ap.add_argument('--model', choices=['bigbird','bert'], default='bigbird')
    ap.add_argument('--sessions', type=Path, default=ROOT/'sessions.zip')
    ap.add_argument('--embeddings', type=Path, default=ROOT/'embeddings.zip')
    ap.add_argument('--output', type=Path, default=ROOT/'generated/embeddings')
    ap.add_argument('--checkpoint', help='Local checkpoint directory; otherwise download the pinned revision.')
    ap.add_argument('--only', choices=['F','A'])
    a = ap.parse_args()
    settings = json.loads((ROOT/'settings.json').read_text())
    cfg = settings['embedding']
    if a.model == 'bert' and a.sample != 'W1':
        ap.error('The paper evaluates BERT on W1 only.')
    if a.sessions.suffix == '.zip':
        sessions = archive_frame(a.sessions, f'{a.dataset}/{a.sample}.csv')
    else:
        sessions = pd.read_csv(a.sessions, dtype={'f_hex':str,'a_cap_hex':str,
                                                 'client_ip':str,'server_ip':str})
    if not np.array_equal(sessions.session_id, np.arange(1,len(sessions)+1)):
        raise ValueError('Session IDs must be 1..N in row order.')
    root = a.output/a.dataset/a.sample
    target = root/a.model
    target.mkdir(parents=True,exist_ok=True)
    reps = [a.only] if a.only else ['F','A']
    if any((target/f'{rep}.npy').exists() for rep in reps):
        raise FileExistsError('Output already exists. Choose a new --output directory.')
    source_digest=hashlib.sha256(sessions[['session_id','client_ip','server_ip','client_port',
        'server_port','respond_time','f_hex','a_cap_hex']].to_csv(index=False).encode()).hexdigest()
    identity=root/'session_source.sha256'
    if identity.exists() and identity.read_text().strip()!=source_digest:
        raise ValueError('This output directory belongs to different sessions; choose a new --output directory.')
    identity.write_text(source_digest+'\n')
    h = np.column_stack([sessions.client_ip.map(lambda x:int(x.split('.')[0])),
                         sessions.server_ip.map(lambda x:int(x.split('.')[0])),
                         sessions.client_port, sessions.server_port, sessions.respond_time]).astype(np.float64)
    if (root/'H.npy').exists() and not np.array_equal(np.load(root/'H.npy'),h):
        raise ValueError('Existing metadata differs from these sessions.')
    np.save(root/'H.npy',h)
    global torch, tok, m_full, m_sparse, dev, THR, BLK
    import torch
    from transformers import AutoTokenizer, AutoModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to reproduce the recorded embedding setup.')
    torch.manual_seed(settings['seed']); np.random.seed(settings['seed'])
    torch.set_num_threads(cfg['threads'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    dev = torch.device('cuda')
    ckpt = checkpoint(a.model,settings,a.checkpoint)
    tok = AutoTokenizer.from_pretrained(ckpt,use_fast=True)
    if a.model == 'bigbird':
        mc = json.loads((ckpt/'config.json').read_text())
        THR = (5+2*mc['num_random_blocks'])*mc['block_size']; BLK = mc['block_size']
        m_full = AutoModel.from_pretrained(ckpt,attention_type='original_full').to(dev).eval()
        m_sparse = AutoModel.from_pretrained(ckpt,attention_type='block_sparse').to(dev).eval()
    else:
        tok.model_max_length = 10**9
        model = AutoModel.from_pretrained(ckpt).to(dev).eval()
        bert = Embedder(str(ckpt),dev,torch,tok,model,cfg['token_budget'],cfg['bert_max_batch'],print)
    for rep in reps:
        hexes = sessions['f_hex' if rep=='F' else 'a_cap_hex'].tolist()
        if a.model == 'bigbird':
            idx, _ = build_index(tok,hexes,cfg['max_positions']-2,THR,print,rep)
            idx.insert(0,'session_id',sessions.session_id.to_numpy())
        else:
            index = root/'bigbird'/f'index_{rep}.csv'
            if index.exists():
                idx = pd.read_csv(index)
            else:
                member=f'{a.dataset}/{a.sample}.csv'
                if a.sessions.suffix=='.zip':
                    with zipfile.ZipFile(a.sessions) as z:
                        actual=hashlib.sha256(z.read(member)).hexdigest()
                else:
                    actual=hashlib.sha256(a.sessions.read_bytes()).hexdigest()
                if actual!=settings['files_sha256'][f'sessions.zip:{member}']:
                    raise ValueError('Run BigBird on these sessions first to create their retained-byte index.')
                idx = archive_frame(a.embeddings,f'{a.dataset}/{a.sample}/bigbird/index_{rep}.csv')
            if not np.array_equal(idx.session_id,sessions.session_id):
                raise ValueError('BigBird retained-byte index has different session IDs.')
            if not np.array_equal(idx.n_bytes_in,[len(x)//2 for x in hexes]):
                raise ValueError('Run BigBird on these sessions first to obtain their retained-byte index.')
        idx.to_csv(target/f'index_{rep}.csv',index=False)
        torch.cuda.synchronize(); start=time.perf_counter()
        result = np.empty((len(sessions),768),np.float32)
        if a.model == 'bigbird':
            lengths = idx.seq_len.tolist()
            order = sorted(range(len(sessions)),key=lambda i:(lengths[i],i))
            batches = make_batches(lengths,order,cfg['token_budget'],cfg['max_batch'],THR,BLK)
            texts = [spaced(x[:2*int(n)]) for x,n in zip(hexes,idx.bytes_used)]
            for bi,rows in enumerate(batches):
                result[rows] = run_safe([texts[i] for i in rows],[lengths[i] for i in rows])[2]
                if bi % 40 == 0: print(f'{rep}: batch {bi+1}/{len(batches)}',flush=True)
        else:
            nbytes = idx.bytes_used.astype(int).tolist()
            for start_row in range(0,len(sessions),500):
                rows=list(range(start_row,min(start_row+500,len(sessions))))
                result[rows],_ = bert.embed_sessions(hexes,nbytes,rows)
                print(f'{rep}: {rows[-1]+1}/{len(sessions)} sessions',flush=True)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        np.save(target/f'{rep}.npy',result)
        report={'model':a.model,'representation':rep,'sessions':len(sessions),
                'seconds':elapsed,'seconds_per_session':elapsed/len(sessions),
                'mean_retained_bytes':float(idx.bytes_used.mean()),
                'seed':settings['seed'],'checkpoint':settings['models'][a.model]['repo'],
                'revision':settings['models'][a.model]['revision'] if not a.checkpoint else 'local override',
                'note':'New run; model initialization excluded. Compact script timing excludes final output write.'}
        (target/f'{rep}_timing.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
