# %% [markdown]
# # Bài 1: Kiểm chứng tin tức — lời giải
#
# Không dùng Transformer/LLM pretrained. Thành phần pretrained duy nhất là vector từ fastText `cc.vi.300`
# (đề cho phép "vector từ của fastText"). Mọi model khác học từ đầu trên `train.csv` + bài báo của train.
#
# 1. Tách câu; **retriever học có giám sát** (HGB trên đặc trưng khớp claim–câu, nhãn từ cột `evidence`) chọn 2 câu bằng chứng.
# 2. ~170 đặc trưng tay: độ phủ idf, n-gram, **diff claim↔bằng chứng** (thay thế ngắn = REFUTED, chèn thêm = NEI),
#    số/thực thể/phủ định mới, căn chỉnh mềm bằng cosine fastText.
# 3. Ba model: (a) HGB trên đặc trưng + xác suất LR-BoW (stacking lồng), (b) LR trên đặc trưng + BoW,
#    (c) ESIM BiLSTM học từ đầu trên (claim, 2 câu bằng chứng), pretrain trên claim tổng hợp sinh từ bài báo train.
# 4. Trộn log-xác suất; **giải mã chung theo bài** với prior số nhãn 2/2/2 (bài train phần lớn có 2 claim/nhãn,
#    test giữ ngẫu nhiên 3–5 claim/bài).
#
# Không học bất kỳ thứ gì trên văn bản test: IDF, từ vựng BoW, retriever, model đều fit trên train.

# %%
import re, math, difflib, gzip, itertools, random, zipfile, os
from collections import Counter
from math import comb, log
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

DATA_DIR = Path(os.environ.get("DATA_DIR", ".."))                    # root repo: train.csv, articles.csv, public_test.csv, ...
VEC_GZ = Path(os.environ.get("VEC_GZ", "vec/cc.vi.300.vec.gz"))      # fastText cc.vi.300 (.vec.gz)
RUN_CV = os.environ.get("RUN_CV", "0") == "1"   # 5-fold CV đầu-cuối trên train (lâu ~5x)
NN_SEEDS = int(os.environ.get("NN_SEEDS", 6))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W_BLEND = {"stack": 0.4, "lrall": 0.4, "nn": 0.2}   # chọn bằng CV lồng trên train
LAM = 0.5                                            # độ mạnh prior 2/2/2 khi giải mã chung
LABELS = ["SUPPORTED", "REFUTED", "NEI"]

train = pd.read_csv(DATA_DIR / "train.csv")
articles = pd.read_csv(DATA_DIR / "articles.csv")
tests = [pd.read_csv(DATA_DIR / "public_test.csv")]
if (DATA_DIR / "private_test.csv").exists():
    tests.append(pd.read_csv(DATA_DIR / "private_test.csv"))
    articles = pd.concat([articles, pd.read_csv(DATA_DIR / "private_articles.csv")], ignore_index=True).drop_duplicates("article_id")
test = pd.concat(tests, ignore_index=True)
if os.environ.get("SMOKE"):  # quick bug check on a small slice
    train = train[train.article_id.isin(train.article_id.unique()[:120])].reset_index(drop=True); test = test.head(60)
print(len(train), "train,", len(test), "test,", len(articles), "articles")

# %% [markdown]
# ## Tách câu, token, vector fastText

# %%
def split_sents(t):
    s = [c for c in re.split(r"(?<=[.!?…;:])\s+", str(t)) if len(c.split()) >= 3]
    return s or [str(t)]

TOK = re.compile(r"\d+(?:[.,/]\d+)*|\w+")
def toks(s): return TOK.findall(str(s).lower())
def toks_cased(s): return TOK.findall(str(s))
NUM = re.compile(r"\d")
NEG = {"không", "chưa", "chẳng", "chả", "đâu", "phủ", "bác", "ngừng", "thiếu"}
CUE = ["chỉ", "duy", "nhất", "tất", "toàn", "mọi", "luôn", "hoàn", "không", "chưa", "lần", "đầu", "tiên",
       "hơn", "giảm", "tăng", "sẽ", "đã", "đang", "vì", "nên", "do", "nhờ", "khiến", "nhưng", "mà", "cũng",
       "rất", "nhiều", "ít", "các", "cả", "đều", "bị", "được", "còn", "vẫn", "mới", "trước", "sau"]

SENTS = {r.article_id: split_sents(r.text) for r in articles.itertuples()}
TEXT = dict(zip(articles.article_id, articles.text))

# fastText: chỉ tra cứu (không huấn luyện) các từ xuất hiện trong dữ liệu cuộc thi
vocab_needed = set(w for t in list(train.claim) + list(train.evidence) + list(test.claim) + list(articles.text) for w in toks(t))
FTW, FTV = [], []
with gzip.open(VEC_GZ, "rt", encoding="utf8", errors="ignore") as f:
    next(f)
    for line in f:
        w, rest = line.split(" ", 1); lw = w.lower()
        if lw in vocab_needed:
            FTW.append(lw); FTV.append(np.asarray(rest.split(), dtype=np.float32)); vocab_needed.discard(lw)
FTV = np.stack(FTV); FTV /= np.linalg.norm(FTV, axis=1, keepdims=True).clip(1e-6)
WV = dict(zip(FTW, FTV)); ZV = np.zeros(300, np.float32)
print(len(FTW), "từ có vector fastText")

# %% [markdown]
# ## Đặc trưng

# %%
def numset(ts):
    out = set()
    for t in ts:
        if NUM.search(t): out.add(t); out.update(re.split(r"[.,/]", t))
    return out

class Idf:
    def __init__(self, docs):
        self.n = len(docs); self.df = Counter(w for d in docs for w in set(toks(d)))
    def __call__(self, w):
        return math.log((self.n + 1) / (self.df.get(w, 0) + 1)) + 1

def bigr(t): return set(zip(t, t[1:]))
def trig(t): return set(zip(t, t[1:], t[2:]))

def wcov(ct, xs, idf):
    tot = sum(idf(w) for w in ct) or 1.0
    return sum(idf(w) for w in ct if w in xs) / tot

def retrieve(ct, sts, idf):
    cb = bigr(ct)
    return np.array([wcov(ct, set(st), idf) + len(cb & bigr(st)) / max(1, len(cb)) for st in sts])

def diff_feats(ct, et):
    sm = difflib.SequenceMatcher(None, et, ct, autojunk=False)
    rep_c = rep_e = ins = dele = nrep = nins = ndel = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace": nrep += 1; rep_e += i2 - i1; rep_c += j2 - j1
        elif op == "insert": nins += 1; ins += j2 - j1
        elif op == "delete": ndel += 1; dele += i2 - i1
    lb = max((b.size for b in sm.get_matching_blocks()), default=0); n = max(1, len(ct))
    return [sm.ratio(), nrep, rep_c / n, rep_e / n, nins, ins / n, ndel, dele / max(1, len(et)), lb / n]

def diff_ops(ct, et, idf):
    """Thay thế ngắn (hai phía <=4 token) là dấu hiệu REFUTED; trả thêm token của các phép diff cho BoW."""
    sm = difflib.SequenceMatcher(None, et, ct, autojunk=False)
    nsm = nnum = nege = negc = 0; mx = 0.0; pairs, ins, rep = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        e, c = et[i1:i2], ct[j1:j2]
        nege += sum(w in NEG for w in e); negc += sum(w in NEG for w in c)
        if op == "replace":
            rep += c
            if len(e) <= 4 and len(c) <= 4:
                nsm += 1; mx = max(mx, max(idf(w) for w in c))
                nnum += any(NUM.search(w) for w in c) and any(NUM.search(w) for w in e)
                pairs += [f"{a}>{b}" for a in e for b in c]
        elif op == "insert": ins += c
    return [nsm, nnum, mx, nege, negc, int((nege > 0) != (negc > 0))], pairs, ins, rep

def vec_feats(ct, et, at, idf):
    def M(ts): return np.stack([WV.get(w, ZV) for w in ts]) if ts else np.zeros((1, 300), np.float32)
    C, E = M(ct), M(et); mx = (C @ E.T).max(1)
    ex = np.array([w in set(et) for w in ct]) if ct else np.zeros(1, bool)
    inv = np.array([w in WV for w in ct]) if ct else np.zeros(1, bool)
    nm = mx[~ex & inv] if ct else np.zeros(0)
    wts = np.array([idf(w) for w in ct]) if ct else np.ones(1)
    def sv(ts, X):
        w = np.array([idf(t) for t in ts]) if ts else np.ones(1); v = (X * w[:, None]).sum(0); return v / (np.linalg.norm(v) + 1e-6)
    cv = sv(ct, C); nov = [w for w in ct if w not in set(et)]
    return [float((mx * wts).sum() / wts.sum()), float(mx.min()), len(nm), float(nm.mean()) if len(nm) else 1.0,
            float(nm.min()) if len(nm) else 1.0, int((nm < 0.35).sum()), int(((nm >= 0.35) & (nm < 0.75)).sum()),
            float(cv @ sv(et, E)), float(cv @ sv(at[:600], M(at[:600]))), float(sv(nov, M(nov)) @ sv(et, E)) if nov else 1.0]

# ---- retriever học có giám sát ----
_SV = {}
def cand_feats(ct, sts, idf):
    def sv(ts):
        if not ts: return ZV
        v = sum(WV.get(w, ZV) * idf(w) for w in ts); return v / (np.linalg.norm(v) + 1e-6)
    key = (id(idf), tuple(map(tuple, sts)))
    if key not in _SV: _SV[key] = [sv(st) for st in sts]
    svs = _SV[key]; cvec, cb, ctr, cs = sv(ct), bigr(ct), trig(ct), set(ct)
    base = retrieve(ct, sts, idf); rk = np.argsort(np.argsort(-base)); n = len(sts)
    return np.array([[base[i], rk[i], base[i] - base.max(), wcov(ct, set(st), idf), np.mean([w in set(st) for w in ct]) if ct else 0,
                      len(cb & bigr(st)) / max(1, len(cb)), len(ctr & trig(st)) / max(1, len(ctr)),
                      wcov(st, cs, idf), float(cvec @ svs[i]), i / n, i, len(st), n] for i, st in enumerate(sts)], dtype=float)

def gold_mask(ev, sts):
    es = set(toks(ev)); fr = np.array([np.mean([w in es for w in st]) if st else 0 for st in sts])
    m = fr >= 0.6
    return m if m.any() else fr == fr.max()

class Retriever:
    def fit(self, df, idf):
        X, Y = [], []
        for c, e, a in zip(df.claim, df.evidence, df.article_id):
            sts = [toks(s) for s in SENTS[a]]; X.append(cand_feats(toks(c), sts, idf)); Y.append(gold_mask(e, sts))
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        self.m.fit(np.vstack(X), np.concatenate(Y)); self.idf = idf
        return self
    def scores(self, ct, sts):
        return self.m.predict_proba(cand_feats(ct, sts, self.idf))[:, 1]

def featurize(df, idf, R):
    rows = []; aux = {k: [] for k in ("ev2", "novel", "missing", "pairs", "ins", "rep")}; cache = {}
    for claim, a in zip(df.claim, df.article_id):
        if a not in cache:
            ss = SENTS[a]; sts = [toks(s) for s in ss]; at = [w for st in sts for w in st]
            cache[a] = (ss, sts, set(at), bigr(at), trig(at), numset(at), at)
        ss, sts, aset, abig, atri, anum, at = cache[a]
        ct = toks(claim); cc = toks_cased(claim)
        sc = retrieve(ct, sts, idf); rs = R.scores(ct, sts); order = np.argsort(-rs)
        i1 = order[0]; i2 = order[1] if len(order) > 1 else i1
        s1 = sts[i1]; w2 = sts[min(i1, i2)] + (sts[max(i1, i2)] if i2 != i1 else [])
        wc = (sts[i1 - 1] if i1 > 0 else []) + s1 + (sts[i1 + 1] if i1 + 1 < len(sts) else [])
        s1set, w2set, wcset, cset = set(s1), set(w2), set(wc), set(ct)
        f = [sc[i1], sc[i2], sc[i1] - sc[i2], np.sort(sc)[-3:].mean(), len(ss), rs[i1], rs[i2], np.sort(rs)[-3:].sum()]
        for X in (s1set, w2set, wcset, aset):
            f += [wcov(ct, X, idf), np.mean([w in X for w in ct]) if ct else 0]
        cb, ctr = bigr(ct), trig(ct)
        for B, T in ((bigr(s1), trig(s1)), (bigr(w2), trig(w2)), (abig, atri)):
            f += [len(cb & B) / max(1, len(cb)), len(ctr & T) / max(1, len(ctr))]
        f += [wcov(s1, cset, idf), len(ct), len(s1), len(ct) / max(1, len(s1))]
        f += diff_feats(ct, s1) + diff_feats(ct, w2)
        nov_a = [w for w in ct if w not in aset]; nov_w = [w for w in ct if w not in w2set]
        f += [len(nov_a), sum(idf(w) for w in nov_a), max([idf(w) for w in nov_a], default=0),
              len(nov_w), sum(idf(w) for w in nov_w), len([w for w in nov_w if w in aset])]
        caps = {t.lower() for k, t in enumerate(cc) if k > 0 and t[:1].isupper()}
        f += [len(caps), len(caps - aset), len(caps - w2set)]
        cn, wn = numset(ct), numset(w2)
        f += [len(cn), len(cn - anum), len(cn - numset(wc) - numset(w2)), len(wn - cn)]
        negc, negw = sum(w in NEG for w in ct), sum(w in NEG for w in w2)
        f += [negc, negw, negc - negw, int(any(w in NEG for w in nov_w)), sum(w in NEG for w in s1)]
        f += [int(w in nov_w) for w in CUE] + [int(w in cset) - int(w in s1set) for w in CUE]
        d2, _, _, _ = diff_ops(ct, w2, idf); d1, pr, ins, rep = diff_ops(ct, s1, idf); f += d1 + d2
        f += vec_feats(ct, s1, [], idf)[:7] + vec_feats(ct, w2, at, idf)
        rows.append(f)
        aux["ev2"].append(ss[min(i1, i2)] + " " + (ss[max(i1, i2)] if i2 != i1 else ""))
        aux["novel"].append(" ".join(nov_w)); aux["missing"].append(" ".join(w for w in s1 if w not in cset))
        aux["pairs"].append(" ".join(pr)); aux["ins"].append(" ".join(ins)); aux["rep"].append(" ".join(rep))
    aux["claim"] = list(df.claim)
    return np.array(rows, dtype=float), aux

# %% [markdown]
# ## Claim tổng hợp (chỉ từ bài báo của tập train) để pretrain ESIM

# %%
def make_synth(art_ids, seed=0):
    rnd = random.Random(seed)
    sents = {a: SENTS[a] for a in art_ids}; allfr = [s for v in sents.values() for s in v]
    widx = {w: i for i, w in enumerate(FTW)}
    freq = pd.Series([w for s in allfr for w in toks(s)]).value_counts()
    common = [w for w in freq.index[:8000] if w in widx and not NUM.search(w)]; CW = FTV[[widx[w] for w in common]]
    STOP = set(freq.index[:60])
    def neighbour(w):
        sims = CW @ FTV[widx[w]]
        for j in np.argsort(-sims)[1:8]:
            if common[j] != w and sims[j] < 0.9: return common[j]
    def caps_spans(s): return [m.group(0) for m in re.finditer(r"(?:[A-ZĐÀ-Ỹ][\wÀ-ỹ]*\s?){1,4}", s) if m.start() > 0]
    def sup(s):
        parts = [p for p in re.split(r",\s*", s) if p]
        if len(parts) > 2 and rnd.random() < 0.7:
            k = rnd.randrange(1, len(parts)); parts = parts[:k] + parts[k + 1:]
        return ", ".join(parts)
    def ref(s, art):
        t = s.split(); kinds = ["num", "ent", "nb", "neg"]; rnd.shuffle(kinds)
        for k in kinds:
            if k == "num":
                ids = [i for i, w in enumerate(t) if re.fullmatch(r"\d+", w)]
                if ids:
                    i = rnd.choice(ids); v = int(t[i])
                    t[i] = str(v + rnd.choice([-3, -2, -1, 1, 2, 5, 10]) if v > 10 else v + rnd.randint(1, 5)); return " ".join(t)
            if k == "ent":
                mine = caps_spans(s); other = [c for x in art if x != s for c in caps_spans(x) if c.strip() not in s]
                if mine and other: return s.replace(rnd.choice(mine).strip(), rnd.choice(other).strip(), 1)
            if k == "nb":
                ids = [i for i, w in enumerate(t) if w.lower() in widx and w.lower() not in STOP and not w[:1].isupper() and len(w) > 1]
                rnd.shuffle(ids)
                for i in ids[:3]:
                    n = neighbour(t[i].lower())
                    if n: t[i] = n; return " ".join(t)
            if k == "neg":
                if " không " in s: return s.replace(" không ", " ", 1)
                ids = [i for i, w in enumerate(t) if w in ("đã", "được", "sẽ", "có", "là")]
                if ids: t.insert(rnd.choice(ids), "không"); return " ".join(t)
    def nei(s):
        fr = rnd.choice(allfr).split(); n = rnd.randint(4, 10); st = rnd.randrange(0, max(1, len(fr) - n))
        frag = " ".join(fr[st:st + n]).strip(".,;:")
        if rnd.random() < 0.6: return s.rstrip(". ") + ", " + frag + "."
        parts = s.split(", "); parts.insert(rnd.randrange(0, len(parts)) + 1, frag); return ", ".join(parts)
    rows = []
    for a, ss in sents.items():
        cand = [s for s in ss if len(s.split()) >= 10]; rnd.shuffle(cand)
        for s in cand[:12]:
            lab = rnd.choice(LABELS)
            c = sup(s) if lab == "SUPPORTED" else ref(s, ss) if lab == "REFUTED" else nei(s)
            if c and c != s or lab == "SUPPORTED": rows.append((a, c, lab))
    return pd.DataFrame(rows, columns=["article_id", "claim", "label"])

# %% [markdown]
# ## Model 1+2: HGB stacking và LR trên đặc trưng + BoW

# %%
BOW_KEYS = ["novel", "missing", "claim", "pairs", "ins", "rep"]
def bow(aux_a, aux_b):
    A, B = [], []
    for k in BOW_KEYS:
        v = TfidfVectorizer(tokenizer=str.split if k == "pairs" else toks, token_pattern=None, lowercase=False,
                            ngram_range=(1, 1) if k == "pairs" else (1, 2), min_df=2, sublinear_tf=True)
        A.append(v.fit_transform(aux_a[k])); B.append(v.transform(aux_b[k]))
    return sp.hstack(A).tocsr(), sp.hstack(B).tocsr()

def fit_stack(Xa, auxa, ya, ga, Xb, auxb):
    A, B = bow(auxa, auxb)
    lr = lambda: LogisticRegression(C=2.0, max_iter=3000)
    inner = np.zeros((len(ya), 3))
    for ia, ib in GroupKFold(5).split(A, groups=ga):
        inner[ib] = lr().fit(A[ia], ya[ia]).predict_proba(A[ib])
    pb = lr().fit(A, ya).predict_proba(B)
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.02, max_leaf_nodes=7, l2_regularization=1.0,
                                         min_samples_leaf=50, random_state=0)
    p_stack = hgb.fit(np.hstack([Xa, inner]), ya).predict_proba(np.hstack([Xb, pb]))
    sc = StandardScaler().fit(Xa); D = lambda X: sp.csr_matrix(np.clip(sc.transform(X), -5, 5))
    p_lr = LogisticRegression(C=0.3, max_iter=5000).fit(sp.hstack([D(Xa), A]).tocsr(), ya).predict_proba(sp.hstack([D(Xb), B]).tocsr())
    return p_stack, p_lr

# %% [markdown]
# ## Model 3: ESIM (BiLSTM + cross-attention) học từ đầu, embedding fastText đóng băng

# %%
LC, LE, NB = 64, 160, 50000
VOCAB = {w: i + 2 for i, w in enumerate(FTW)}
FTW_MAT = np.zeros((len(FTW) + 2, 300), np.float32); FTW_MAT[2:] = FTV

def h(s):
    v = 2166136261
    for ch in s.encode(): v = ((v ^ ch) * 16777619) & 0xFFFFFFFF
    return v % NB + 1

def encode(texts, other, ctx_sets, L):
    n = len(texts); I = np.zeros((n, L), np.int64); G = np.zeros((n, L, 3), np.int64); Fz = np.zeros((n, L, 5), np.float32)
    for r, (t, o, cs) in enumerate(zip(texts, other, ctx_sets)):
        os_ = set(toks(o))
        for k, w in enumerate(toks_cased(t)[:L]):
            lw = w.lower(); s = f"<{lw}>"
            I[r, k] = VOCAB.get(lw, 1); G[r, k] = [h(s[:3]), h(s[-3:]), h(s)]
            Fz[r, k] = [lw in os_, lw in cs, bool(NUM.search(lw)), w[:1].isupper() and k > 0, 1.0]
    return I, G, Fz

class ESIM(nn.Module):
    def __init__(s, nf, d=128):
        super().__init__()
        s.ft = nn.Embedding.from_pretrained(torch.tensor(FTW_MAT), freeze=True, padding_idx=0); s.ftp = nn.Linear(300, d)
        s.ng = nn.Embedding(NB + 1, d, padding_idx=0); s.fp = nn.Linear(5, 16)
        s.enc = nn.LSTM(d + 16, d, batch_first=True, bidirectional=True)
        s.comp = nn.LSTM(8 * d, d, batch_first=True, bidirectional=True)
        s.hf = nn.Sequential(nn.Linear(nf, 128), nn.ReLU(), nn.Dropout(0.2))
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * d + 128, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
        s.dp = nn.Dropout(0.3)
    def embed(s, I, G, Fz):
        return s.dp(torch.cat([s.ftp(s.ft(I)) + s.ng(G).sum(2), F.relu(s.fp(Fz))], -1))
    def forward(s, ci, cg, cf, ei, eg, ef, hf):
        cm, em = ci > 0, ei > 0
        a, _ = s.enc(s.embed(ci, cg, cf)); b, _ = s.enc(s.embed(ei, eg, ef))
        att = a @ b.transpose(1, 2)
        ab = torch.softmax(att.masked_fill(~em[:, None], -1e4), 2) @ b
        ba = torch.softmax(att.masked_fill(~cm[:, :, None], -1e4).transpose(1, 2), 2) @ a
        va, _ = s.comp(torch.cat([a, ab, a - ab, a * ab], -1)); vb, _ = s.comp(torch.cat([b, ba, b - ba, b * ba], -1))
        pool = lambda v, m: torch.cat([v.masked_fill(~m[..., None], -1e4).max(1)[0], (v * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)], -1)
        return s.head(torch.cat([pool(va, cm), pool(vb, em), s.hf(hf)], -1))

def fit_nn(Xa, auxa, ya, dfa, Xb, auxb, dfb, syn, seeds):
    asets = {}
    ctx = lambda df: [asets.setdefault(x, set(toks(TEXT[x]))) for x in df.article_id]
    sc = StandardScaler().fit(Xa)
    def pack(df, aux, X):
        c = encode(list(df.claim), list(aux["ev2"]), ctx(df), LC); e = encode(list(aux["ev2"]), list(df.claim), [set()] * len(df), LE)
        return [torch.tensor(v) for v in (*c, *e)] + [torch.tensor(np.clip(sc.transform(X), -5, 5), dtype=torch.float32)]
    A, B = pack(dfa, auxa, Xa), pack(dfb, auxb, Xb)
    Xs, auxs, dfs, ys = syn; S = pack(dfs, auxs, Xs)
    out = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = ESIM(Xa.shape[1]).to(DEV)
        for D_, yy, EP in [(S, ys, 4), (A, ya, 14)]:   # pretrain synthetic -> fine-tune real
            yt = torch.tensor(yy); bs = 32
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * ((len(yy) + bs - 1) // bs))
            for ep in range(EP):
                m.train(); perm = torch.randperm(len(yy))
                for i in range(0, len(yy), bs):
                    j = perm[i:i + bs]
                    loss = F.cross_entropy(m(*[t[j].to(DEV) for t in D_]), yt[j].to(DEV), label_smoothing=0.1)
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            out.append(np.concatenate([torch.softmax(m(*[t[i:i + 256].to(DEV) for t in B]), -1).cpu().numpy() for i in range(0, len(B[0]), 256)]))
    return np.mean(out, 0)

# %% [markdown]
# ## Ghép: fit trên một tập train, dự đoán một tập khác; giải mã chung theo bài

# %%
def fit_predict(trn, tst, nn_seeds=NN_SEEDS):
    ya = trn.label.map({l: i for i, l in enumerate(LABELS)}).values
    idf = Idf([TEXT[a] for a in trn.article_id.unique()] + list(trn.claim))
    R = Retriever().fit(trn, idf)
    Xa, auxa = featurize(trn, idf, R); Xb, auxb = featurize(tst, idf, R)
    p_stack, p_lr = fit_stack(Xa, auxa, ya, trn.article_id.values, Xb, auxb)
    syn = make_synth(trn.article_id.unique()); Xs, auxs = featurize(syn, idf, R)
    p_nn = fit_nn(Xa, auxa, ya, trn, Xb, auxb, tst, (Xs, auxs, syn, syn.label.map({l: i for i, l in enumerate(LABELS)}).values), range(nn_seeds))
    parts = {"stack": p_stack, "lrall": p_lr, "nn": p_nn}
    B = np.exp(sum(W_BLEND[k] * np.log(np.clip(p, 1e-6, 1)) for k, p in parts.items()))
    return B / B.sum(1, keepdims=True), parts

def count_logprior(cnt, pool=(2, 2, 2)):
    n = sum(cnt); p = np.prod([comb(k, c) for k, c in zip(pool, cnt)]) / comb(sum(pool), n) if n <= sum(pool) else 0
    return log(p) if p > 0 else -1e9

def joint_decode(P, groups, lam=LAM):
    """Chọn bộ nhãn cho cả bài: tổng log-xác suất + lam * log P(số nhãn | rút ngẫu nhiên từ bài 2/2/2)."""
    out = P.argmax(1).copy(); L = np.log(np.clip(P, 1e-9, 1))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 6: continue
        best = max(itertools.product(range(3), repeat=len(idx)),
                   key=lambda asg: L[idx, asg].sum() + lam * count_logprior(np.bincount(asg, minlength=3)))
        out[idx] = best
    return out

# %% [markdown]
# ## (Tuỳ chọn) CV 5-fold đầu-cuối theo bài báo

# %%
if RUN_CV:
    y = train.label.map({l: i for i, l in enumerate(LABELS)}).values
    oof = np.zeros((len(train), 3)); oof_parts = {k: np.zeros((len(train), 3)) for k in W_BLEND}
    for k, (a, b) in enumerate(GroupKFold(5).split(train, groups=train.article_id)):
        P, parts = fit_predict(train.iloc[a].reset_index(drop=True), train.iloc[b].reset_index(drop=True))
        oof[b] = P
        for n_, p in parts.items(): oof_parts[n_][b] = p
        print("fold", k, round(f1_score(y[b], P.argmax(1), average="macro"), 4), flush=True)
    for n_, p in oof_parts.items(): print(n_, round(f1_score(y, p.argmax(1), average="macro"), 4))
    print("blend CV macro-F1", round(f1_score(y, oof.argmax(1), average="macro"), 4))
    rng = np.random.default_rng(0); s0, s1 = [], []
    for rep in range(10):   # mô phỏng test: giữ 3-5 claim ngẫu nhiên mỗi bài
        keep = np.array([i for idx in train.groupby("article_id").indices.values() for i in rng.permutation(idx)[:min(len(idx), rng.integers(3, 6))]])
        s0.append(f1_score(y[keep], oof[keep].argmax(1), average="macro"))
        s1.append(f1_score(y[keep], joint_decode(oof[keep], train.article_id.values[keep]), average="macro"))
    print("mô phỏng test: argmax", round(np.mean(s0), 4), "| giải mã chung", round(np.mean(s1), 4))
    np.save("oof_pipeline.npy", oof)
    for n_, p in oof_parts.items(): np.save(f"oof_pipeline_{n_}.npy", p)

# %% [markdown]
# ## Dự đoán test và ghi bài nộp

# %%
if os.environ.get("NO_FINAL"): raise SystemExit
P, parts = fit_predict(train, test)
np.save("test_probs.npy", P)
pred = joint_decode(P, test.article_id.values)
sub = test[["id"]].assign(label=[LABELS[i] for i in pred])
sub.to_csv("bai1_submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write("bai1_submission.csv", arcname="bai1_submission.csv")
print(len(sub), "dòng"); print(sub.label.value_counts())
