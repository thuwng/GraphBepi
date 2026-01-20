import os
import torch
import numpy as np
import pandas as pd
import pickle as pk
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,trange
from preprocess import *
from graph_construction import calcPROgraph

# ==== Added robust PDB downloader (GraphBepi Kaggle fix) ====
import requests, time, gzip, shutil, io
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def _requests_session(max_retries=5, backoff_factor=0.5):
    r = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(['GET', 'HEAD'])
    )
    s = requests.Session()
    s.headers.update({"User-Agent": "GraphBepi/1.0 (+https://github.com/biomed-AI/GraphBepi)"})
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s

def download_pdb_with_fallback(pdb_id_4, out_pdb_path):
    """
    Tải file PDB hoặc CIF ổn định hơn cho Kaggle.
    """
    pdb = pdb_id_4.lower()
    sess = _requests_session()

    urls = [
        f"https://files.rcsb.org/download/{pdb}.pdb",
        f"https://files.rcsb.org/download/{pdb}.cif",
        f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdb[1:3]}/pdb{pdb}.ent.gz"
    ]
    for url in urls:
        try:
            resp = sess.get(url, timeout=60, stream=True)
            if resp.status_code != 200:
                continue
            if url.endswith(".gz"):
                with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz, open(out_pdb_path, "wb") as fo:
                    shutil.copyfileobj(gz, fo)
                return True
            with open(out_pdb_path, "wb") as fo:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fo.write(chunk)
            return True
        except Exception:
            time.sleep(0.5)
            continue
    return False
# ============================================================

# prot_amino2id={
#     '<pad>': 0, '</s>': 1, '<unk>': 2, 'A': 3,
#     'L': 4, 'G': 5, 'V': 6, 'S': 7,
#     'R': 8, 'E': 9, 'D': 10, 'T': 11,
#     'I': 12, 'P': 13, 'K': 14, 'F': 15,
#     'Q': 16, 'N': 17, 'Y': 18, 'M': 19,
#     'H': 20, 'W': 21, 'C': 22, 'X': 23,
#     'B': 24, 'O': 25, 'U': 26, 'Z': 27
# }

#vocabulary của esm-2
amino2id={
    '<null_0>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
    'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 
    'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, 
    '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32, '<cath>': 33, '<af2>': 34
}
class chain:
    def __init__(self):
        self.sequence=[] #list kí tự aa
        self.amino=[] #list id (theo amino2id)
        self.coord=[] #list tọa độ cacbon alpha cho mỗi residue
        self.site={}
        self.date=''
        self.length=0
        self.adj=None
        self.edge=None
        self.feat=None
        self.dssp=None
        self.name=''
        self.chain_name=''
        self.protein_name=''
    def add(self,amino,pos,coord): #thêm 1 residue vào chain
        self.sequence.append(DICT[amino])
        self.amino.append(amino2id[DICT[amino]])
        self.coord.append(coord)
        self.site[pos]=self.length
        self.length+=1
    def process(self):
        self.amino=torch.LongTensor(self.amino)
        self.coord=torch.FloatTensor(self.coord)
        self.label=torch.zeros_like(self.amino)
        self.sequence=''.join(self.sequence)
    def extract(self,model,device,path):
        if len(self)>1024 or model is None:
            return
        layer = model.num_layers
        f = lambda x: model(
            x.to(device).unsqueeze(0),
            repr_layers=[layer]
        )["representations"][layer].squeeze(0).cpu()
        with torch.no_grad():
            feat=f(self.amino)
        torch.save(feat,f'{path}/feat/{self.name}_esm2.ts')
    def load_dssp(self, path):
        dssp = torch.Tensor(np.load(f'{path}/dssp/{self.name}.npy'))
        pos  = np.load(f'{path}/dssp/{self.name}_pos.npy')

        self.dssp = torch.Tensor([
            -2.4492936e-16, -2.4492936e-16,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]).repeat(self.length, 1)
        self.rsa = torch.zeros(self.length)

        missed = 0
        for i in range(len(dssp)):
            k = pos[i]

            # normalize key variants
            ks = str(k).strip()          # '260'
            k0 = ks                       # primary string key
            k_int = None
            try:
                k_int = int(ks)           # 260 (if pure number)
            except:
                pass

            idx = None
            # try direct matches first
            if k in self.site:
                idx = self.site[k]
            elif k0 in self.site:
                idx = self.site[k0]
            elif k_int is not None and k_int in self.site:
                idx = self.site[k_int]
            else:
                # last-resort: handle insertion codes like '260A' by prefix match
                # (same idea as your update() method)
                for key in self.site.keys():
                    if str(key).startswith(k0):
                        idx = self.site.get(key)
                        break

            if idx is None:
                missed += 1
                continue

            self.dssp[idx] = dssp[i]
            # rsa threshold should be written at residue index, not loop index i
            if dssp[i][4] > 0.15:
                self.rsa[idx] = 1

        self.rsa = self.rsa.bool()
        if missed > 0:
            print(f"[WARN] DSSP mapping missed {missed} residues for {self.name}")

    def load_feat(self,path):
        self.feat=torch.load(f'{path}/feat/{self.name}_esm2.ts')
    def load_adj(self,path,self_cycle=False):
        graph=torch.load(f'{path}/graph/{self.name}.graph')
        self.adj=graph['adj'].to_dense()
        self.edge=graph['edge'].to_dense()
        if not self_cycle:
            self.adj[range(len(self)),range(len(self))]=0
            self.edge[range(len(self)),range(len(self))]=0
    def get_adj(self,path,dseq=3,dr=10,dlong=5,k=10):
        graph=calcPROgraph(self.sequence,self.coord,dseq,dr,dlong,k)
        torch.save(graph,f'{path}/graph/{self.name}.graph')
    def update(self,pos,amino):
        if amino not in DICT.keys():
            return
        amino_id=amino2id[DICT[amino]]
        idx=self.site.get(pos,None)
        if idx is None:
            for i in self.site.keys():
                # print(i,pos)
                if i[:len(pos)]==pos:
                    idx=self.site.get(i)
                    if amino_id==self.amino[idx]:
                        self.label[idx]=1
                        return
        elif amino_id!=self.amino[idx]:
            for i in self.site.keys():
                if i[:len(pos)]==pos:
                    idx=self.site.get(i)
                    if amino_id==self.amino[idx]:
                        self.label[idx]=1
                        return
        else:
            self.label[idx]=1
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return self.amino[idx],self.coord[idx],self.label[idx]
def collate_fn(batch):
    edges = [item['edge'] for item in batch]
    feats = [item['feat'] for item in batch]
    labels = torch.cat([item['label'] for item in batch],0)
    return feats,edges,labels

def extract_chain(root,pid,chain,force=False):
    if not force and os.path.exists(f'{root}/purePDB/{pid}_{chain}.pdb'):
        return True
    # if not os.path.exists(f'{root}/PDB/{pid}.pdb'):
    #     retry=5
    #     pdb=None
    #     while retry>0:
    #         try:
    #             with rq.get(f'https://files.rcsb.org/download/{pid}.pdb') as f:
    #                 if f.status_code==200:
    #                     pdb=f.content
    #                     break
    #         except:
    #             retry-=1
    #             continue
    #     if pdb is None:
    #         print(f'PDB file {pid} failed to download')
    #         return False
    #     with open(f'{root}/PDB/{pid}.pdb','wb') as f:
    #         f.write(pdb)

    if not os.path.exists(f'{root}/PDB/{pid}.pdb'):
        os.makedirs(f'{root}/PDB', exist_ok=True)
        out_pdb = f'{root}/PDB/{pid}.pdb'
        ok = download_pdb_with_fallback(pid, out_pdb)
        if not ok:
            print(f'PDB file {pid} failed to download (after retries)')
            return False

    lines=[]
    with open(f'{root}/PDB/{pid}.pdb','r') as f:
        for line in f:
            if line[:6]=='HEADER':
                lines.append(line)
            if line[:6].strip()=='TER' and line[21]==chain:
                lines.append(line)
                break
            feats=judge(line,None)
            if feats is not None and feats[1]==chain:
                lines.append(line)
    with open(f'{root}/purePDB/{pid}_{chain}.pdb','w') as f:
        for i in lines:
            f.write(i)
    return True
def process_chain(data,root,pid,model,device):
    get_dssp(pid,root)
    same={}
    with open(f'{root}/purePDB/{pid}.pdb','r') as f:
        for line in f:
            if line[:6]=='HEADER':
                date=line[50:59].strip()
                data.date=date
                continue
            feats=judge(line,'CA')
            if feats is None:
                continue
            amino,_,site,x,y,z=feats
            if len(amino)>3:
                if same.get(site) is None:
                    same[site]=amino[0]
                if same[site]!=amino[0]:
                    continue
                amino=amino[-3:]
            data.add(amino,site,[x,y,z])
    data.process()
    data.get_adj(root)
    data.extract(model,device,root)
    return data
def initial(file,root,model=None,device='cpu',from_native_pdb=True):
    df=pd.read_csv(f'{root}/{file}',header=0,index_col=0)
    prefix=df.index
    labels=df['Epitopes (resi_resn)']
    samples=[]
    with tqdm(prefix) as tbar:
        for i in tbar:
            tbar.set_postfix(protein=i)
            if from_native_pdb:
                state=extract_chain(root,i[:4],i[-1])
                if not state:
                    continue
            data=chain()
            p,c=i.split('_')
            data.protein_name=p
            data.chain_name=c
            data.name=f"{p}_{c}"
            process_chain(data,root,i,model,device)
            label=labels.loc[i].split(', ')
            for j in label:
                site,amino=j.split('_')
                data.update(site,amino)
            samples.append(data)
    with open(f'{root}/total.pkl','wb') as f:
        pk.dump(samples,f)


def export_tabular(root, out_dir="./tabular", split='all'):
    """Export per-residue tabular features for XGBoost.
    Produces: <out_dir>/<split>.npz with arrays: X (N x D), y (N,), names (N,), idx (N,), resn (N,)
    split: 'train' or 'test' or 'all' (default 'all' -> concatenate train+test)
    """
    os.makedirs(out_dir, exist_ok=True)
    pk_map = {
        'train': f'{root}/train.pkl',
        'test': f'{root}/test.pkl',
        'all': None,
    }
    samples = []
    if split in ('train','test'):
        p = pk_map[split]
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run initial() first to build pickles")
        with open(p,'rb') as f:
            samples = pk.load(f)
    else:
        # concat train + test if available
        files = [f'{root}/train.pkl', f'{root}/test.pkl']
        for p in files:
            if os.path.exists(p):
                with open(p,'rb') as f:
                    samples += pk.load(f)
        if len(samples)==0:
            # fallback to total.pkl
            with open(f'{root}/total.pkl','rb') as f:
                samples = pk.load(f)

    rows_X = []
    rows_y = []
    rows_name = []
    rows_idx = []
    rows_resn = []

    for s in tqdm(samples, desc='Exporting residues'):
        # ensure features loaded
        try:
            s.load_feat(root)
            s.load_dssp(root)
            s.load_adj(root,self_cycle=False)
        except Exception as e:
            print(f"[WARN] Failed to load features for {s.name}: {e}")
            continue
        L = len(s)
        feat = s.feat.numpy() if isinstance(s.feat, torch.Tensor) else np.array(s.feat)
        dssp = s.dssp.numpy() if isinstance(s.dssp, torch.Tensor) else np.array(s.dssp)
        adj = s.adj.numpy() if isinstance(s.adj, torch.Tensor) else np.array(s.adj)
        edge = s.edge.numpy() if isinstance(s.edge, torch.Tensor) else np.array(s.edge)
        amino_ids = s.amino.numpy() if isinstance(s.amino, torch.Tensor) else np.array(s.amino)

        for i in range(L):
            esm_i = feat[i]
            dssp_i = dssp[i]
            deg = float(adj[i].sum())
            # neighbor mean edge features
            neighbors = adj[i] > 0
            if neighbors.any():
                neigh_edge_mean = edge[i, neighbors].mean(axis=0)
            else:
                neigh_edge_mean = np.zeros(edge.shape[2], dtype=np.float32)
            amino_id = float(amino_ids[i])
            x = np.concatenate([esm_i.astype(np.float32), dssp_i.astype(np.float32), np.array([deg], dtype=np.float32), neigh_edge_mean.astype(np.float32), np.array([amino_id], dtype=np.float32)])
            rows_X.append(x)
            rows_y.append(float(s.label[i].item() if isinstance(s.label, torch.Tensor) else s.label[i]))
            rows_name.append(s.name)
            rows_idx.append(i)
            rows_resn.append(s.sequence[i])

    X = np.vstack(rows_X).astype(np.float32)
    y = np.array(rows_y, dtype=np.uint8)
    names = np.array(rows_name, dtype=object)
    idxs = np.array(rows_idx, dtype=np.int32)
    resn = np.array(rows_resn, dtype=object)
    out_path = os.path.join(out_dir, f'{split}.npz')
    np.savez_compressed(out_path, X=X, y=y, names=names, idxs=idxs, resn=resn)
    print(f"[DONE] Exported {X.shape[0]} residues to {out_path}")
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/kaggle/input/dataset/data/BCE_633')
    parser.add_argument('--out', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='all', choices=['train','test','all'])
    args = parser.parse_args()
    export_tabular(args.root, args.out, args.split)
