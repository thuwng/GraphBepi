import os
import pickle as pk
import traceback
from tqdm import tqdm

import torch
import esm

from utils import chain, extract_chain, process_chain


def choose_device(gpu):
    if gpu == -1 or not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if gpu < 0 or gpu >= n:
        return "cpu"
    return f"cuda:{gpu}"


def ensure_folders(root):
    for d in ["PDB", "purePDB", "feat", "dssp", "graph"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def load_esm(esm_size, device):
    if esm_size == "3B":
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_size == "150M":
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    else:
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    return model


def list_pdbs(pdb_root):
    out = []
    for fn in os.listdir(pdb_root):
        if fn.endswith(".pdb"):
            out.append(fn.replace(".pdb", "").lower())
    return sorted(out)


def main(args):
    if args.cache:
        os.environ["TORCH_HOME"] = args.cache
        os.makedirs(args.cache, exist_ok=True)

    root = args.root
    ensure_folders(root)

    device = choose_device(args.gpu)
    print(f"[INFO] Device: {device}")

    esm_model = load_esm(args.esm_size, device)

    pdb_ids = list_pdbs(args.pdb_dir)
    print(f"[INFO] Found {len(pdb_ids)} PDBs")

    test_samples = []

    for pdb in tqdm(pdb_ids, desc="Building DiscoTope test set"):
        try:
            chains = extract_chain(root, pdb, None, pdb_dir=args.pdb_dir)
        except Exception:
            continue

        for ch in chains:
            name = f"{pdb}_{ch}"
            obj = chain()
            obj.protein_name = pdb
            obj.chain_name = ch
            obj.name = name

            try:
                process_chain(obj, root, obj.name, esm_model, device)
            except Exception:
                traceback.print_exc()
                continue

            # Dummy label (all zeros)
            obj.label = torch.zeros(obj.length, dtype=torch.long)

            test_samples.append(obj)

    out_path = os.path.join(root, "test.pkl")
    with open(out_path, "wb") as f:
        pk.dump(test_samples, f)

    print(f"[DONE] test.pkl written: {out_path}")
    print(f"[INFO] Total chains: {len(test_samples)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--esm_size", default="650M", choices=["150M", "650M", "3B"])
    ap.add_argument("--cache", default="/kaggle/working/graphbepi_cache")
    args = ap.parse_args()

    main(args)
