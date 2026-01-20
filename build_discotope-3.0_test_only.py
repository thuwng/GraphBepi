import os
import pickle as pk
import traceback
from tqdm import tqdm
import argparse

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


def list_pdb_chains(pdb_root):
    pairs = []
    for fn in os.listdir(pdb_root):
        if not fn.endswith(".pdb"):
            continue
        if "_renum" in fn:
            continue
        if "_" not in fn:
            continue

        name = fn.replace(".pdb", "")
        pdb, ch = name.split("_")
        pairs.append((pdb.lower(), ch.upper()))
    return sorted(pairs)


def main(args):
    if args.cache:
        os.environ["TORCH_HOME"] = args.cache
        os.makedirs(args.cache, exist_ok=True)

    root = args.root
    ensure_folders(root)

    device = choose_device(args.gpu)
    print(f"[INFO] Device: {device}")

    esm_model = load_esm(args.esm_size, device)

    pdb_chains = list_pdb_chains(os.path.join(root, "PDB"))
    print(f"[INFO] Found {len(pdb_chains)} PDB-chain pairs")

    test_samples = []

    for pdb, ch in tqdm(pdb_chains, desc="Building DiscoTope test set"):
        ok = extract_chain(root, pdb, ch)
        if not ok:
            print(f"[WARN] extract_chain failed: {pdb}_{ch}")
            continue

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

        obj.label = torch.zeros(obj.length, dtype=torch.long)
        test_samples.append(obj)

    out_path = os.path.join(root, "test.pkl")
    with open(out_path, "wb") as f:
        pk.dump(test_samples, f)

    print(f"[DONE] test.pkl written: {out_path}")
    print(f"[INFO] Total chains: {len(test_samples)}")


# =========================
# ENTRY POINT (BẮT BUỘC)
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--esm_size", type=str, default="650M")
    parser.add_argument("--cache", type=str, default=None)

    args = parser.parse_args()
    main(args)
