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


def main(args):
    root = args.root
    ensure_folders(root)

    device = choose_device(args.gpu)
    print(f"[INFO] Device: {device}")

    esm_model = load_esm(args.esm_size, device)

    samples = []

    pdb_dir = args.pdb_dir
    label_dir = args.label_dir

    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])

    for pdb_file in tqdm(pdb_files, desc="Building DiscoTope test set"):
        name = pdb_file.replace(".pdb", "")
        pdb_id = name.lower()

        pdb_path = os.path.join(pdb_dir, pdb_file)
        label_path = os.path.join(label_dir, name + ".pt")

        if not os.path.exists(label_path):
            print(f"[WARN] Missing label: {name}")
            continue

        try:
            labels = torch.load(label_path)  # shape: [L]
        except Exception:
            print(f"[WARN] Cannot load label: {name}")
            continue

        # assume single chain A
        ch = "A"

        try:
            ok = extract_chain(root, pdb_path, ch, is_file=True)
            if not ok:
                continue
        except Exception:
            continue

        obj = chain()
        obj.protein_name = pdb_id
        obj.chain_name = ch
        obj.name = f"{pdb_id}_{ch}"

        try:
            process_chain(obj, root, obj.name, esm_model, device)
        except Exception:
            traceback.print_exc()
            continue

        # assign labels
        for i, y in enumerate(labels):
            if y == 1:
                try:
                    res_id = str(i + 1)
                    res_name = obj.residue[res_id]
                    obj.update(res_id, res_name)
                except Exception:
                    pass

        samples.append(obj)

    out_path = os.path.join(root, "test.pkl")
    with open(out_path, "wb") as f:
        pk.dump(samples, f)

    print(f"[DONE] test.pkl saved: {out_path}")
    print(f"[INFO] Total samples: {len(samples)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--esm_size", default="650M")
    args = ap.parse_args()

    main(args)
