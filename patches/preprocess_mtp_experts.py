"""Split fused Qwen3.5-MTP MoE experts into per-expert HF tensors so slime's
(#1702) Qwen3.5 MTP bridge converter can ingest them.

源 checkpoint 把 MTP experts 存成 FUSED:
    mtp.layers.0.mlp.experts.gate_up_proj  [E, 2F, H]
    mtp.layers.0.mlp.experts.down_proj     [E, H, F]
slime (#1702) bridge 期望 INDIVIDUAL:
    mtp.layers.0.mlp.experts.{i}.gate_proj.weight  [F, H]
    mtp.layers.0.mlp.experts.{i}.up_proj.weight    [F, H]
    mtp.layers.0.mlp.experts.{i}.down_proj.weight  [H, F]

输出目录:软链所有原始 shard(fused 张量物理仍在,但从 index 删除->加载器忽略),
拷贝 config/tokenizer,写一个新 shard 放 split 后的张量,并重写 index。

用法:
    python preprocess_mtp_experts.py <SRC_HF_DIR> <DST_HF_DIR>
之后把 slime 转换的 --hf-checkpoint 指向 <DST_HF_DIR> 生成 *_torch_dist_mtp,
而 sglang 推理的 --hf-checkpoint 仍指向原始 fused <SRC_HF_DIR>。
"""

import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/model_dist/Qwen3.5-35B-A3B"
DST = sys.argv[2] if len(sys.argv) > 2 else "/root/model_dist/Qwen3.5-35B-A3B_mtp_individual"
NEW_SHARD = "model-mtp-experts-individual.safetensors"

FUSED = ["mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"]

os.makedirs(DST, exist_ok=True)

index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wmap = index["weight_map"]

# --- infer F (moe_intermediate_size) from the fused gate_up shape ---
gu_shard = wmap["mtp.layers.0.mlp.experts.gate_up_proj"]
with safe_open(os.path.join(SRC, gu_shard), framework="pt") as f:
    gu_shape = f.get_slice("mtp.layers.0.mlp.experts.gate_up_proj").get_shape()
E, twoF, H = gu_shape
assert twoF % 2 == 0
F = twoF // 2
print(f"experts={E}  ffn(F)={F}  hidden(H)={H}")

# --- 1. mirror the source dir: symlink shards, copy the rest ---
for fn in os.listdir(SRC):
    s = os.path.join(SRC, fn)
    d = os.path.join(DST, fn)
    if fn == "model.safetensors.index.json":
        continue
    if os.path.isdir(s):
        continue
    if fn.endswith(".safetensors"):
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(os.path.abspath(s), d)
    else:
        if not os.path.exists(d):
            shutil.copy2(s, d)

# --- 2. build per-expert tensors ---
new_tensors = {}
for fused_name in FUSED:
    shard = wmap[fused_name]
    with safe_open(os.path.join(SRC, shard), framework="pt") as f:
        w = f.get_tensor(fused_name)
    if "gate_up_proj" in fused_name:
        # w: [E, 2F, H] -> gate=[:F], up=[F:]
        for i in range(w.shape[0]):
            gu = w[i]
            new_tensors[f"mtp.layers.0.mlp.experts.{i}.gate_proj.weight"] = gu[:F, :].contiguous().clone()
            new_tensors[f"mtp.layers.0.mlp.experts.{i}.up_proj.weight"] = gu[F:, :].contiguous().clone()
    else:  # down_proj: [E, H, F]
        for i in range(w.shape[0]):
            new_tensors[f"mtp.layers.0.mlp.experts.{i}.down_proj.weight"] = w[i].contiguous().clone()

print(f"created {len(new_tensors)} individual tensors (expect {E*3})")
assert len(new_tensors) == E * 3

save_file(new_tensors, os.path.join(DST, NEW_SHARD), metadata={"format": "pt"})

# --- 3. rewrite index: drop fused, add individual -> new shard ---
new_map = {k: v for k, v in wmap.items() if k not in FUSED}
for name in new_tensors:
    new_map[name] = NEW_SHARD
index["weight_map"] = new_map
index.get("metadata", {}).pop("total_size", None)
json.dump(index, open(os.path.join(DST, "model.safetensors.index.json"), "w"))

# --- verify ---
chk = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]
assert all(fn not in chk for fn in FUSED), "fused names still in index!"
assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in chk
assert f"mtp.layers.0.mlp.experts.{E-1}.down_proj.weight" in chk
print("OK -> wrote", DST)
print("  individual mtp keys in index:", sum(1 for k in chk if "mtp.layers.0.mlp.experts." in k and ".weight" in k))
