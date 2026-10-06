"""Dump upstream SkinTokens intermediates for the Rust parity checks.

Run from $SKINTOKENS_HOME/upstream with the pinned venv:

    SKINTOKENS_DEVICE=cpu ../.venv/bin/python /path/to/parity/dump_ref.py IN.glb OUTDIR --mode rig|skin --precision f32
    SKINTOKENS_DEVICE=mps ../.venv/bin/python /path/to/parity/dump_ref.py IN.glb OUTDIR --mode rig|skin --precision bf16

Writes .npy files into OUTDIR (all float arrays float32):

  asset_vertices/faces/vertex_normals/face_normals   bpy-loaded mesh (model frame, before normalization)
  asset_joints/parents                                 (skin mode) skeleton as loaded
  norm_vertices, norm_joints                           after the predict transform (trim + affine)
  sampled                                              (54000, 6) the model input [vertices, normals]
  start_tokens                                         prompt token ids
  enc_choice, enc_query                                Michelangelo: 2048 random picks, 512 final query indices
  enc_latents                                          (512, 512) encoder output before output_proj
  mesh_cond                                            (512, 896) after output_proj
  vae_query                                            (384,) indices into `sampled` of the VAE cond queries
  cond_latents                                         (384, 512)
  prefix_ids, prefix_logits                            logits (len, vocab) for start tokens + a fixed continuation
  greedy_ids                                           num_beams=1, greedy, up to 400 new tokens
  beam_ids                                             num_beams=10, no sampling (deterministic beam search)
  dec_ids, dec_skin                                    VAE decoder: skin (54000, J) for fixed token ids
  sample_ids                                           one sampled run (demo settings) for reference

Every network piece runs with the same sampled indices so the Rust side can
reproduce it exactly. f32 = model.float() on CPU, no autocast; bf16 = the
production path (bf16 weights, autocast on MPS).
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "compat"))

from src.data.transform import Transform  # noqa: E402
from src.device import DEVICE  # noqa: E402
from src.model.michelangelo.models.tsal import sal_perceiver  # noqa: E402
from src.model.skin_vae.autoencoders import skin_fsq_cvae_model as cvae  # noqa: E402
from src.model.tokenrig import decode, encode_mesh_cond, get_logits_processor  # noqa: E402
from src.rig_package.parser.bpy import BpyParser  # noqa: E402
from src.server.spec import get_model  # noqa: E402
from src.tokenizer.parse import get_tokenizer  # noqa: E402
from src.tokenizer.spec import TokenizeInput  # noqa: E402

CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("out")
ap.add_argument("--mode", choices=["rig", "skin"], default="rig")
ap.add_argument("--precision", choices=["f32", "bf16"], default="f32")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-gen", action="store_true", help="no greedy/beam/sampled runs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def save(name, x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype.kind == "f":
        x = x.astype(np.float32)
    elif x.dtype.kind in "iub":
        x = x.astype(np.int64)
    np.save(os.path.join(a.out, name + ".npy"), x)


np.random.seed(a.seed)
torch.manual_seed(a.seed)
model = get_model(CKPT)
model.eval()
if a.precision == "f32":
    model = model.float()
dev = DEVICE
tokenizer = get_tokenizer(**model.tokenizer_config)
transform = Transform.parse(**model.transform_config["predict_transform"])


def ctx():
    if a.precision == "f32":
        return torch.autocast(device_type=dev, enabled=False)
    return torch.autocast(device_type=dev, dtype=torch.bfloat16)


# ------------------------------------------------------------------ inputs
asset = BpyParser.load(filepath=a.input)
asset.cls = "articulation"
asset.path = a.input
save("asset_vertices", asset.vertices)
save("asset_faces", asset.faces)
save("asset_vertex_normals", asset.vertex_normals)
save("asset_face_normals", asset.face_normals)
if a.mode == "skin":
    save("asset_joints", asset.joints)
    save("asset_parents", asset.parents)
else:
    asset.matrix_local = None
    asset.parents = None
    asset.joint_names = None
    asset.skin = None
np.random.seed(a.seed)
transform.apply(asset=asset)
save("norm_vertices", asset.vertices)
tokens = None
if a.mode == "skin":
    save("norm_joints", asset.joints)
    save("norm_parents", asset.parents)
    tokens = tokenizer.tokenize(TokenizeInput(joints=asset.joints, parents=asset.parents, cls=asset.cls,
                                              joint_names=asset.joint_names))
v = torch.from_numpy(asset.sampled_vertices).float().to(dev)
n = torch.from_numpy(asset.sampled_normals).float().to(dev)
save("sampled", torch.cat([v, n], -1))
if tokens is not None:
    start = tokens.tolist()
else:
    start = model.make_start_tokens(cls=["articulation"])[0]
save("start_tokens", np.array(start))

# ------------------------------------------------------- capture sampling
rec = {}
_fps = sal_perceiver.fps


def fps_mich(x, batch, ratio, random_start=False):
    idx = _fps(x, batch, ratio, random_start)
    rec["enc_fps"] = idx.cpu().numpy()
    return idx


sal_perceiver.fps = fps_mich
_choice_rng = np.random.default_rng


class Rng:
    def __init__(self, seed=None):
        self.r = _choice_rng(seed)

    def choice(self, *args, **kw):
        out = self.r.choice(*args, **kw)
        rec.setdefault("choices", []).append(out)
        return out


sal_perceiver.np.random.default_rng = Rng  # same numpy module object for both files


def sample_features(self, x, num_tokens=128, seed=None):
    rng = np.random.default_rng(12345)
    indices = rng.choice(x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1])
    pts = x[0, indices]
    fidx = _fps(pts[:, :3], torch.zeros(len(indices), dtype=torch.long, device=x.device), 0.25, False)
    rec["vae_query"] = indices[fidx.cpu().numpy()]
    return pts[fidx].unsqueeze(0)


cvae.SkinFSQCVAEModel._sample_features = sample_features

with torch.no_grad(), ctx():
    cond = torch.cat([v, n], -1).unsqueeze(0)
    _, cond_latents = model.vae.model._encode(x=None, cond=cond, num_tokens=model.tokens_per_skin,
                                              cond_tokens=model.tokens_skin_cond, return_z=False)
    shape_embed, latents, _, _ = model.mesh_encoder.encode_latents(pc=v.unsqueeze(0), feats=n.unsqueeze(0))
    mesh_cond = model.output_proj(latents)
np.random.default_rng = _choice_rng
choice = [c for c in rec["choices"] if len(c) == 2048][0]
save("enc_choice", choice)
save("enc_query", choice[rec["enc_fps"]])
save("enc_latents", latents[0])
save("mesh_cond", mesh_cond[0])
save("vae_query", rec["vae_query"])
save("cond_latents", cond_latents[0])

emb = model.transformer.get_input_embeddings()


def logits_for(ids):
    with torch.no_grad(), ctx():
        e = emb(torch.tensor(ids, device=dev).unsqueeze(0))
        out = model.transformer(inputs_embeds=torch.cat([mesh_cond, e], 1))
    return out.logits[0, mesh_cond.shape[1] - 1:].float()


def generate(**kw):
    start_t = torch.tensor(start, device=dev)
    with torch.no_grad(), ctx():
        e = emb(start_t.unsqueeze(0))
        res = model.transformer.generate(
            inputs_embeds=torch.cat([mesh_cond, e], 1), bos_token_id=tokenizer.bos, eos_token_id=model.eos,
            pad_token_id=tokenizer.pad,
            logits_processor=get_logits_processor(tokenizer=tokenizer, eos=model.eos,
                                                  tokens_per_skin=model.tokens_per_skin, start_tokens=start_t),
            max_length=2048, num_return_sequences=1, **kw)
    return start + res[0].tolist()


if not a.skip_gen:
    greedy = generate(do_sample=False, num_beams=1, repetition_penalty=2.0, max_new_tokens=400)
    save("greedy_ids", np.array(greedy))
    beam = generate(do_sample=False, num_beams=10, repetition_penalty=2.0)
    save("beam_ids", np.array(beam))
    torch.manual_seed(a.seed)
    sampled_run = generate(do_sample=True, num_beams=10, top_k=5, top_p=0.95, temperature=1.0, repetition_penalty=2.0)
    save("sample_ids", np.array(sampled_run))
    full = beam
else:
    full = None

# logits for a fixed prefix: the prompt plus the first 40 tokens of the beam run
prefix = (full[: len(start) + 40] if full is not None else start)
save("prefix_ids", np.array(prefix))
save("prefix_logits", logits_for(prefix))

# VAE decoder for fixed token ids: the beam run's skin tokens (first 2 joints)
if full is not None and tokenizer.eos in full:
    ids = torch.tensor(full, device=dev)
    with torch.no_grad(), ctx():
        d = decode(cond=cond[0], cond_latents=cond_latents[0], inputs_ids=ids, tokenizer=tokenizer,
                   tokens_per_skin=model.tokens_per_skin, vae=model.vae)
    if d["skin_pred"] is not None:
        save("dec_ids", ids)
        save("dec_skin", d["skin_pred"])
print("dumped", a.out)
