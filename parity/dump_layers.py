"""Per-module outputs (forward hooks) of the shape encoder, VAE condition
encoder and decoder on a dump_ref.py directory's inputs (fp32, CPU)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "compat"))
from src.server.spec import get_model
d = sys.argv[1]; out = os.path.join(d, "layers"); os.makedirs(out, exist_ok=True)
m = get_model("experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt").float().eval()
S = torch.from_numpy(np.load(f"{d}/sampled.npy"))
eq = np.load(f"{d}/enc_query.npy"); vq = np.load(f"{d}/vae_query.npy")
cnt = {}
def hook(name):
    def f(mod, inp, outp):
        k = cnt.get(name, 0); cnt[name] = k + 1
        o = outp[0] if isinstance(outp, tuple) else outp
        np.save(f"{out}/{name}.{k}.npy", o.detach().float().numpy()[0])
    return f
enc = m.mesh_encoder.encoder
enc.input_proj.register_forward_hook(hook("enc.input_proj"))
enc.cross_attn.register_forward_hook(hook("enc.cross"))
for i, b in enumerate(enc.self_attn.resblocks): b.register_forward_hook(hook(f"enc.block{i}"))
ce = m.vae.model.cond_encoder
ce.proj_in.register_forward_hook(hook("vae.proj_in"))
for i, b in enumerate(ce.blocks): b.register_forward_hook(hook(f"vae.block{i}"))
with torch.no_grad(), torch.autocast("cpu", enabled=False):
    # encoder with fixed queries: replicate CrossAttentionEncoder._forward
    pc, ft = S[None, :, :3], S[None, :, 3:]
    data = enc.input_proj(torch.cat([enc.fourier_embedder(pc), ft], -1))
    q = enc.input_proj(torch.cat([enc.fourier_embedder(pc[:, eq]), ft[:, eq]], -1))
    lat = enc.ln_post(enc.self_attn(enc.cross_attn(q, data)))
    np.save(f"{out}/enc.final.npy", lat[0].numpy())
    emb = m.vae.model.embedder
    x = S[None]
    kv = torch.cat([emb(x[..., :3]), x[..., 3:]], -1); qq = kv[:, vq]
    np.save(f"{out}/vae.embed_q.npy", qq[0].numpy())
    h = ce(qq, kv)
    np.save(f"{out}/vae.final.npy", m.vae.model.cond_quant(h)[0].numpy())
print(sorted(os.listdir(out)))
# decoder, first joint of dec_ids
ids = np.load(f"{d}/dec_ids.npy"); w = int(np.where(ids == 258)[0][0])
idx = torch.from_numpy(ids[w + 1:w + 5] - 267)[None]
vae = m.vae
with torch.no_grad(), torch.autocast("cpu", enabled=False):
    z = vae.model.FSQ.indices_to_codes(idx).reshape(1, 4, -1)
    np.save(f"{out}/dec.z.npy", z[0].numpy())
    cl = torch.from_numpy(np.load(f"{d}/cond_latents.npy"))[None]
    zz = vae.model.post_quant(torch.cat([z, cl], 1)); np.save(f"{out}/dec.post_quant.npy", zz[0].numpy())
    h = zz
    for b in vae.model.decoder.blocks[:-1]: h = b(h)
    np.save(f"{out}/dec.kv.npy", h[0].numpy())
    pos, feat = S[None, :, :3], S[None, :, 3:]
    q = vae.model.decoder.proj_query(torch.cat([vae.model.embedder(pos), feat], -1))
    np.save(f"{out}/dec.q.npy", q[0, :2000].numpy())
    l = vae.model.decoder.blocks[-1](q, encoder_hidden_states=h)
    np.save(f"{out}/dec.cross.npy", l[0, :2000].numpy())
    lo = vae.model.decoder.proj_out(vae.model.decoder.norm_out(l))
    np.save(f"{out}/dec.logits.npy", lo[0, :, 0].numpy())
print("decoder layers dumped")
