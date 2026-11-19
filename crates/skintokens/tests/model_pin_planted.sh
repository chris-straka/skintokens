#!/usr/bin/env bash
# Proof that the model pin can fail (skintokens-11): the pin passes on the real checkpoint and
# fails, saying the output drifted (not a load error), on a copy with one weight tensor
# perturbed: in the checkpoint's largest tensor, every 400th element's sign flipped (the top
# bit of every 400th 4-byte group: one f32, or one of two 16-bit values), rewritten as a valid
# zip. Needs the checkpoint (--data skintokens-weights) and about 3 GB of free disk.
#   fleet run --snapshot --cores 6 --data skintokens-weights -- bash crates/skintokens/tests/model_pin_planted.sh
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
SRC=${SKINTOKENS_PIN_CKPT:-${FLEET_DATA_SKINTOKENS_WEIGHTS:-}/grpo_1400.ckpt}
[ -f "$SRC" ] || { echo "FAIL  no checkpoint ($SRC)"; echo "PLANTED FAIL"; exit 1; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
pin() { SKINTOKENS_PIN_CKPT=$1 cargo test --release -p skintokens --test model_pin -- --ignored model_pin > "$T/pin.out" 2>&1; }
pin "$SRC"
if [ $? -eq 0 ]; then echo "PASS  the pin holds on the real checkpoint"; else echo "FAIL  the pin fails on the real checkpoint"; grep -E "drifted|panicked|joint|share|vertex" "$T/pin.out" | head -12 | sed 's/^/      /'; fails=$((fails + 1)); fi
python3 - "$SRC" "$T/perturbed.ckpt" <<'PY' || { echo "FAIL  could not write the perturbed checkpoint"; exit 1; }
import sys, zipfile
src, dst = sys.argv[1:3]
zin = zipfile.ZipFile(src)
infos = zin.infolist()
data = [i for i in infos if "/data/" in i.filename]
big = max(data, key=lambda i: i.file_size)
with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
    for i in infos:
        b = zin.read(i.filename)
        if i.filename == big.filename:
            b = bytearray(b)
            for k in range(3, len(b), 1600):
                b[k] ^= 0x80
            b = bytes(b)
        zout.writestr(i, b, compress_type=zipfile.ZIP_STORED)
print(f"perturbed {big.filename} ({big.file_size} bytes)")
PY
pin "$T/perturbed.ckpt"
code=$?
if [ $code -ne 0 ] && grep -q "drifted from the pin" "$T/pin.out"; then echo "PASS  the pin fails on the perturbed checkpoint, as drift"
else echo "FAIL  perturbed checkpoint: exit $code, $(grep -cE 'drifted from the pin' "$T/pin.out") drift lines"; grep -E "drifted|panicked|error|Error" "$T/pin.out" | head -8 | sed 's/^/      /'; fails=$((fails + 1)); fi
grep -E "^(joint|a vertex|vertices|joints:)" "$T/pin.out" | head -6 | sed 's/^/      /'
if [ "$fails" -eq 0 ]; then echo "PLANTED PASS"; exit 0; else echo "PLANTED FAIL ($fails)"; exit 1; fi
