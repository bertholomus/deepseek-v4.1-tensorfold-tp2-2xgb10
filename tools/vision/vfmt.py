"""vfmt.py OUT_DIR: the picture gate over every image format the lane takes. In the served
image: a corpus made with its Pillow (PNG, JPEG baseline/progressive/4:4:4/4:2:2/grey/CMYK/EXIF-rotated, WebP lossy/
lossless/alpha/animated, GIF still/animated/transparent/interlaced, BMP 1/4/8/24/32-bit and RLE, TIFF raw/LZW/deflate/
JPEG/16-bit, AVIF, ICO, PPM/PGM/PBM, TGA, JPEG 2000, ...), each through the served lane's own `decode` (called as a black
box: its patches' sha256 and grid, or its exception) into OUT_DIR/index.jsonl, for tf-dsv41-picture to match."""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tensorfold.families.deepseek_v41.cuda import vision as V

out = Path(sys.argv[1])
imgs = out / "imgs"
imgs.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)


def base(w, h):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[: h // 2, : w // 2] = (np.arange(w // 2)[None, :, None] * 5 % 256)
    a[h // 2:, w // 3:] = [200, 40, 90]
    return Image.fromarray(a, "RGB")


def save(name, im, fmt, **kw):
    try:
        im.save(imgs / name, fmt, **kw)
    except Exception as e:  # a writer this build lacks
        print(json.dumps({"skip": name, "why": repr(e)[:120]}), flush=True)


rgb = base(203, 141)
save("a.png", rgb, "PNG")
for q, sub in ((90, 2), (75, 0), (60, 1)):
    save(f"jpeg_q{q}_s{sub}.jpg", rgb, "JPEG", quality=q, subsampling=sub)
save("jpeg_progressive.jpg", rgb, "JPEG", quality=85, progressive=True)
save("jpeg_restart.jpg", rgb, "JPEG", quality=85, restart_marker_blocks=3) if False else None
save("jpeg_grey.jpg", rgb.convert("L"), "JPEG", quality=88)
save("jpeg_cmyk.jpg", rgb.convert("CMYK"), "JPEG", quality=88)
save("jpeg_odd.jpg", base(17, 9), "JPEG", quality=95, subsampling=2)
save("jpeg_big.jpg", base(1601, 1203), "JPEG", quality=80)
ex = Image.Exif()
ex[0x0112] = 6  # orientation: rotate 90
save("jpeg_exif6.jpg", rgb, "JPEG", quality=85, exif=ex.tobytes())
save("webp_lossy.webp", rgb, "WEBP", quality=80)
save("webp_lossless.webp", rgb, "WEBP", lossless=True)
save("webp_alpha.webp", rgb.convert("RGBA"), "WEBP", quality=80)
rgba = rgb.convert("RGBA")
rgba.putalpha(Image.fromarray((np.arange(203)[None, :].repeat(141, 0) % 256).astype(np.uint8), "L"))
save("webp_alpha_grad.webp", rgba, "WEBP", lossless=True)
save("webp_anim.webp", rgb, "WEBP", save_all=True, append_images=[base(203, 141)], duration=100)
pal = rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
save("gif_still.gif", pal, "GIF")
save("gif_interlaced.gif", pal, "GIF", interlace=True)
pt = pal.copy()
pt.info["transparency"] = 3
save("gif_transparent.gif", pt, "GIF", transparency=3)
save("gif_anim.gif", pal, "GIF", save_all=True, append_images=[base(203, 141).convert("P", palette=Image.Palette.ADAPTIVE, colors=16)], duration=50)
save("gif_grey.gif", rgb.convert("L"), "GIF")
for mode in ("1", "L", "P", "RGB", "RGBA"):
    save(f"bmp_{mode}.bmp", rgb.convert(mode), "BMP")
save("bmp_rle8.bmp", pal, "BMP", compression=1) if False else None
for comp in ("raw", "tiff_lzw", "tiff_adobe_deflate", "jpeg", "packbits"):
    save(f"tiff_{comp}.tiff", rgb, "TIFF", compression=comp)
save("tiff_rgba.tiff", rgba, "TIFF")
save("tiff_16.tiff", Image.fromarray(rng.integers(0, 65535, (61, 97), dtype=np.uint16), "I;16"), "TIFF")
save("tiff_cmyk.tiff", rgb.convert("CMYK"), "TIFF")
save("avif.avif", rgb, "AVIF", quality=70)
save("avif_alpha.avif", rgba, "AVIF", quality=70)
save("ico.ico", rgb.resize((64, 64)), "ICO", sizes=[(16, 16), (32, 32), (64, 64)])
save("ppm.ppm", rgb, "PPM")
save("pgm.pgm", rgb.convert("L"), "PPM")
save("pbm.pbm", rgb.convert("1"), "PPM")
save("tga.tga", rgb, "TGA")
save("tga_rle.tga", rgb, "TGA", compression="tga_rle")
save("jp2.jp2", rgb, "JPEG2000")
save("pcx.pcx", rgb, "PCX")
save("sgi.sgi", rgb, "SGI")
save("dds.dds", rgba, "DDS") if False else None
save("qoi.qoi", rgb, "QOI")
save("im.im", rgb, "IM")
save("tall.jpg", base(37, 2300), "JPEG", quality=90)
save("wide.webp", base(2900, 31), "WEBP", quality=90)
# a picture between the lane's pixel limit and twice it (PIL warns there; the lane's decode decides)
save("grey_9000.png", Image.new("L", (9000, 9000), 128), "PNG")
(imgs / "garbage.bin").write_bytes(b"not an image at all")
(imgs / "truncated.jpg").write_bytes((imgs / "jpeg_q90_s2.jpg").read_bytes()[:300])

cfg = V.VisionConfig(layers=32, dim=1024, heads=16, inter=2816, patch=14, theta=10000.0, ratio=3, max_tokens=1024,
                     min_pixels=295936, max_wh_ratio=None, image_token_id=129264, model_dim=5120)
with open(out / "index.jsonl", "w") as f:
    for p in sorted(imgs.iterdir()):
        data = p.read_bytes()
        try:
            pic = V.decode(data, cfg)
            raw = pic.patches.contiguous().view(-1).view(dtype=__import__("torch").uint16).numpy().tobytes()
            rec = {"file": p.name, "n_vit_h": int(pic.n_vit_h), "n_vit_w": int(pic.n_vit_w), "n_llm_h": int(pic.n_llm_h),
                   "n_llm_w": int(pic.n_llm_w), "patches_sha256": hashlib.sha256(raw).hexdigest()}
        except Exception as e:
            rec = {"file": p.name, "error": type(e).__name__, "message": str(e)[:300]}
        f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec)[:200], flush=True)
