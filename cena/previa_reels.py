#!/usr/bin/env python3
"""Previa de enquadramento no Reels: selo ANIMACAO ILUSTRATIVA, pilula ERRADO e legenda de teste em 2 linhas
na especificacao da skill videos-instagram (Montserrat 700 60 px, contorno 6 px #1E3A2F, base em y = 1250).
Baixa a Montserrat do repositorio oficial do Google Fonts se faltar.
Uso: python3 -I previa_reels.py --cfg c.json entrada.mp4 saida.mp4"""
import argparse, json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument('--cfg', required=True); ap.add_argument('ent'); ap.add_argument('sai')
A = ap.parse_args(); ent, sai = A.ent, A.sai; PV = json.load(open(A.cfg))['previa']
W, H, FPS = 1080, 1920, 30
FONTE = os.path.expanduser('~/.fonts/Montserrat.ttf')
if not os.path.exists(FONTE):
    os.makedirs(os.path.dirname(FONTE), exist_ok=True)
    subprocess.run(['curl', '-sSL', '-o', FONTE, 'https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf'], check=True)

def fonte(px, peso):
    f = ImageFont.truetype(FONTE, px); f.set_variation_by_axes([peso]); return f

def ease_out(x):
    x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3

def pilula(texto, px, peso, fundo, cor, rastreio=0.08):
    f = fonte(px, peso)
    letras = [(c, f.getlength(c)) for c in texto]
    larg = sum(l for _, l in letras) + rastreio * px * (len(letras) - 1)
    pad_x, pad_y = int(px * 0.7), int(px * 0.42)
    asc, desc = f.getmetrics()
    im = Image.new('RGBA', (int(larg + 2 * pad_x), asc + desc + 2 * pad_y), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, im.width - 1, im.height - 1], radius=im.height // 2, fill=fundo)
    x = pad_x
    for c, l in letras:
        d.text((x, pad_y), c, font=f, fill=cor); x += l + rastreio * px
    return im

selo = pilula('ANIMAÇÃO ILUSTRATIVA', 36, 600, (20, 24, 22, 153), (242, 242, 235, 255))
errado = pilula('ERRADO', 36, 700, (224, 70, 47, 235), (255, 255, 255, 255))

def legenda(linhas, destaque=None):
    f = fonte(60, 700); asc, desc = f.getmetrics(); alt = int(60 * 1.18)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    base = 1250
    for k, ln in enumerate(linhas):
        y = base - (len(linhas) - 1 - k) * alt - asc
        larg = f.getlength(ln); x = (W - larg) / 2
        sombra = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ds = ImageDraw.Draw(sombra)
        ds.text((x, y + 4), ln, font=f, fill=(0, 0, 0, 115), stroke_width=6, stroke_fill=(0, 0, 0, 115))
        im = Image.alpha_composite(sombra, im); d = ImageDraw.Draw(im)
        d.text((x, y), ln, font=f, fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(30, 58, 47, 255))
    return im

leg = legenda(PV['legenda'])

def quadro_overlay(t):
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    e = ease_out(t / 0.4)
    s = selo.copy(); s.putalpha(s.getchannel('A').point(lambda v: int(v * e)))
    ov.alpha_composite(s, (PV['selo'][0], int(PV['selo'][1] + 30 * (1 - e))))
    if t >= PV['errado_t']:                                       # quando o liquido toca o tampo
        e2 = ease_out((t - PV['errado_t']) / 0.4)
        r = errado.copy(); r.putalpha(r.getchannel('A').point(lambda v: int(v * e2)))
        ov.alpha_composite(r, (int(PV['errado'][0] - 30 * (1 - e2)), PV['errado'][1]))
    if 0.5 <= t <= 7.6:
        e3 = ease_out((t - 0.5) / 0.12) * (1 - ease_out((t - 7.4) / 0.2))
        l2 = leg.copy(); l2.putalpha(l2.getchannel('A').point(lambda v: int(v * e3)))
        ov.alpha_composite(l2)
    return np.asarray(ov).astype(np.float32)

dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', ent, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', ent, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                        '-c:a', 'copy', '-movflags', '+faststart', sai], stdin=subprocess.PIPE)
i = 0
while True:
    raw = dec.stdout.read(W * H * 3)
    if len(raw) < W * H * 3:
        break
    fr = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.float32)
    ov = quadro_overlay(i / FPS); a = ov[..., 3:4] / 255
    out = ov[..., :3] * a + fr * (1 - a)
    enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes()); i += 1
enc.stdin.close(); enc.wait(); dec.wait()
print('ok', sai, i, 'quadros')
