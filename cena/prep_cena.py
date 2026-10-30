#!/usr/bin/env python3
"""Preparacao de cena com liquido despejado de um recipiente (frigideira, panela, balde).
  camadas   --cena img --cfg c.json --dir D   recorte refinado (rembg + GrabCut com sementes), fio de liquido
                                              em camada propria, fundo reconstruido (LaMa)
  rig       --cfg c.json --dir D               aplica os ajustes de esqueleto do cfg ao rig do anima.py
  recipiente --cfg c.json --dir D              tira o liquido de dentro do recipiente e reconstroi a borda
  calibrar  --cfg c.json --dir D               ajusta o modelo 3D do recipiente a poca original (beta0, h0, V0)
Saidas em D: cena1080.png, personagem.png, fio.png, fundo.png, anima/ (do anima.py), cena/ (pronta para animar),
mask_liquido_recipiente.npy, elipse_ext.json, calib.json."""
import argparse, json, math, os, shutil, subprocess, sys
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.expanduser('~/anima'))
import anima
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipiente3d as R3


def cor_liquido(rgb, c):
    r, b = rgb[..., 0].astype(int), rgb[..., 2].astype(int)
    return (r > c['r_min']) & (r - b > c['r_menos_b_min']) & (b < c['b_max'])


def maior(m, min_px=0):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return np.zeros(m.shape, bool)
    if min_px:
        return np.isin(lab, [i for i in range(1, n) if st[i, 4] > min_px])
    return lab == (1 + np.argmax(st[1:, 4]))


def cmd_camadas(a, cfg):
    W, H = cfg['quadro']; d = a.dir; os.makedirs(d, exist_ok=True)
    cena = np.asarray(anima.ajustar(Image.open(a.cena).convert('RGB'), W, H))
    Image.fromarray(cena).save(f'{d}/cena1080.png')
    m = anima.recortar(cena); np.save(f'{d}/mask_rembg.npy', m)
    rc = cfg['recorte']
    gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8); gc[m > 40] = cv2.GC_PR_FGD
    gc[cv2.erode((m > 200).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = cv2.GC_FGD
    sem = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1, e in rc['linhas_frente']:                       # partes finas que o rembg deixou transparentes
        cv2.line(sem, (x0, y0), (x1, y1), 1, e)
    gc[sem > 0] = cv2.GC_FGD
    prov = cv2.dilate(((m > 30) | (sem > 0)).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    gc[~prov] = cv2.GC_BGD
    for x0, y0, x1, y1 in rc['caixas_fundo']:                           # objetos do fundo colados no personagem
        gc[y0:y1, x0:x1] = cv2.GC_BGD
    x0, y0, x1, y1 = rc['caixa_grabcut']
    sub = gc[y0:y1, x0:x1].copy(); bgm = np.zeros((1, 65)); fgm = np.zeros((1, 65))
    cv2.grabCut(np.ascontiguousarray(cena[y0:y1, x0:x1]), sub, None, bgm, fgm, 6, cv2.GC_INIT_WITH_MASK)
    res = np.zeros((H, W), np.uint8); res[y0:y1, x0:x1] = ((sub == 1) | (sub == 3))
    keep = maior(res)
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.0)
    alpha = np.clip(np.where(keep & (m > 200), 1.0, soft), 0, 1)
    # fio de liquido em camada propria
    f = cfg['fio']; fx0, fy0, fx1, fy1 = f['caixa']
    reg = np.zeros((H, W), bool); reg[fy0:fy1, fx0:fx1] = True
    fio = maior(reg & cor_liquido(cena, cfg['liquido_cor']) & (alpha > 0.3))
    fio = cv2.morphologyEx(fio.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    yy = np.arange(H)[:, None]
    fa = cv2.GaussianBlur(fio.astype(np.float32), (0, 0), 0.8) * alpha
    fa_l = fa * np.clip((yy - f['y_entrada']) / (f['y_entrada_fim'] - f['y_entrada']), 0, 1)
    pa = np.clip(alpha * np.where(yy >= f['y_entrada_fim'], 1 - cv2.GaussianBlur(fio.astype(np.float32), (0, 0), 1.2), 1), 0, 1)
    # tira o resto do fio que ficou na camada do personagem (fantasma)
    rem = cv2.dilate((fa_l * 255 > 10).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(np.float32)
    pa8 = (pa * 255).astype(np.uint8).astype(np.float32) / 255
    pa8 = pa8 * np.where(yy >= f['y_entrada_fim'], 1 - cv2.GaussianBlur(rem, (0, 0), 0.8), 1)
    pa8[(np.broadcast_to(yy >= f['y_entrada_fim'], pa8.shape)) & (pa8 < 0.1)] = 0
    Image.fromarray(np.dstack([cena, (np.clip(pa8, 0, 1) * 255).astype(np.uint8)])).save(f'{d}/personagem.png')
    Image.fromarray(np.dstack([cena, (fa_l * 255).astype(np.uint8)])).save(f'{d}/fio.png')
    buraco = cv2.dilate((alpha > 0.05).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    Image.fromarray(anima.lama(cena, buraco)).save(f'{d}/fundo.png')
    print('camadas prontas:', d, '| px do fio:', int(fio.sum()))


def cmd_rig(a, cfg):
    p = f'{a.dir}/anima/p1_rig.json'; r = json.load(open(p)); aj = cfg['personagem']['rig_ajustes']
    r['j'].update(aj.get('j', {}))
    for k in ('olhos', 'boca'):
        if k in aj:
            r[k] = aj[k]
    json.dump(r, open(p, 'w'), indent=1)
    subprocess.run([sys.executable, '-I', os.path.expanduser('~/anima/anima.py'), 'checar', '--dir', f'{a.dir}/anima'], check=True)


def ajustar_elipse(alpha, rc):
    a8 = (alpha > 128).astype(np.uint8)
    cs, _ = cv2.findContours(a8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cs])
    x0, y0, x1, y1 = rc['elipse_externa_limites']
    sel = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)]
    for ex0, ey0, ex1, ey1 in rc['elipse_externa_excluir']:
        sel = sel[~((sel[:, 0] > ex0) & (sel[:, 0] < ex1) & (sel[:, 1] > ey0) & (sel[:, 1] < ey1))]
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(sel.astype(np.float32))
    return [float(cx), float(cy), float(MA), float(ma), float(ang)]


def cmd_recipiente(a, cfg):
    d = a.dir; rc = cfg['recipiente']
    p = np.asarray(Image.open(f'{d}/anima/p1.png').convert('RGBA'))
    fundo = np.asarray(Image.open(f'{d}/fundo.png').convert('RGB'))
    H, W = p.shape[:2]; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c = p[..., :3].astype(int)
    oleo = (c[..., 0] > 120) & (c[..., 0] - c[..., 2] > 60) & (c[..., 1] > 60) & (p[..., 3] > 40)
    x0, y0, x1, y1 = rc['caixa_liquido']; reg = np.zeros(oleo.shape, bool); reg[y0:y1, x0:x1] = True
    m = cv2.morphologyEx((oleo & reg).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keep = maior(m, 100); np.save(f'{d}/mask_liquido_recipiente.npy', keep)
    bur = cv2.dilate(keep.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    al = p[..., 3:4] / 255.
    rec = anima.lama((p[..., :3] * al + fundo * (1 - al)).astype(np.uint8), bur)
    e = ajustar_elipse(p[..., 3], rc); json.dump(e, open(f'{d}/elipse_ext.json', 'w'))
    def sil(enc):
        s = np.zeros((H, W), np.uint8)
        cv2.ellipse(s, ((e[0], e[1]), (e[2] - enc, e[3] - enc), e[4]), 1, -1)
        return s
    # 1) cor reconstruida no buraco; alfa = silhueta da elipse externa
    out = p.copy(); out[..., :3] = np.where(bur[..., None], rec, p[..., :3])
    s9 = cv2.GaussianBlur(sil(3).astype(np.float32), (0, 0), 0.9)
    alfa = np.where(bur, s9, p[..., 3].astype(np.float32) / 255)
    bx0, by0, bx1, by1 = rc['fio_resto_caixa']
    rreg = np.zeros((H, W), bool); rreg[by0:by1, bx0:bx1] = True
    alfa = np.where(rreg & (s9 < 0.5), 0, alfa)
    lum = out[..., :3].astype(float) @ [0.299, 0.587, 0.114]
    claro = bur & (lum > 120) & (s9 > 0.05)                                # tom de fundo vazado na borda
    out[..., :3] = np.where(claro[..., None], cv2.inpaint(out[..., :3].copy(), claro.astype(np.uint8), 4, cv2.INPAINT_TELEA), out[..., :3])
    # 2) silhueta lisa perto do bico
    sx0, sy0, sx1, sy1 = rc['suavizar_bico']
    w = np.zeros((H, W), np.float32); w[sy0:sy1, sx0:sx1] = 1; w = cv2.GaussianBlur(w, (0, 0), 4)
    a2 = alfa * (1 - w) + s9 * w
    novo = (a2 > 0.3) & (alfa < 0.3)
    out[..., :3] = np.where(novo[..., None], cv2.inpaint(out[..., :3].copy(), novo.astype(np.uint8), 5, cv2.INPAINT_TELEA), out[..., :3])
    out[..., 3] = (np.clip(a2, 0, 1) * 255).astype(np.uint8)
    # 3) faixa da borda externa repintada pelo perfil do trecho intacto
    ex, ey, ea, eb = rc['elipse_interna']
    ri = np.sqrt(((xx - ex) / ea) ** 2 + ((yy - ey) / eb) ** 2)
    t = np.radians(e[4]); dx, dy = xx - e[0], yy - e[1]
    u = dx * np.cos(t) + dy * np.sin(t); v = -dx * np.sin(t) + dy * np.cos(t)
    ro = np.sqrt((u / (e[2] / 2)) ** 2 + (v / (e[3] / 2)) ** 2)
    banda = (ri > 1.0) & (ro < 1.0)
    rn = np.clip((ri - 1.0) / np.maximum(ri - 1.0 + (1.0 - ro), 1e-3), 0, 1)
    angp = np.degrees(np.arctan2(yy - ey, xx - ex))
    f0, f1 = rc['faixa_fonte_graus']; t0, t1 = rc['faixa_alvo_graus']
    fonte = banda & (angp > f0) & (angp < f1) & (p[..., 3] > 200)
    bins = np.linspace(0, 1, 13); prof = []
    for i in range(12):
        mm = fonte & (rn >= bins[i]) & (rn < bins[i + 1]); prof.append(out[..., :3][mm].mean(0) if mm.sum() > 3 else None)
    prof = np.array([q if q is not None else prof[k - 1] for k, q in enumerate(prof)])
    alvo = banda & (angp > t0) & (angp <= t1) & (yy > rc['faixa_alvo_y_min'])
    cor = prof[np.clip((rn * 12).astype(int), 0, 11)] + np.random.default_rng(3).normal(0, 2.0, (H, W, 3))
    wb = cv2.GaussianBlur(alvo.astype(np.float32), (0, 0), 1.5)[..., None]
    out[..., :3] = np.clip(out[..., :3] * (1 - wb) + cor * wb, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(f'{d}/personagem_sem_liquido.png')
    cd = f'{d}/cena'; os.makedirs(cd, exist_ok=True)
    Image.fromarray(out).save(f'{cd}/p1.png')
    for nome in ('p1_rig.json', 'projeto.json'):
        shutil.copy(f'{d}/anima/{nome}', f'{cd}/{nome}')
    shutil.copy(f'{d}/fundo.png', f'{cd}/fundo.png'); shutil.copy(f'{d}/fio.png', f'{cd}/oleo.png')
    print('recipiente sem liquido:', f'{d}/personagem_sem_liquido.png', '| px de liquido removidos:', int(keep.sum()))


def cmd_calibrar(a, cfg):
    d = a.dir; rc = cfg['recipiente']
    pan = R3.Panela(Rr=rc['Rr'], Rf=rc['Rf'], Dp=rc['Dp'], E=rc['camera_elevacao'])
    ex, ey, ea, eb = rc['elipse_interna']
    b0 = math.asin(eb / ea) - pan.E
    mk = np.load(f'{d}/mask_liquido_recipiente.npy')
    H, W = mk.shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lim = rc['y_lim_poca'] + 3
    y0, y1, x0, x1 = int(ey - eb - 10), int(ey + eb + 20), int(ex - ea - 10), int(ex + ea + 10)
    ref = (mk & (yy < lim))[y0:y1, x0:x1]
    X, Y, YY = xx[y0:y1, x0:x1] - ex, yy[y0:y1, x0:x1] - ey, yy[y0:y1, x0:x1]
    best = (0, 0)
    for h in np.linspace(pan.y_bico(b0) - 25, pan.y_bico(b0) + 10, 141):
        m, _, _, _ = pan.superficie(h, b0, X, Y); m = m & (YY < lim)
        iou = (m & ref).sum() / max((m | ref).sum(), 1)
        if iou > best[0]:
            best = (iou, h)
    h0 = best[1]; V0 = pan.volume(h0, b0)
    json.dump({'beta0': b0, 'h0': h0, 'V0_px3': V0, 'iou': best[0]}, open(f'{d}/calib.json', 'w'), indent=1)
    print(f'beta0 {math.degrees(b0):.1f} graus | h0 {h0:.1f} px (carga {h0 - pan.y_bico(b0):.1f}) | '
          f'volume na imagem {V0 * R3.ML_PX3:.0f} mL | IoU {best[0]:.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['camadas', 'rig', 'recipiente', 'calibrar'])
    ap.add_argument('--cena'); ap.add_argument('--cfg', required=True); ap.add_argument('--dir', required=True)
    a = ap.parse_args(); cfg = json.load(open(a.cfg))
    {'camadas': cmd_camadas, 'rig': cmd_rig, 'recipiente': cmd_recipiente, 'calibrar': cmd_calibrar}[a.cmd](a, cfg)


if __name__ == '__main__':
    main()
