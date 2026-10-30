#!/usr/bin/env python3
"""Render da cena de despejo: personagem (anima.py) + liquido no recipiente (modelo 3D) + fio + gotas + poca com
transbordo para a cuba + ambiente (vento na sombra de folhagem e na planta) + reenquadramento acima da legenda.
Le D/cena (p1.png sem liquido, p1_rig.json, fundo.png, oleo.png), D/cena1080.png, D/mask_liquido_recipiente.npy,
D/calib.json, D/simulacao.npz.
Uso: python3 -I cena_despejo.py --cfg c.json --dir D --saida v.mp4 [--quadros 0.5,1.2] [--sem-reenquadrar]"""
import argparse, json, math, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.expanduser('~/anima'))
import anima
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ambiente as AMB
import liquido_queda as LQ
import recipiente3d as R3
import liquido_recipiente as LR

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True); ap.add_argument('--dir', required=True); ap.add_argument('--saida', required=True)
ap.add_argument('--quadros'); ap.add_argument('--sem-reenquadrar', action='store_true')
a = ap.parse_args()
cfg = json.load(open(a.cfg)); D = a.dir; d = f'{D}/cena'
FPS = cfg['fps']; DUR = cfg['duracao']; N = int(round(DUR * FPS))
rc, po, pe = cfg['recipiente'], cfg['pouso'], cfg['personagem']
rig = json.load(open(f'{d}/p1_rig.json'))
rgba = np.asarray(Image.open(f'{d}/p1.png').convert('RGBA'))
oleo = np.asarray(Image.open(f'{d}/oleo.png').convert('RGBA')).astype(np.float32)
fundo = np.asarray(Image.open(f'{d}/fundo.png').convert('RGB')).astype(np.float32)
cena = np.asarray(Image.open(f'{D}/cena1080.png').convert('RGB'))
H, W = fundo.shape[:2]
sim = np.load(f'{D}/simulacao.npz'); ST = sim['T']
def S(t, k):
    return float(np.interp(t, ST, sim[k]))

pan = R3.Panela(Rr=rc['Rr'], Rf=rc['Rf'], Dp=rc['Dp'], E=rc['camera_elevacao'])
b0 = json.load(open(f'{D}/calib.json'))['beta0']
ex, ey, ea, eb = rc['elipse_interna']; CENTRO_BORDA = np.array([ex, ey])
liq = LR.LiquidoRecipiente(pan, cena, np.load(f'{D}/mask_liquido_recipiente.npy'), centro=(ex, ey), y_lim=rc['y_lim_poca'])
OSSO = rc['osso']; LADO = OSSO.split('_')[-1]

am = cfg['ambiente']; vento = AMB.Vento()
sombra = AMB.SombraFolhas(fundo, tuple(am['sombra']['caixa']), [tuple(f) for f in am['sombra']['faixas']],
                          tuple(am['sombra']['pivo']), tuple(am['sombra']['rampas']), vento) if 'sombra' in am else None
planta = AMB.Planta(fundo, tuple(am['planta']['caixa']), am['planta']['base_y'], vento) if 'planta' in am else None

def s5(x):
    x = min(max(x, 0.0), 1.0); return x * x * x * (x * (6 * x - 15) + 10)

def ruido(t, sem, f=0.35):
    r = np.random.default_rng(sem)
    fr = f * np.array([1.0, 1.618, 2.414, 3.303]); ph = r.uniform(0, 2 * np.pi, 4); amp = np.array([1, .6, .35, .2])
    return float((amp * np.sin(2 * np.pi * fr * t + ph)).sum() / amp.sum())

def respira(t, per=4.3):
    x = (t / per) % 1.0
    return s5(x / 0.4) if x < 0.4 else 1 - s5((x - 0.4) / 0.6)

bmin, bmax = math.degrees(sim['beta'].min()), math.degrees(sim['beta'].max())
def estado(t):
    delta, esc, raiz = {}, {}, {'ang': 0, 'dx': 0, 'dy': 0, 'sx': 1, 'sy': 1}
    b = 0.007 * (respira(t) - 0.5)
    esc['peito'] = (1 + b, 1 + 0.8 * b)
    beta = S(t, 'beta'); bdeg = math.degrees(beta)
    p = (bdeg - bmin) / max(bmax - bmin, 1e-3)
    k = math.sin(beta + pan.E) / math.sin(b0 + pan.E)                       # recipiente gira em profundidade
    esc[OSSO] = (1.0, k)
    delta[OSSO] = -0.05 * (bdeg - math.degrees(b0)) + 0.15 * ruido(t, 11, 0.9)
    delta[f'antebraco_{LADO}'] = -0.9 * p + 0.12 * ruido(t, 12, 0.5)
    delta[f'braco_{LADO}'] = -1.2 * p + 0.10 * ruido(t, 13, 0.3)
    outro = 'd' if LADO == 'e' else 'e'
    delta[f'braco_{outro}'] = 0.25 * ruido(t, 14, 0.25)
    delta[f'antebraco_{outro}'] = 0.20 * ruido(t, 15, 0.3)
    olhar = sum(g * s5((t - t_ini) / dur) for t_ini, dur, g in pe['olhar'])
    delta['cabeca'] = olhar + 0.35 * ruido(t, 16, 0.22)
    raiz['ang'] = 0.25 * p + 0.10 * ruido(t, 17, 0.12)
    olho = 1.0
    for tp in pe['piscadas']:
        x = t - tp
        if 0 <= x < 0.22:
            olho = min(olho, abs(x - 0.09) / 0.09 if x < 0.18 else 1)
    return delta, esc, raiz, olho, beta

cams, ossos = anima.montar_personagem(rgba, rig, 8)
rend = anima.Render(W, H)
for c in cams:
    rend.preparar(c)
    c.rosto = any(c.rgba[int((bb[1] + bb[3]) / 2), int((bb[0] + bb[2]) / 2), 3] > 128 for bb in rig.get('olhos', []))

def piscar_perfil(img, cx, s_):
    """Piscada que funciona de perfil: comprime o olho para a linha do meio e desenha a linha dos cilios fina,
    curva para baixo e afinando nas pontas."""
    x0, y0, x1, y1 = [int(round(v)) for v in cx]
    if s_ >= 0.98:
        return
    h = y1 - y0; cy = (y0 + y1) / 2; s_ = max(s_, 0.04)
    ys_ = np.arange(y0, y1, dtype=np.float32)
    src = np.where(np.abs(ys_ - cy) <= s_ * h / 2, cy + (ys_ - cy) / s_, np.where(ys_ < cy, y0 - 2, y1 + 1))
    my = np.repeat(src[:, None], x1 - x0, 1).astype(np.float32)
    mx = np.repeat(np.arange(x0, x1, dtype=np.float32)[None], len(ys_), 0)
    reg = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    if s_ < 0.45:
        orig = img[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.float32)
        escuro = orig[np.argsort(orig.sum(1))[:max(1, len(orig) // 15)]].mean(0)
        ww = x1 - x0; xx_ = np.arange(ww, dtype=np.float32); u = (xx_ - ww * 0.5) / (ww * 0.40)
        yc = (cy - y0) + 0.10 * h * (1 - np.clip(u, -1, 1) ** 2)
        esp = 1.6 * np.sqrt(np.clip(1 - u ** 2, 0, 1))
        yy_ = np.arange(h, dtype=np.float32)[:, None]
        cob = np.clip(esp[None] - np.abs(yy_ - yc[None]) + 0.5, 0, 1) * (1 - min(s_ / 0.45, 1)) * 0.85
        reg[..., :3] = reg[..., :3] * (1 - cob[..., None]) + escuro * cob[..., None]
    img[y0:y1, x0:x1] = reg.astype(np.uint8)

ys, xs = np.nonzero(oleo[..., 3] > 10)
fy0, fy1, fx0, fx1 = ys.min() - 2, ys.max() + 3, xs.min() - 14, xs.max() + 15
LABIO = np.array(cfg['fio']['labio'], float)
spr = oleo[fy0:fy1, fx0:fx1].copy(); spr[..., :3] *= spr[..., 3:4] / 255
G = 8900.0; V0 = 230.0; POUSO = po['y']
fio = LQ.Fio(spr, fy0, fx0, LABIO, cfg['fio']['y_corte'], POUSO, sim, q_ref=float(sim['q_img']), g=G, v0=V0)
Qs = sim['Q']; t_off = float(ST[len(Qs) - 1 - np.argmax(Qs[::-1] > 0.3)])
gotas = LQ.Gotas(fio.gota, POUSO, [t_off + dt for dt in cfg['gotas_apos_fim']], G)
poca = LQ.PocaSim(fundo, tuple(po['centro_poca']), po['achat'], sim, tuple(po['caixa_poca']), cfg.get('cuba'))

lab_cache = {}
def lab_em(tt):
    tt = min(max(tt, 0.0), DUR); k = round(tt * 1000)
    if k not in lab_cache:
        dd, ee, rr, _, _ = estado(tt); M = anima.fk(rig, ossos, dd, ee, rr)
        lab_cache[k] = (M[OSSO] @ np.r_[LABIO, 1])[:2]
    return lab_cache[k]

def hist(tt):
    p = lab_em(tt); v = (lab_em(tt) - lab_em(tt - 1 / FPS)) * FPS
    return np.array([p, v])

def pouso_gota(t0):
    tf = t0 + 0.30; dist = POUSO - (lab_em(tf)[1] + 14)
    return tf + (-40 + math.sqrt(40 * 40 + 2 * G * dist)) / G
ondas_gotas = [(float(lab_em(t0 + 0.3)[0]), POUSO, pouso_gota(t0)) for t0 in gotas.ini if pouso_gota(t0) < DUR]

quadros = sorted({min(int(round(float(x) * FPS)), N - 1) for x in a.quadros.split(',')}) if a.quadros else None
ff = None
if not quadros:
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709',
                           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', a.saida], stdin=subprocess.PIPE)
base = os.path.splitext(a.saida)[0]; folha = []; pior = (0, 0)
marcas = set(np.linspace(0, N - 1, 12).astype(int))
zc = cfg['camera']
for i in (quadros or range(N)):
    t = i / FPS
    delta, es, raiz, olho, beta = estado(t)
    M = anima.fk(rig, ossos, delta, es, raiz)
    itens = []
    for c in cams:
        P = anima.deformar(c, M)
        ar = anima.area(P, c.T)[c.cheio] / (c.area0[c.cheio] + 1e-6)
        ruim = 100 * (np.sum(c.area0[c.cheio][ar < 0]) + np.sum(c.area0[c.cheio][ar > 2])) / np.sum(c.area0[c.cheio])
        if ruim > pior[0]:
            pior = (ruim, t)
        tx = None
        if olho < 0.98 and c.rosto:
            tx = c.rgba.copy()
            for bb in rig['olhos']:
                piscar_perfil(tx, bb, olho)
            c._sujo = True
        elif getattr(c, '_sujo', False):
            tx = c.rgba; c._sujo = False
        itens.append((c, P, tx))
    fg = rend.quadro(itens).astype(np.float32)
    Cp = (M[OSSO] @ np.r_[CENTRO_BORDA, 1])[:2]
    Pr = (M[OSSO] @ np.r_[CENTRO_BORDA + [100, 0], 1])[:2]
    gama = math.atan2(Pr[1] - Cp[1], Pr[0] - Cp[0])
    Lp = liq.camada(beta, S(t, 'h'), S(t, 'Q'), t, Cp, gama, (0.0, pan.Rr * math.sin(beta + pan.E)), W, H,
                    q_ref=float(sim['q_img']))
    if Lp is not None:
        ap_ = Lp[..., 3:4] / 255
        fg = np.dstack([Lp[..., :3] + fg[..., :3] * (1 - ap_), Lp[..., 3:4] + fg[..., 3:4] * (1 - ap_)])
    camada = np.zeros((H, W, 4), np.float32)
    imp = fio.desenhar(camada, t, hist)
    gotas.desenhar(camada, t, hist)
    frente = camada + fg * (1 - camada[..., 3:4] / 255)
    fundo_t = fundo
    if sombra:
        fundo_t = sombra.aplicar(fundo_t, t)
    if planta:
        fundo_t = planta.aplicar(fundo_t, t)
    fundo_t = poca.aplicar(fundo_t, t, (imp[0], imp[1]) if imp else None, imp[2] if imp else 0.0, camada, ondas_gotas)
    al = frente[..., 3:4] / 255
    out = np.clip(frente[..., :3] + fundo_t * (1 - al), 0, 255).astype(np.uint8)
    if not a.sem_reenquadrar:
        z = zc['zoom'][0] + (zc['zoom'][1] - zc['zoom'][0]) * s5(t / DUR); Fx, Fy = zc['ancora']
        out = cv2.warpAffine(out, np.float32([[z, 0, Fx * (1 - z)], [0, z, Fy * (1 - z)]]), (W, H),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if quadros:
        Image.fromarray(out).save(f'{base}_t{t:.2f}.png'); print('quadro', f'{base}_t{t:.2f}.png'); continue
    ff.stdin.write(out.tobytes())
    if i in marcas:
        folha.append((t, cv2.resize(out, (270, 480), interpolation=cv2.INTER_AREA)))
if ff:
    ff.stdin.close(); ff.wait()
    fl = Image.new('RGB', (6 * 278 + 8, 2 * 512 + 8), (30, 30, 30)); dr = ImageDraw.Draw(fl)
    for k, (t, im) in enumerate(folha):
        x, y = 8 + (k % 6) * 278, 8 + (k // 6) * 512
        fl.paste(Image.fromarray(im), (x, y + 24)); dr.text((x + 4, y + 2), f'{t:.2f}s', fill=(255, 210, 80), font=anima.fonte(18))
    fl.save(f'{base}_folha.png')
    json.dump({'gotas_pouso': [g[2] for g in ondas_gotas], 't_inicio': float(fio.t_on), 't_fim_fio': t_off,
               't_impacto': float(sim['t_imp'])}, open(f'{base}_eventos.json', 'w'), indent=1)
    print(f'ok {a.saida}; pior quadro {pior[1]:.2f} s com {pior[0]:.2f}% da area dobrada ou esticada 2x')
