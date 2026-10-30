#!/usr/bin/env python3
"""anima.py: marionete 2D por esqueleto. Faz personagens de imagens estaticas se moverem.
  preparar --dir T --cena img.png [--caixa x0,y0,x1,y1 ...] [--tipo humano|cadeia]
  preparar --dir T --fundo fundo.png --personagem p1.png [--personagem p2.png ...]
  checar   --dir T                      (refaz mapas depois de editar pN_rig.json)
  animar   --dir T --roteiro r.json --saida v.mp4 [--alfa camada.webm|.mov] [--previa]
Coordenadas em pixels do quadro de trabalho (padrao 1080x1920). Sufixo _e/_d = lado esquerdo/direito DA IMAGEM."""
import argparse, json, math, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

MOD = os.path.expanduser('~/modelos')
F = 0.25                                   # escala do mapa de distancias geodesicas

def fonte(t):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', t)
    except OSError:
        return ImageFont.load_default()

def ajustar(im, W, H):
    """Escala para cobrir WxH e corta no centro (mesma transformacao em todas as camadas)."""
    w, h = im.size; s = max(W / w, H / h)
    im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    x0, y0 = (im.width - W) // 2, (im.height - H) // 2
    return im.crop((x0, y0, x0 + W, y0 + H))

_LAMA = None

def _lama512(img, msk):
    global _LAMA
    import onnxruntime as ort
    if _LAMA is None:
        _LAMA = ort.InferenceSession(f'{MOD}/lama_fp32.onnx', providers=['CPUExecutionProvider'])
    m = (msk > 0).astype(np.float32)
    img = img.astype(np.float32) * (1 - m[..., None])          # o modelo nao apaga sozinho a area mascarada
    out = _LAMA.run(None, {'image': img.transpose(2, 0, 1)[None] / 255,
                           'mask': m[None, None]})[0][0].transpose(1, 2, 0)
    return np.clip(out * 255 if out.max() <= 1.5 else out, 0, 255).astype(np.uint8)

def lama(rgb, mask):
    """Reconstroi a area mascarada com LaMa (ONNX, entrada fixa 512x512). Reduz a imagem so o necessario
    para o buraco ter ~280 px no lado menor (sobra contexto em volta) e cobre o buraco com blocos de 512
    sobrepostos, misturados por peso (sem emenda)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return rgb
    if not os.path.exists(f'{MOD}/lama_fp32.onnx'):
        print('AVISO: modelo LaMa ausente; fundo reconstruido com OpenCV (qualidade menor). Prefira a camada de fundo.')
        return cv2.inpaint(rgb, mask.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    Hh, Ww = mask.shape
    s = min(1.0, 280 / max(min(np.ptp(xs), np.ptp(ys)) + 1, 1), 512 / min(Hh, Ww))
    sw, sh = max(int(Ww * s), 512), max(int(Hh * s), 512)
    im_s = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    mk_s = cv2.dilate((cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8),
                      np.ones((5, 5), np.uint8))
    acc = np.zeros((sh, sw, 3), np.float32); pes = np.zeros((sh, sw), np.float32)
    r = np.minimum(np.arange(512), np.arange(512)[::-1]).astype(np.float32) + 8
    peso = np.outer(r, r)
    ys_s, xs_s = np.nonzero(mk_s)
    by0, by1, bx0, bx1 = ys_s.min(), ys_s.max(), xs_s.min(), xs_s.max()
    cy0 = int(np.clip((by0 + by1) / 2 - 256, 0, sh - 512)) if by1 - by0 < 384 else None
    cx0 = int(np.clip((bx0 + bx1) / 2 - 256, 0, sw - 512)) if bx1 - bx0 < 384 else None
    lin = [cy0] if cy0 is not None else sorted({int(np.clip(y, 0, sh - 512)) for y in range(by0 - 128, by1 - 127, 320)} | {int(np.clip(by1 - 384, 0, sh - 512))})
    col = [cx0] if cx0 is not None else sorted({int(np.clip(x, 0, sw - 512)) for x in range(bx0 - 128, bx1 - 127, 320)} | {int(np.clip(bx1 - 384, 0, sw - 512))})
    for ty in lin:
        for tx in col:
            mt = mk_s[ty:ty + 512, tx:tx + 512]
            if not mt.any():
                continue
            o = _lama512(im_s[ty:ty + 512, tx:tx + 512], mt).astype(np.float32)
            acc[ty:ty + 512, tx:tx + 512] += o * peso[..., None]; pes[ty:ty + 512, tx:tx + 512] += peso
    out_s = np.where(pes[..., None] > 0, acc / np.maximum(pes, 1e-6)[..., None], im_s)
    out = cv2.resize(out_s.astype(np.uint8), (Ww, Hh), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 2)[..., None]
    return (out * f + rgb * (1 - f)).astype(np.uint8)

def recortar(rgb):
    try:
        from rembg import remove, new_session
        return np.asarray(remove(Image.fromarray(rgb), session=new_session('isnet-general-use'), only_mask=True))
    except Exception as e:
        raise SystemExit(f'Recorte automatico indisponivel ({e}). Peca o personagem em PNG transparente e use --fundo/--personagem.')

def pose(rgba):
    """Pontos de pose (MediaPipe) do personagem sobre fundo cinza, na caixa do personagem."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    a = rgba[..., 3:4] / 255.
    img = (rgba[..., :3] * a + 128 * (1 - a)).astype(np.uint8)
    ys, xs = np.nonzero(rgba[..., 3] > 127)
    x0, y0 = max(xs.min() - 40, 0), max(ys.min() - 40, 0)
    x1, y1 = min(xs.max() + 40, img.shape[1]), min(ys.max() + 40, img.shape[0])
    opt = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=f'{MOD}/pose_landmarker_full.task'))
    with vision.PoseLandmarker.create_from_options(opt) as det:
        for (a0, b0, a1, b1) in ((x0, y0, x1, y1), (0, 0, img.shape[1], img.shape[0])):   # caixa e, se falhar, quadro todo
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[b0:b1, a0:a1])))
            if r.pose_landmarks:
                return [(a0 + p.x * (a1 - a0), b0 + p.y * (b1 - b0), p.visibility) for p in r.pose_landmarks[0]]
    return None

# ---------------------------------------------------------------- esqueletos
def ossos_de(tipo):
    if tipo == 'cadeia':
        return [('seg1', None, 'c0', 'c1', 'corpo'), ('seg2', 'seg1', 'c1', 'c2', 'corpo'), ('seg3', 'seg2', 'c2', 'c3', 'corpo')]
    o = [('tronco', None, 'pelvis', 'peito', 'corpo'), ('peito', 'tronco', 'peito', 'pescoco', 'corpo'),
         ('cabeca', 'peito', 'pescoco', 'cabeca', 'corpo')]
    for s in 'ed':
        o += [(f'clav_{s}', 'peito', 'pescoco', f'ombro_{s}', 'corpo'),
              (f'braco_{s}', f'clav_{s}', f'ombro_{s}', f'cotovelo_{s}', f'braco_{s}'),
              (f'antebraco_{s}', f'braco_{s}', f'cotovelo_{s}', f'punho_{s}', f'braco_{s}'),
              (f'mao_{s}', f'antebraco_{s}', f'punho_{s}', f'mao_{s}', f'braco_{s}'),
              (f'quad_{s}', None, 'pelvis', f'quadril_{s}', 'corpo'),
              (f'coxa_{s}', f'quad_{s}', f'quadril_{s}', f'joelho_{s}', f'perna_{s}'),
              (f'canela_{s}', f'coxa_{s}', f'joelho_{s}', f'tornozelo_{s}', f'perna_{s}'),
              (f'pe_{s}', f'canela_{s}', f'tornozelo_{s}', f'pe_{s}', f'perna_{s}')]
    return o

def rig_auto(rgba, tipo=None):
    a = rgba[..., 3] > 127
    ys, xs = np.nonzero(a)
    alt = int(np.ptp(ys))
    try:
        lm = None if tipo == 'cadeia' else pose(rgba)
    except Exception as e:
        print(f'AVISO: deteccao de pose indisponivel ({e}); juntas em cadeia, ajuste a mao se for humano.'); lm = None
    ok = lm is not None and min(lm[i][2] for i in (11, 12, 23, 24)) > 0.5
    if tipo == 'humano' and not ok:
        print('AVISO: pose nao detectada; rig humano em cadeia vertical. Posicione as juntas a mao.')
    if not ok:
        def cx(y):
            c = np.nonzero(a[int(y)])[0]
            return float(c.mean()) if len(c) else float(xs.mean())
        y0, y1 = ys.max(), ys.min()
        J = {f'c{i}': [cx(y0 + (y1 - y0) * f - (2 if i == 0 else 0)), float(y0 + (y1 - y0) * f)]
             for i, f in enumerate((0, 1 / 3, 2 / 3, 0.98))}
        return dict(tipo='cadeia', j=J, chao=J['c0'], altura=alt, olhos=[], boca=None, profundidade={}, separar=[])
    P = lambda i: np.array(lm[i][:2])
    A, B = [11, 13, 15, 19], [12, 14, 16, 20]
    if P(11)[0] > P(12)[0]:
        A, B = B, A
    C, D = [23, 25, 27, 31], [24, 26, 28, 32]
    if P(23)[0] > P(24)[0]:
        C, D = D, C
    J = {}
    for s, br, pe in (('e', A, C), ('d', B, D)):
        for n, i in zip(('ombro', 'cotovelo', 'punho', 'mao'), br):
            J[f'{n}_{s}'] = P(i)
        for n, i in zip(('quadril', 'joelho', 'tornozelo', 'pe'), pe):
            J[f'{n}_{s}'] = P(i)
    J['pelvis'] = (J['quadril_e'] + J['quadril_d']) / 2
    J['pescoco'] = (J['ombro_e'] + J['ombro_d']) / 2
    J['peito'] = J['pelvis'] + 0.62 * (J['pescoco'] - J['pelvis'])
    nariz, larg = P(0), abs(J['ombro_d'][0] - J['ombro_e'][0])
    col = np.nonzero(a[:, int(max(nariz[0] - .08 * larg, 0)):int(nariz[0] + .08 * larg) + 1].any(1))[0]
    J['cabeca'] = np.array([nariz[0], float(col.min()) + 4]) if len(col) else nariz + (nariz - J['pescoco'])
    centralizar(J, a)
    de = abs(P(2)[0] - P(5)[0]); mb = (P(9) + P(10)) / 2; mw = max(abs(P(9)[0] - P(10)[0]), de * .5)
    olhos = [[e[0] - .34 * de, e[1] - .32 * de, e[0] + .34 * de, e[1] + .32 * de] for e in (P(2), P(5))]
    boca = [mb[0] - .65 * mw, mb[1] - .3 * mw, mb[0] + .65 * mw, mb[1] + .3 * mw]
    olhos, boca = refinar_rosto(rgba, (P(2) + P(5)) / 2, larg, olhos, boca)
    chao = [(J['tornozelo_e'][0] + J['tornozelo_d'][0]) / 2, float(ys.max())]
    r = lambda v: [round(float(x), 1) for x in v]
    return dict(tipo='humano', j={k: r(v) for k, v in J.items()}, chao=r(chao), altura=alt,
                olhos=[r(o) for o in sorted(olhos)], boca=r(boca),
                profundidade={'braco_e': 1, 'braco_d': 1, 'perna_e': -1, 'perna_d': -1}, separar=[],
                visibilidade={str(i): round(lm[i][2], 2) for i in (13, 14, 15, 16, 25, 26, 27, 28)})

def _eixo(dt, q, u, W, H):
    """Ponto do eixo do membro mais proximo de q, na perpendicular a u (maximo local da distancia a borda)."""
    v = np.array([-u[1], u[0]]); off = np.arange(-60, 61)
    pts = np.clip(np.round(q + off[:, None] * v).astype(int), 0, [W - 1, H - 1])
    prof = dt[pts[:, 1], pts[:, 0]]
    k = 60
    if prof[k] == 0:
        dentro = np.nonzero(prof > 0)[0]
        if not len(dentro):
            return q
        k = dentro[np.argmin(np.abs(dentro - 60))]
    while 0 < k < len(prof) - 1 and max(prof[k - 1], prof[k + 1]) > prof[k]:
        k = k - 1 if prof[k - 1] > prof[k + 1] else k + 1
    return q + off[k] * v

def centralizar(J, a):
    """Leva cotovelos, punhos, joelhos e tornozelos ao eixo do membro (ombro e quadril ficam como vieram:
    conferir no mapa, porque a pose costuma por o ombro no meio do volume do ombro, fora do eixo do braco)."""
    from scipy import ndimage
    dt = ndimage.distance_transform_edt(a)
    H, W = a.shape
    for s in 'ed':
        for de, j in (('ombro', 'cotovelo'), ('cotovelo', 'punho'), ('quadril', 'joelho'), ('joelho', 'tornozelo')):
            p, q = np.array(J[f'{de}_{s}'], float), np.array(J[f'{j}_{s}'], float)
            u = (q - p) / (np.linalg.norm(q - p) + 1e-9)
            J[f'{j}_{s}'] = _eixo(dt, q, u, W, H)


def refinar_rosto(rgba, centro_olhos, larg, olhos, boca):
    """Olhos = 2 manchas escuras perto dos olhos da pose; boca = mancha de cor diferente da pele abaixo deles."""
    img = rgba[..., :3].astype(np.float32); al = rgba[..., 3] > 127
    cx, cy = centro_olhos; r = 0.42 * larg
    x0, x1, y0, y1 = int(max(cx - r, 0)), int(cx + r), int(max(cy - .6 * r, 0)), int(cy + .6 * r)
    reg, ra = img[y0:y1, x0:x1], al[y0:y1, x0:x1]
    lum = reg @ [0.299, 0.587, 0.114]
    if ra.sum() < 50:
        return olhos, boca
    esc = (lum < min(np.percentile(lum[ra], 8), 90)) & ra
    n, lab, st, cen = cv2.connectedComponentsWithStats(esc.astype(np.uint8))
    cand = [i for i in range(1, n) if st[i, 4] > 20 and st[i, 2] < r and st[i, 3] < r]
    cand = sorted(cand, key=lambda i: -st[i, 4])[:6]
    par = None
    for i in cand:                                     # par de manchas parecidas, lado a lado
        for k in cand:
            if k <= i:
                continue
            dy, dx = abs(cen[i][1] - cen[k][1]), abs(cen[i][0] - cen[k][0])
            if dy < 0.25 * dx and 0.5 < st[i, 4] / st[k, 4] < 2 and dx > 0.08 * larg:
                par = (i, k); break
        if par:
            break
    if par:
        olhos = []
        for i in sorted(par, key=lambda i: cen[i][0]):
            bx, by, bw, bh = st[i, :4]
            olhos.append([round(float(x0 + bx - .3 * bw), 1), round(float(y0 + by - .35 * bh), 1),
                          round(float(x0 + bx + 1.3 * bw), 1), round(float(y0 + by + 1.35 * bh), 1)])
        ey = np.mean([(o[1] + o[3]) / 2 for o in olhos]); ex = np.mean([(o[0] + o[2]) / 2 for o in olhos])
        dx = abs((olhos[1][0] + olhos[1][2]) / 2 - (olhos[0][0] + olhos[0][2]) / 2)
        bx0, bx1, by0, by1 = int(ex - .7 * dx), int(ex + .7 * dx), int(ey + .45 * dx), int(ey + 1.6 * dx)
        rb, rba = img[by0:by1, bx0:bx1], al[by0:by1, bx0:bx1]
        if rba.sum() > 50:
            pele = np.median(rb[rba], 0)
            dif = (np.linalg.norm(rb - pele, axis=-1) > 60) & rba
            n, lab, st, cen = cv2.connectedComponentsWithStats(dif.astype(np.uint8))
            c = [i for i in range(1, n) if st[i, 4] > 15 and st[i, 2] < 1.4 * dx]
            if c:
                i = min(c, key=lambda i: abs(cen[i][0] - (ex - bx0)) + 0.5 * cen[i][1])
                bx, by, bw, bh = st[i, :4]
                boca = [round(float(bx0 + bx - .1 * bw), 1), round(float(by0 + by - .25 * bh), 1),
                        round(float(bx0 + bx + 1.1 * bw), 1), round(float(by0 + by + 1.25 * bh), 1)]
    return olhos, boca

# ---------------------------------------------------------------- pesos e malha
def dist_geo(alpha, rig, ossos):
    """Distancia geodesica (dentro da silhueta) de cada pixel a cada osso, normalizada pela espessura do osso."""
    from skimage.graph import MCP_Geometric
    from scipy import ndimage
    ms = cv2.resize((alpha > 20).astype(np.uint8), None, fx=F, fy=F, interpolation=cv2.INTER_AREA) > 0
    dt = ndimage.distance_transform_edt(ms)
    _, (iy, ix) = ndimage.distance_transform_edt(~ms, return_indices=True)
    custo = np.where(ms, 1.0, np.inf)
    h, w = ms.shape; D = []
    for nome, pai, de, para, g in ossos:
        a, b = np.array(rig['j'][de]) * F, np.array(rig['j'][para]) * F
        n = int(np.hypot(*(b - a))) + 2
        pts = np.round(a + (b - a) * np.linspace(0, 1, n)[:, None]).astype(int)
        px, py = np.clip(pts[:, 0], 0, w - 1), np.clip(pts[:, 1], 0, h - 1)
        ini = sorted({(int(iy[y, x]), int(ix[y, x])) for x, y in zip(px, py)})
        d, _ = MCP_Geometric(custo).find_costs(ini)
        esp = max(float(np.median([dt[y, x] for y, x in ini])), 2.0)
        D.append(d / esp)
    D = np.stack(D, -1)[iy, ix]
    return D

def pesos_d(d):
    w = 1.0 / (np.where(np.isfinite(d), d, 1e3) + 0.25) ** 4
    return w / w.sum(-1, keepdims=True)

def pesos_em(D, pts):
    """Pesos nos pontos, com distancia interpolada (bilinear) no mapa de baixa resolucao."""
    Df = np.where(np.isfinite(D), D, 1e3).astype(np.float32)
    x = np.clip(pts[:, 0] * F - 0.5, 0, D.shape[1] - 1.001); y = np.clip(pts[:, 1] * F - 0.5, 0, D.shape[0] - 1.001)
    x0, y0 = x.astype(int), y.astype(int); fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    d = (Df[y0, x0] * (1 - fx) * (1 - fy) + Df[y0, x0 + 1] * fx * (1 - fy)
         + Df[y0 + 1, x0] * (1 - fx) * fy + Df[y0 + 1, x0 + 1] * fx * fy)
    return pesos_d(d)

def suavizar(W, T, n_v, it=30):
    """Suaviza os pesos ao longo da malha (media com vizinhos), tirando o ruido que dobra triangulos."""
    from scipy import sparse
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_v, n_v))
    A.data[:] = 1
    grau = np.asarray(A.sum(1)).ravel(); grau[grau == 0] = 1
    for _ in range(it):
        W = 0.5 * W + 0.5 * (A @ W) / grau[:, None]
    return W

def malha(alpha, passo):
    H, W = alpha.shape
    m = cv2.dilate((alpha > 8).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    gh, gw = -(-H // passo), -(-W // passo)
    mp_ = np.zeros((gh * passo, gw * passo), np.uint8); mp_[:H, :W] = m
    cel = mp_.reshape(gh, passo, gw, passo).max((1, 3)) > 0
    cy, cx = np.nonzero(cel)
    idx = -np.ones((gh + 1, gw + 1), int); usados = np.zeros((gh + 1, gw + 1), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            usados[cy + dy, cx + dx] = True
    vy, vx = np.nonzero(usados)
    idx[vy, vx] = np.arange(len(vy))
    V = np.stack([np.minimum(vx * passo, W), np.minimum(vy * passo, H)], 1).astype(np.float32)
    a, b, c, d = idx[cy, cx], idx[cy, cx + 1], idx[cy + 1, cx], idx[cy + 1, cx + 1]
    T = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    centro = np.stack([(cx + 0.5) * passo, (cy + 0.5) * passo], 1)
    return V, T, np.concatenate([centro, centro])

# ---------------------------------------------------------------- camadas por personagem
class Camada:
    def __init__(self, rgba, rig, ossos, nomes, prof, passo):
        self.rgba = rgba
        self.nomes = nomes                                   # ossos que movem esta camada
        sub = [o for o in ossos if o[0] in nomes]
        from scipy import ndimage
        alpha = rgba[..., 3]; H, W = alpha.shape
        D = dist_geo(alpha, rig, sub)
        V, T, tc = malha(alpha, passo)
        W0 = pesos_em(D, V)
        grupos = [o[4] for o in sub]
        # vertices na fresta (transparentes) sao duplicados por parte do corpo: assim a fresta entre braco e
        # tronco nao vira triangulo esticado (franja) quando o braco se move
        gn = sorted(set(grupos)); gb = np.array([gn.index(g) for g in grupos])
        labg = gb[np.where(np.isfinite(D), D, 1e9).argmin(-1)]
        h, w = labg.shape
        ms = cv2.resize((alpha > 20).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
        cg = labg[np.clip((tc[:, 1] * F).astype(int), 0, h - 1), np.clip((tc[:, 0] * F).astype(int), 0, w - 1)]
        solto = alpha[np.clip(V[:, 1].astype(int), 0, H - 1), np.clip(V[:, 0].astype(int), 0, W - 1)] < 20
        perto = {}
        for g in range(len(gn)):
            mg = ms & (labg == g)
            if mg.any():
                perto[g] = ndimage.distance_transform_edt(~mg, return_indices=True)[1]
        Vn, Wn, mapa = [V], [W0], {}; prox = len(V); T = T.copy()
        for t in range(len(T)):
            g = int(cg[t])
            for k in range(3):
                v = T[t, k]
                if not solto[v] or len(gn) == 1:
                    continue
                if (v, g) not in mapa:
                    if g in perto:
                        y, x = min(int(V[v, 1] * F), h - 1), min(int(V[v, 0] * F), w - 1)
                        iy, ix = perto[g]; wv = pesos_d(D[iy[y, x], ix[y, x]])
                    else:
                        wv = W0[v]
                    mapa[(v, g)] = prox; prox += 1; Vn.append(V[v][None]); Wn.append(wv[None])
                T[t, k] = mapa[(v, g)]
        self.V = np.concatenate(Vn).astype(np.float32); self.W = suavizar(np.concatenate(Wn), T, prox)
        # cada vertice so mistura o osso dominante com vizinhos no esqueleto (pai, filhos, irmaos da mesma
        # junta); peso de osso distante (peito no braco) dobra triangulos quando o membro gira muito
        pai_d = {o[0]: o[1] for o in ossos}
        viz = np.zeros((len(sub), len(sub)), bool)
        for i, oi in enumerate(sub):
            for j, oj in enumerate(sub):
                viz[i, j] = (i == j or pai_d[oi[0]] == oj[0] or pai_d[oj[0]] == oi[0]
                             or (pai_d[oi[0]] == pai_d[oj[0]] and oi[2] == oj[2]))
        self.W = self.W * viz[self.W.argmax(1)]
        self.W = self.W / self.W.sum(1, keepdims=True)
        dom = self.W.argmax(1)
        dt = np.array([prof.get(grupos[k], 0) for k in dom])
        tri_d = dt[T].max(1)
        self.T = T[np.argsort(tri_d, kind='stable')]
        cen = self.V[self.T].mean(1).astype(int)
        H, W = rgba.shape[:2]
        self.cheio = rgba[np.clip(cen[:, 1], 0, H - 1), np.clip(cen[:, 0], 0, W - 1), 3] > 128
        self.area0 = area(self.V, self.T)
        self.prof = float(min(prof.get(g, 0) for g in grupos)) if all(g != 'corpo' for g in grupos) else 0.0
        # centro de rotacao continuo por vertice: media das juntas que ligam cada par de ossos, pesada por
        # w_i*w_j. Gira o vertice em torno da junta certa e evita o ombro afinar ao girar.
        pai = {o[0]: o[1] for o in ossos}; de = {o[0]: o[2] for o in ossos}
        def anc(n):
            r = [n]
            while pai.get(r[-1]):
                r.append(pai[r[-1]])
            return r
        B = len(sub); Cp = np.zeros((B, B, 2))
        for i in range(B):
            for j in range(B):
                ai, aj = anc(sub[i][0]), anc(sub[j][0])
                if sub[j][0] in ai and i != j:
                    k = ai[ai.index(sub[j][0]) - 1]; Cp[i, j] = rig['j'][de[k]]
                elif sub[i][0] in aj and i != j:
                    k = aj[aj.index(sub[i][0]) - 1]; Cp[i, j] = rig['j'][de[k]]
                else:
                    com = next((x for x in ai if x in aj), None)
                    ki = ai[ai.index(com) - 1] if com and ai.index(com) > 0 else ai[-1]
                    kj = aj[aj.index(com) - 1] if com and aj.index(com) > 0 else aj[-1]
                    Cp[i, j] = (np.array(rig['j'][de[ki]]) + np.array(rig['j'][de[kj]])) / 2
        Wp = np.einsum('ni,nj->nij', self.W, self.W) * (1 - np.eye(B))[None]
        den = Wp.sum((1, 2))
        num = np.einsum('nij,ijk->nk', Wp, Cp)
        self.C = np.where(den[:, None] > 1e-6, num / np.maximum(den, 1e-9)[:, None], self.V).astype(np.float64)

def area(V, T):
    p = V[T]
    return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))

def partes(rgba, rig, ossos):
    """Rotulo de grupo (corpo, braco_e...) por pixel, pela menor distancia geodesica normalizada."""
    D = dist_geo(rgba[..., 3], rig, ossos)
    lab = np.argmin(np.where(np.isfinite(D), D, 1e9), -1)
    grupos = sorted({o[4] for o in ossos})
    g_idx = np.array([grupos.index(o[4]) for o in ossos])[lab]
    H, W = rgba.shape[:2]
    return cv2.resize(g_idx.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST), grupos

def montar_personagem(rgba, rig, passo, log=print):
    ossos = ossos_de(rig['tipo'])
    prof = rig.get('profundidade', {})
    sep = [g for g in rig.get('separar', []) if any(o[4] == g for o in ossos)]
    if not sep:
        return [Camada(rgba, rig, ossos, [o[0] for o in ossos], prof, passo)], ossos
    from scipy import ndimage
    alpha = rgba[..., 3].astype(np.float32); H, W = alpha.shape
    D = dist_geo(rgba[..., 3], rig, ossos)
    up = lambda X: cv2.resize(np.where(np.isfinite(X), X, 1e3).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    corpo = rgba.copy(); camadas = []; removido = np.zeros((H, W), np.float32); fech_r = 0
    for g in sep:
        ig = [k for k, o in enumerate(ossos) if o[4] == g]; io = [k for k, o in enumerate(ossos) if o[4] != g]
        membro = (up(D[..., ig].min(-1)) < up(D[..., io].min(-1))) & (alpha > 8)       # corte em resolucao cheia
        n, lab, st, _ = cv2.connectedComponentsWithStats(membro.astype(np.uint8))
        pts = [np.array(rig['j'][ossos[k][3]]).astype(int) for k in ig]
        ok = {lab[min(max(p[1], 0), H - 1), min(max(p[0], 0), W - 1)] for p in pts} - {0}
        if ok:                                           # so o pedaco ligado aos ossos do membro, sem furos
            membro = np.isin(lab, list(ok))
        membro = ndimage.binary_fill_holes(membro) & (alpha > 8)
        rj = np.array(rig['j'][ossos[ig[0]][2]])
        esp = float(np.median(ndimage.distance_transform_edt(membro)[membro])) * 2 + 1
        yy, xx = np.ogrid[:H, :W]
        disco = ((xx - rj[0]) ** 2 + (yy - rj[1]) ** 2 < (0.75 * esp) ** 2) & (alpha > 8)
        kd = max(int(0.4 * esp), 3) * 2 + 1                       # na camada do membro, so o disco junto ao membro
        disco_m = disco & (cv2.dilate(membro.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))) > 0)
        suave = cv2.GaussianBlur(membro.astype(np.float32), (0, 0), 1.2)
        cam = rgba.copy(); cam[..., 3] = (alpha * np.maximum(suave, disco_m)).astype(np.uint8)
        nomes = [o[0] for o in ossos if o[4] == g]
        camadas.append(Camada(cam, rig, ossos, nomes, prof, passo))
        rem = cv2.GaussianBlur(cv2.dilate(membro.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.0)
        removido = np.maximum(removido, np.clip(rem * 1.5, 0, 1))     # tira o membro inteiro do tronco, com a borda (sem toco nem contorno)
        fech_r = max(fech_r, int(0.9 * esp))
        log(f'  camada separada: {g} (espessura ~{esp:.0f} px)')
    base = (alpha > 8) & (removido < 0.5)
    k = 2 * max(fech_r, 5) + 1
    fech = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fech = cv2.GaussianBlur(fech.astype(np.float32), (0, 0), 2) > 0.5          # contorno do tronco liso
    ke = 2 * max(int(0.15 * fech_r / 0.9), 3) + 1                # tronco se estende um pouco por baixo da emenda
    ext = cv2.dilate(base.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))) > 0
    preencher = (fech | ext) & ~base & (cv2.dilate((removido > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (alpha > 8)
    a_ = rgba[..., 3:4] / 255.
    plano = (rgba[..., :3] * a_ + 128 * (1 - a_)).astype(np.uint8)
    if preencher.any():
        plano = lama(plano, preencher)
        log(f'  tronco reconstruido atras do membro: {int(preencher.sum())} px')
    pf = cv2.GaussianBlur(preencher.astype(np.float32), (0, 0), 1.0)
    corpo[..., :3] = (plano * pf[..., None] + rgba[..., :3] * (1 - pf[..., None])).astype(np.uint8)
    corpo[..., 3] = np.clip(np.maximum(alpha * (1 - removido), 255 * pf), 0, 255).astype(np.uint8)
    nomes = [o[0] for o in ossos if o[4] not in sep]
    camadas.append(Camada(corpo, rig, ossos, nomes, prof, passo))
    camadas.sort(key=lambda c: c.prof)
    return camadas, ossos

# ---------------------------------------------------------------- cinematica
def ang(J, a, b):
    d = np.subtract(J[b], J[a]); return math.degrees(math.atan2(d[1], d[0]))

def wrap(a):
    return (a + 180) % 360 - 180

def fk(rig, ossos, delta, escala, raiz):
    """Matriz afim 3x3 (repouso -> pose) de cada osso."""
    J = rig['j']; piv = np.array(rig['chao'], float)
    a = math.radians(raiz.get('ang', 0)); c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]]) @ np.diag([raiz.get('sx', 1), raiz.get('sy', 1)])
    Mr = np.eye(3); Mr[:2, :2] = R; Mr[:2, 2] = piv - R @ piv + [raiz.get('dx', 0), raiz.get('dy', 0)]
    M = {}
    for nome, pai, de, para, g in ossos:
        j = np.array(J[de], float); u = np.subtract(J[para], J[de]); u = u / (np.linalg.norm(u) + 1e-9); v = np.array([-u[1], u[0]])
        sa, sp = escala.get(nome, (1, 1))
        t = math.radians(delta.get(nome, 0)); c, s = math.cos(t), math.sin(t)
        A = np.array([[c, -s], [s, c]]) @ (sa * np.outer(u, u) + sp * np.outer(v, v))
        L = np.eye(3); L[:2, :2] = A; L[:2, 2] = j - A @ j
        M[nome] = (M[pai] if pai else Mr) @ L
    return M

def deformar(cam, M):
    """Mistura com centro de rotacao: gira o vertice em torno da junta pela rotacao media normalizada
    (sem o encolhimento da mistura linear) e leva a junta para a posicao da pose."""
    A = np.stack([M[n] for n in cam.nomes])
    L, t = A[:, :2, :2], A[:, :2, 2]
    Ab = np.einsum('nb,bij->nij', cam.W, L)
    det = Ab[:, 0, 0] * Ab[:, 1, 1] - Ab[:, 0, 1] * Ab[:, 1, 0]
    alvo = cam.W @ (L[:, 0, 0] * L[:, 1, 1] - L[:, 0, 1] * L[:, 1, 0])
    f = np.sqrt(np.clip(alvo, 1e-4, None) / np.clip(det, 1e-4, None))
    Ab = Ab * np.clip(f, 1, 3)[:, None, None]
    Cp = np.einsum('nb,bij,nj->ni', cam.W, L, cam.C) + cam.W @ t
    P = np.einsum('nij,nj->ni', Ab, cam.V - cam.C) + Cp
    return P.astype(np.float32)

# ---------------------------------------------------------------- acoes
def env(t, a, b, r=0.35):
    if t <= a or t >= b:
        return 0.0
    r = min(r, (b - a) / 2)
    x = min((t - a) / r, (b - t) / r, 1.0)
    return x * x * (3 - 2 * x)

def ruido(t, f, sem):
    return sum(math.sin(2 * math.pi * f * k * t + sem * (k + 1.7)) / k for k in (1, 2.31, 3.7)) / 1.8

def dir_fora(lado, acima):
    sx = -1 if lado == 'e' else 1
    return math.degrees(math.atan2(-math.sin(math.radians(acima)), sx * math.cos(math.radians(acima))))

def braco_para(J, lado, ab, aa, am=None, e=1.0, esc=None):
    """Deltas locais para levar o braco a direcao ab (braco) e aa (antebraco). Na transicao o cotovelo
    dobra antes do ombro e o braco encurta (como se viesse para a camera), para a mao nao varrer longe."""
    s = lado
    rb, ra, rm = ang(J, f'ombro_{s}', f'cotovelo_{s}'), ang(J, f'cotovelo_{s}', f'punho_{s}'), ang(J, f'punho_{s}', f'mao_{s}')
    db = wrap(ab - rb); da = db + wrap(aa - ra - db); dm = da + wrap((aa if am is None else am) - rm - da)
    eb = min(max((e - 0.2) / 0.8, 0), 1); eb = eb * eb * (3 - 2 * eb)
    ea = min(e / 0.65, 1); ea = ea * ea * (3 - 2 * ea)
    if esc is not None and 0 < e < 1 and abs(da) > 60:
        enc = math.sin(math.pi * e)
        esc[f'braco_{s}'] = (1 - 0.18 * enc, 1); esc[f'antebraco_{s}'] = (1 - 0.42 * enc, 1)
    return {f'braco_{s}': db * eb, f'antebraco_{s}': (da - db) * ea, f'mao_{s}': (dm - da) * ea}

def somar(d, extra):
    for k, v in extra.items():
        d[k] = d.get(k, 0) + v

def boca_sintetica(t):
    v = 0.55 + 0.45 * math.sin(2 * math.pi * 4.3 * t) * math.sin(2 * math.pi * 1.1 * t + 0.7)
    return max(0.0, min(1.0, v * (0.6 + 0.4 * abs(math.sin(2 * math.pi * 0.37 * t)))))

def estado(p, rig, t, cache):
    """Calcula deltas, escalas, raiz e efeitos de rosto de um personagem no tempo t."""
    J, tipo, h = rig['j'], rig['tipo'], rig.get('altura', 800)
    delta, esc, raiz = {}, {}, {'ang': 0, 'dx': 0, 'dy': 0, 'sx': 1, 'sy': 1}
    olho, boca = 1.0, 0.0
    sem = sum(map(ord, p['id'])) % 97
    if p.get('base', 'parado') == 'parado':
        b = 0.0125 * math.sin(2 * math.pi * t / 3.6 + sem)
        if tipo == 'humano':
            esc['peito'] = (1 + b, 1 + 0.6 * b)
            raiz['ang'] += 0.45 * math.sin(2 * math.pi * t / 5.3 + sem)
            delta['cabeca'] = 1.2 * math.sin(2 * math.pi * t / 4.4 + 1 + sem)
            for s, f in (('e', 1), ('d', -1)):
                delta[f'braco_{s}'] = f * 1.2 * math.sin(2 * math.pi * t / 3.6 + sem + .4)
            if rig.get('olhos'):                          # piscada automatica a cada ~3,5 s
                per = 3.4 + (sem % 7) * 0.12; fase = (t + sem * .37) % per
                if fase < 0.2:
                    olho = min(olho, abs(fase - 0.08) / 0.08 if fase < 0.16 else 1)
        else:
            for k, (n, f) in enumerate((('seg1', .3), ('seg2', .8), ('seg3', 1.))):
                delta[n] = 1.4 * f * math.sin(2 * math.pi * 0.45 * t - 0.6 * k + sem)
            raiz['sx'] = raiz['sy'] = 1 + 0.008 * math.sin(2 * math.pi * t / 3.2)
    for a in p.get('acoes', []):
        nome, ini, fim = a['acao'], a.get('inicio', 0), a.get('fim', 1e9)
        e = env(t, ini, fim, a.get('rampa', 0.5 if nome in ('acenar', 'apontar', 'comemorar') else 0.35)); tl = t - ini; it = a.get('intensidade', 1.0)
        lado = a.get('lado', 'e')
        if nome == 'piscar':
            for tp in a.get('em', []):
                if 0 <= t - tp < 0.2:
                    x = t - tp; olho = min(olho, abs(x - 0.08) / 0.08 if x < 0.16 else 1)
            continue
        if e <= 0:
            continue
        if nome == 'acenar' and tipo == 'humano':
            somar(delta, braco_para(J, lado, dir_fora(lado, 32), -90 + (14 if lado == 'e' else -14), e=e, esc=esc))
            osc = 22 * it * math.sin(2 * math.pi * a.get('freq', 1.6) * tl)
            somar(delta, {f'antebraco_{lado}': osc * e, f'mao_{lado}': 0.5 * osc * e})
        elif nome == 'apontar' and tipo == 'humano':
            if 'alvo' in a:
                o = J[f'ombro_{lado}']; d = math.degrees(math.atan2(a['alvo'][1] - o[1], a['alvo'][0] - o[0]))
            else:
                d = dir_fora(lado, a.get('graus', 8))
            somar(delta, braco_para(J, lado, d, d + (6 if lado == 'e' else -6), e=e, esc=esc))
        elif nome == 'comemorar' and tipo == 'humano':
            for s in 'ed':
                somar(delta, braco_para(J, s, dir_fora(s, 58), dir_fora(s, 84), e=e, esc=esc))
                somar(delta, {f'antebraco_{s}': 6 * math.sin(2 * math.pi * 2.2 * tl) * e})
            raiz['dy'] -= 0.012 * h * abs(math.sin(math.pi * 2.2 * tl)) * e
        elif nome == 'inclinar_cabeca' and tipo == 'humano':
            g = a.get('graus', 9) * (1 if lado == 'd' else -1)
            somar(delta, {'cabeca': g * e, 'peito': -0.25 * g * e})
        elif nome == 'sim' and tipo == 'humano':
            q = abs(math.sin(2 * math.pi * a.get('freq', 1.3) * tl / 2))
            sa = esc.get('cabeca', (1, 1)); esc['cabeca'] = (sa[0] * (1 - 0.06 * it * q * e), sa[1])
            raiz['dy'] += 0.004 * h * q * e
        elif nome == 'falar':
            if a.get('audio'):
                ch = a['audio']
                if ch not in cache:
                    cache[ch] = envelope_audio(ch, 30)
                env_a = cache[ch]; k = int((t - a.get('audio_inicio', ini)) * 30)
                o = float(env_a[k]) if 0 <= k < len(env_a) else 0.0
            else:
                o = boca_sintetica(tl)
            boca = max(boca, o * e * it)
            if tipo == 'humano':
                somar(delta, {'cabeca': (2.2 * ruido(t, 0.45, sem) + 0.8 * o) * e})
                sa = esc.get('cabeca', (1, 1)); esc['cabeca'] = (sa[0] * (1 - 0.01 * o * e), sa[1])
            else:
                raiz['sy'] *= 1 + 0.025 * o * e; raiz['sx'] *= 1 - 0.012 * o * e
        elif nome in ('pular', 'quicar'):
            dur = (fim - ini) if nome == 'pular' else 1 / a.get('freq', 1.4)
            q = (tl % dur) / dur; alt = a.get('altura', 0.06 if nome == 'pular' else 0.035) * h
            if q < 0.2:
                sq = 0.06 * math.sin(math.pi * q / 0.2); raiz['sy'] *= 1 - sq; raiz['sx'] *= 1 + 0.6 * sq
            elif q < 0.8:
                z = (q - 0.2) / 0.6; raiz['dy'] -= alt * 4 * z * (1 - z)
                st = 0.035 * math.sin(math.pi * z); raiz['sy'] *= 1 + st; raiz['sx'] *= 1 - 0.5 * st
                if tipo == 'humano':
                    for s, f in (('e', 1), ('d', -1)):
                        somar(delta, {f'braco_{s}': f * 18 * math.sin(math.pi * z)})
            else:
                sq = 0.05 * math.sin(math.pi * (q - 0.8) / 0.2); raiz['sy'] *= 1 - sq; raiz['sx'] *= 1 + 0.6 * sq
        elif nome == 'balancar':
            ks = ('seg1', 'seg2', 'seg3') if tipo == 'cadeia' else ('tronco', 'peito', 'cabeca')
            for k, (n, f) in enumerate(zip(ks, (.3, .8, 1.))):        # base quase parada, topo balanca mais
                somar(delta, {n: a.get('graus', 6) * f * math.sin(2 * math.pi * a.get('freq', 0.8) * tl - 0.6 * k) * e})
        elif nome == 'pulsar':
            f = 1 + a.get('quanto', 0.03) * math.sin(2 * math.pi * a.get('freq', 1.2) * tl) * e
            raiz['sx'] *= f; raiz['sy'] *= f
        elif nome == 'tremer':
            raiz['ang'] += a.get('graus', 2.0) * ruido(t, 7, sem) * e
            raiz['dx'] += 0.004 * h * ruido(t, 9, sem + 3) * e
    return delta, esc, raiz, olho, boca

def envelope_audio(arq, fps):
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', arq, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True).stdout
    x = np.frombuffer(pcm, np.float32); hop = 16000 // fps
    n = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[max(0, i * hop - hop // 2): i * hop + hop + hop // 2] ** 2)) for i in range(n)])
    db = 20 * np.log10(rms + 1e-6); ref = np.percentile(db, 95)
    o = np.clip((db - (ref - 22)) / 18, 0, 1)
    out = np.zeros_like(o); v = 0.0
    for i, x_ in enumerate(o):
        v = x_ if x_ > v else v * 0.55 + x_ * 0.45
        out[i] = v
    return out

# ---------------------------------------------------------------- rosto (textura)
def piscar_tex(img, cx, s):
    x0, y0, x1, y1 = [int(round(v)) for v in cx]
    if s >= 0.98 or y1 - y0 < 3:
        return
    H, W = img.shape[:2]
    x0, x1, y0, y1 = max(x0, 0), min(x1, W - 1), max(y0, 2), min(y1, H - 3)
    h = y1 - y0; cy = (y0 + y1) / 2; s = max(s, 0.04)
    ys = np.arange(y0, y1, dtype=np.float32)
    src = np.where(np.abs(ys - cy) <= s * h / 2, cy + (ys - cy) / s, np.where(ys < cy, y0 - 2, y1 + 1))
    my = np.repeat(src[:, None], x1 - x0, 1).astype(np.float32)
    mx = np.repeat(np.arange(x0, x1, dtype=np.float32)[None], len(ys), 0)
    reg = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if s < 0.3:
        orig = img[y0:y1, x0:x1, :3].reshape(-1, 3); esc = orig[np.argsort(orig.sum(1))[:max(1, len(orig) // 20)]].mean(0) * 0.85
        lw = max(2, int(h * 0.07)); yc = int(cy - y0)
        ww = x1 - x0; xs = np.arange(ww); curva = (0.10 * h * ((xs - ww / 2) / (ww / 2)) ** 2).astype(int)
        for dx in range(int(ww * .12), int(ww * .88)):
            reg[max(yc - lw // 2 + curva[dx], 0):yc + lw // 2 + 1 + curva[dx], dx, :3] = esc
    img[y0:y1, x0:x1] = reg

def boca_tex(img, cx, o, k=1.6):
    if o <= 0.02:
        return
    x0, y0, x1, y1 = cx; w, h = x1 - x0, y1 - y0; mx_, my_ = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = img.shape[:2]
    X0, X1 = int(max(x0 - .25 * w, 0)), int(min(x1 + .25 * w, W)); Y0, Y1 = int(max(y0 - 1.3 * h, 0)), int(min(y1 + 1.6 * h, H))
    yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
    gx = np.clip(1 - ((xx - mx_) / (0.62 * w)) ** 2, 0, 1)
    gy = np.clip(1 - ((yy - my_) / (0.5 * (Y1 - Y0))) ** 2, 0, 1)
    sc = 1 + k * o * gx * gy
    sy = my_ + (yy - my_) / sc - 0.18 * h * o * gx * gy
    reg = cv2.remap(img, xx, sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dentro = (np.abs(sy - my_) < 0.5 * h) & (np.abs(xx - mx_) < 0.45 * w)
    borda = np.concatenate([reg[0, :, :3], reg[-1, :, :3], reg[:, 0, :3], reg[:, -1, :3]]).astype(np.float32)
    dif = np.linalg.norm(reg[..., :3].astype(np.float32) - np.median(borda, 0), axis=-1)
    cor_boca = np.clip((dif - 25) / 40, 0, 1)                      # escurece so o que tem cor de boca, nao a pele
    esc = 1 - 0.38 * o * cv2.GaussianBlur((dentro * cor_boca).astype(np.float32), (0, 0), 1.5)
    reg[..., :3] = (reg[..., :3] * esc[..., None]).astype(np.uint8)
    img[Y0:Y1, X0:X1] = reg

# ---------------------------------------------------------------- render
class Render:
    def __init__(self, W, H):
        import moderngl
        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context(backend='egl'); self.W, self.H = W, H
        self.fbo = self.ctx.simple_framebuffer((W, H), components=4)
        self.prog = self.ctx.program(vertex_shader='''#version 330
            uniform vec2 tam; in vec2 pos; in vec2 uv; out vec2 v_uv;
            void main(){ v_uv = uv; gl_Position = vec4(pos.x / tam.x * 2.0 - 1.0, 1.0 - pos.y / tam.y * 2.0, 0.0, 1.0); }''',
            fragment_shader='''#version 330
            uniform sampler2D tex; in vec2 v_uv; out vec4 cor;
            void main(){ cor = texture(tex, v_uv); }''')
        self.prog['tam'].value = (W, H)
        self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA

    def preparar(self, cam):
        cam.tex = self.ctx.texture((self.W, self.H), 4, premult(cam.rgba).tobytes())
        cam.tex.build_mipmaps(); cam.tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
        cam.vpos = self.ctx.buffer(reserve=cam.V.nbytes)
        uv = (cam.V / [self.W, self.H]).astype(np.float32)
        cam.vao = self.ctx.vertex_array(self.prog, [(cam.vpos, '2f', 'pos'), (self.ctx.buffer(uv.tobytes()), '2f', 'uv')],
                                        self.ctx.buffer(cam.T.astype('i4').tobytes()))

    def quadro(self, itens):
        self.fbo.use(); self.fbo.clear(0, 0, 0, 0)
        for cam, P, tex in itens:
            if tex is not None:
                cam.tex.write(premult(tex).tobytes()); cam.tex.build_mipmaps()
            cam.vpos.write(P.tobytes()); cam.tex.use(0); cam.vao.render()
        return np.frombuffer(self.fbo.read(components=4), np.uint8).reshape(self.H, self.W, 4)[::-1]

def premult(rgba):
    a = rgba[..., 3:4].astype(np.uint16)
    out = rgba.copy(); out[..., :3] = (rgba[..., :3] * a // 255).astype(np.uint8)
    return out

# ---------------------------------------------------------------- mapas de conferencia
CORES = {'corpo': (255, 210, 60), 'braco_e': (60, 200, 255), 'braco_d': (255, 90, 200), 'perna_e': (90, 230, 120), 'perna_d': (255, 150, 60)}

def mapa_rig(d, pid, fundo, rgba, rig):
    W, H = fundo.size
    base = fundo.convert('RGBA'); base.alpha_composite(Image.fromarray(rgba))
    im = Image.blend(base.convert('RGB'), Image.new('RGB', (W, H), (0, 0, 0)), 0.35)
    dr = ImageDraw.Draw(im); fn = fonte(16)
    for x in range(0, W, 100):
        dr.line([(x, 0), (x, H)], fill=(90, 90, 90) if x % 500 else (160, 160, 160), width=1)
        if x % 200 == 0:
            dr.text((x + 3, 3), str(x), fill=(200, 200, 200), font=fn)
    for y in range(0, H, 100):
        dr.line([(0, y), (W, y)], fill=(90, 90, 90) if y % 500 else (160, 160, 160), width=1)
        if y % 200 == 0:
            dr.text((3, y + 3), str(y), fill=(200, 200, 200), font=fn)
    J = rig['j']
    for nome, pai, de, para, g in ossos_de(rig['tipo']):
        dr.line([tuple(J[de]), tuple(J[para])], fill=CORES.get(g, (255, 255, 255)), width=5)
    for k, v in J.items():
        dr.ellipse([v[0] - 7, v[1] - 7, v[0] + 7, v[1] + 7], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        dr.text((v[0] + 9, v[1] - 8), k, fill=(255, 255, 255), font=fn, stroke_width=2, stroke_fill=(0, 0, 0))
    for c in rig.get('olhos', []):
        dr.rectangle(c, outline=(255, 40, 40), width=2)
    if rig.get('boca'):
        dr.rectangle(rig['boca'], outline=(255, 40, 40), width=2)
    c = rig['chao']; dr.line([(c[0] - 30, c[1]), (c[0] + 30, c[1])], fill=(255, 60, 60), width=4)
    im.save(f'{d}/{pid}_rig.png')
    if rig.get('olhos') or rig.get('boca'):                    # recorte do rosto ampliado 2x, grade de 20 px
        cxs = [b for b in rig.get('olhos', []) + ([rig['boca']] if rig.get('boca') else [])]
        x0, y0 = min(b[0] for b in cxs) - 60, min(b[1] for b in cxs) - 60
        x1, y1 = max(b[2] for b in cxs) + 60, max(b[3] for b in cxs) + 60
        x0, y0 = int(max(x0, 0)) // 20 * 20, int(max(y0, 0)) // 20 * 20
        r = base.convert('RGB').crop((x0, y0, int(x1), int(y1))); r = r.resize((r.width * 2, r.height * 2), Image.LANCZOS)
        dr = ImageDraw.Draw(r)
        for x in range(x0, int(x1), 20):
            dr.line([(2 * (x - x0), 0), (2 * (x - x0), r.height)], fill=(0, 255, 255) if x % 100 == 0 else (120, 120, 120))
            if x % 100 == 0:
                dr.text((2 * (x - x0) + 2, 2), str(x), fill=(0, 255, 255), font=fn)
        for y in range(y0, int(y1), 20):
            dr.line([(0, 2 * (y - y0)), (r.width, 2 * (y - y0))], fill=(0, 255, 255) if y % 100 == 0 else (120, 120, 120))
            if y % 100 == 0:
                dr.text((2, 2 * (y - y0) + 2), str(y), fill=(0, 255, 255), font=fn)
        for b in cxs:
            dr.rectangle([2 * (b[0] - x0), 2 * (b[1] - y0), 2 * (b[2] - x0), 2 * (b[3] - y0)], outline=(255, 40, 40), width=2)
        r.save(f'{d}/{pid}_rosto.png')
    lab, grupos = partes(rgba, rig, ossos_de(rig['tipo']))
    cor = np.array([CORES.get(g, (200, 200, 200)) for g in grupos], np.uint8)[lab]
    a = (rgba[..., 3:4] > 20)
    mix = np.where(a, (0.45 * rgba[..., :3] + 0.55 * cor).astype(np.uint8), 30)
    Image.fromarray(mix.astype(np.uint8)).save(f'{d}/{pid}_partes.png')
    avisos = []
    for g in grupos:                                           # membro encostado no corpo?
        if g == 'corpo':
            continue
        m = (lab == grupos.index(g)) & a[..., 0]; c = (lab == grupos.index('corpo')) & a[..., 0]
        contato = m & cv2.dilate(c.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        oss = next(o for o in ossos_de(rig['tipo']) if o[4] == g); rj = np.array(J[oss[2]])
        ys, xs = np.nonzero(contato)
        longe = np.hypot(xs - rj[0], ys - rj[1]) > 0.08 * rig.get('altura', 800)
        if longe.sum() > 0.04 * rig.get('altura', 800):
            avisos.append(f'{g} encostado no corpo ({int(longe.sum())} px de contato longe da junta): '
                          f'se ele for se mover muito, inclua "{g}" em "separar"')
    return avisos

# ---------------------------------------------------------------- comandos
def cmd_preparar(a):
    d = a.dir; os.makedirs(d, exist_ok=True); W, H = a.larg, a.alt
    pers = []
    if a.cena:
        cena = np.asarray(ajustar(Image.open(a.cena).convert('RGB'), W, H))
        m = recortar(cena).astype(np.float32)
        caixas = [list(map(int, c.split(','))) for c in a.caixa] if a.caixa else None
        if not caixas:                       # um personagem por grupo de pedacos proximos (braco solto, chapeu)
            n, lab, st, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8))
            caixas = []
            for i in np.argsort(-st[1:, 4]) + 1:
                if st[i, 4] < 0.002 * W * H:
                    continue
                c = [st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]]
                for b in caixas:
                    mg = 0.06 * (b[3] - b[1])
                    if c[0] < b[2] + mg and c[2] > b[0] - mg and c[1] < b[3] + mg and c[3] > b[1] - mg:
                        b[:] = [min(b[0], c[0]), min(b[1], c[1]), max(b[2], c[2]), max(b[3], c[3])]; break
                else:
                    if st[i, 4] > 0.01 * W * H:
                        caixas.append(c)
            print('caixas dos personagens:', caixas)
        uniao = np.zeros((H, W), bool)
        for k, (x0, y0, x1, y1) in enumerate(caixas, 1):
            mk = np.zeros((H, W), np.float32); mk[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            rgba = np.dstack([cena, mk.astype(np.uint8)])
            Image.fromarray(rgba).save(f'{d}/p{k}.png'); pers.append(f'p{k}'); uniao |= mk > 20
        buraco = cv2.dilate(uniao.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        print('reconstruindo o fundo atras dos personagens (LaMa)...')
        Image.fromarray(lama(cena, buraco)).save(f'{d}/fundo.png')
    else:
        ajustar(Image.open(a.fundo).convert('RGB'), W, H).save(f'{d}/fundo.png')
        for k, p in enumerate(a.personagem, 1):
            im = ajustar(Image.open(p).convert('RGBA'), W, H)
            im.save(f'{d}/p{k}.png'); pers.append(f'p{k}')
    json.dump({'larg': W, 'alt': H, 'personagens': pers}, open(f'{d}/projeto.json', 'w'), indent=1)
    fundo = Image.open(f'{d}/fundo.png')
    for pid in pers:
        rgba = np.asarray(Image.open(f'{d}/{pid}.png'))
        rig = rig_auto(rgba, a.tipo)
        json.dump(rig, open(f'{d}/{pid}_rig.json', 'w'), indent=1, ensure_ascii=False)
        avisos = mapa_rig(d, pid, fundo, rgba, rig)
        print(f'{pid}: rig {rig["tipo"]}, {len(rig["j"])} juntas -> {pid}_rig.png, {pid}_partes.png'
              + (f', {pid}_rosto.png' if rig.get('olhos') else ''))
        for av in avisos:
            print('  AVISO:', av)

def cmd_checar(a):
    d = a.dir; pj = json.load(open(f'{d}/projeto.json')); fundo = Image.open(f'{d}/fundo.png')
    for pid in pj['personagens']:
        rig = json.load(open(f'{d}/{pid}_rig.json'))
        for av in mapa_rig(d, pid, fundo, np.asarray(Image.open(f'{d}/{pid}.png')), rig):
            print(f'{pid} AVISO:', av)
        print(f'{pid}: mapas refeitos')

def cmd_animar(a):
    d = a.dir; pj = json.load(open(f'{d}/projeto.json')); R = json.load(open(a.roteiro))
    W, H = pj['larg'], pj['alt']; fps = R.get('fps', 30); dur = R['duracao']
    esc = 0.5 if a.previa else 1.0; Wo, Ho = int(W * esc) // 2 * 2, int(H * esc) // 2 * 2
    fundo = np.asarray(Image.open(f'{d}/{R.get("fundo", "fundo.png")}').convert('RGB')).astype(np.float32)
    rend = Render(W, H); persos = []
    for p in R['personagens']:
        rig = json.load(open(f'{d}/{p["id"]}_rig.json'))
        rgba = np.asarray(Image.open(f'{d}/{p["id"]}.png').convert('RGBA'))
        print(f'{p["id"]}: montando malha e pesos...')
        cams, ossos = montar_personagem(rgba, rig, 12 if a.previa else 8)
        caixas = rig.get('olhos', []) + ([rig['boca']] if rig.get('boca') else [])
        for c in cams:
            rend.preparar(c)
            c.rosto = any(c.rgba[int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2), 3] > 128 for b in caixas)
        persos.append((p, rig, cams, ossos))
    cache = {}; n = int(round(dur * fps)); folha = []; alerta = {}
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{Wo}x{Ho}', '-r', str(fps), '-i', '-']
    if R.get('audio'):
        cmd += ['-itsoffset', str(R.get('audio_inicio', 0)), '-i', R['audio'], '-c:a', 'aac', '-b:a', '192k']
    cmd += ['-t', f'{dur:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', a.saida]
    quadros = sorted({min(int(round(float(x) * fps)), n - 1) for x in a.quadros.split(',')}) if a.quadros else None
    ff = None if quadros else subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fa = None
    if a.alfa and not quadros:
        ca = (['-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le'] if a.alfa.endswith('.mov') else
              ['-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', '0', '-crf', '32', '-deadline', 'realtime', '-cpu-used', '8', '-auto-alt-ref', '0'])
        fa = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{Wo}x{Ho}', '-r', str(fps), '-i', '-']
                              + ca + [a.alfa], stdin=subprocess.PIPE)
    cam_cfg = R.get('camera', {}); z0, z1 = cam_cfg.get('zoom', [1, 1]); foco = cam_cfg.get('foco', [W / 2, H / 2])
    par = cam_cfg.get('paralaxe', 0.4)
    marcas = set(np.linspace(0, n - 1, 12).astype(int))
    base = os.path.splitext(a.saida)[0]
    for i in (quadros or range(n)):
        t = i / fps; itens = []
        for p, rig, cams, ossos in persos:
            delta, es, raiz, olho, boca = estado(p, rig, t, cache)
            M = fk(rig, ossos, delta, es, raiz)
            for c in cams:
                P = deformar(c, M)
                ar = area(P, c.T)[c.cheio] / (c.area0[c.cheio] + 1e-6)
                if len(ar):                       # % da area visivel dobrada (<0) ou esticada (>2x)
                    ruim = 100 * (np.sum(c.area0[c.cheio][ar < 0]) + np.sum(c.area0[c.cheio][ar > 2])) / np.sum(c.area0[c.cheio])
                    if ruim > alerta.get(p['id'], (0, 0))[0]:
                        alerta[p['id']] = (ruim, t)
                tx = None
                if (olho < 0.98 or boca > 0.02) and c.rosto:
                    tx = c.rgba.copy()
                    for b in rig.get('olhos', []):
                        piscar_tex(tx, b, olho)
                    if rig.get('boca'):
                        boca_tex(tx, rig['boca'], boca)
                    c._sujo = True
                elif getattr(c, '_sujo', False):
                    tx = c.rgba; c._sujo = False
                itens.append((c, P, tx))
        fg = rend.quadro(itens).astype(np.float32)
        q = t / dur; z = z0 + (z1 - z0) * (q * q * (3 - 2 * q))
        def cam(img, zz):
            if abs(zz - 1) < 1e-4:
                return img
            Mz = np.float32([[zz, 0, foco[0] * (1 - zz)], [0, zz, foco[1] * (1 - zz)]])
            return cv2.warpAffine(img, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bg = cam(fundo, 1 + (z - 1) * (1 - par)); fg = cam(fg, z)
        al = fg[..., 3:4] / 255
        out = fg[..., :3] + bg * (1 - al)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if esc != 1:
            out = cv2.resize(out, (Wo, Ho), interpolation=cv2.INTER_AREA)
        if quadros:
            Image.fromarray(out).save(f'{base}_t{t:.2f}.png'); print(f'quadro {t:.2f} s -> {base}_t{t:.2f}.png'); continue
        ff.stdin.write(out.tobytes())
        if fa:
            rgb = np.where(al > 0, fg[..., :3] / np.maximum(al, 1e-3), 0)
            fr = np.dstack([np.clip(rgb, 0, 255), fg[..., 3]]).astype(np.uint8)
            fa.stdin.write((cv2.resize(fr, (Wo, Ho), interpolation=cv2.INTER_AREA) if esc != 1 else fr).tobytes())
        if i in marcas:
            folha.append((t, cv2.resize(out, (270, 480), interpolation=cv2.INTER_AREA)))
        if i % fps == 0:
            print(f'  {t:4.1f} s / {dur} s', flush=True)
    if quadros:
        return
    ff.stdin.close(); ff.wait()
    if fa:
        fa.stdin.close(); fa.wait()
    fl = Image.new('RGB', (6 * 278 + 8, 2 * 512 + 8), (30, 30, 30)); dr = ImageDraw.Draw(fl)
    for k, (t, im) in enumerate(folha):
        x, y = 8 + (k % 6) * 278, 8 + (k // 6) * 512
        fl.paste(Image.fromarray(im), (x, y + 24)); dr.text((x + 4, y + 2), f'{t:.2f}s', fill=(255, 210, 80), font=fonte(18))
    fl.save(f'{base}_folha.png')
    print(f'ok: {a.saida} ({Wo}x{Ho}, {fps} fps, {dur} s), folha {base}_folha.png')
    for pid, (v, t) in alerta.items():
        flag = '  <- conferir com --quadros' if v > 0.3 else ''
        print(f'  {pid}: pior quadro {t:.2f} s, {v:.2f}% da area dobrada ou esticada 2x{flag}')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('preparar'); p.add_argument('--dir', required=True); p.add_argument('--cena')
    p.add_argument('--fundo'); p.add_argument('--personagem', action='append'); p.add_argument('--caixa', action='append')
    p.add_argument('--tipo', choices=['humano', 'cadeia']); p.add_argument('--larg', type=int, default=1080); p.add_argument('--alt', type=int, default=1920)
    p = sp.add_parser('checar'); p.add_argument('--dir', required=True)
    p = sp.add_parser('animar'); p.add_argument('--dir', required=True); p.add_argument('--roteiro', required=True)
    p.add_argument('--saida', required=True); p.add_argument('--alfa'); p.add_argument('--previa', action='store_true')
    p.add_argument('--quadros', help='so salva PNGs nesses tempos, ex.: 0.5,1.6 (sem video)')
    a = ap.parse_args()
    if a.cmd == 'preparar' and not (a.cena or (a.fundo and a.personagem)):
        ap.error('use --cena OU --fundo com --personagem')
    {'preparar': cmd_preparar, 'checar': cmd_checar, 'animar': cmd_animar}[a.cmd](a)

if __name__ == '__main__':
    main()
