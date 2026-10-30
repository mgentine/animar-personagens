"""Detalhes de ambiente para a cena do oleo: vento, sombra de folhagem na parede, planta do balcao e
reflexo do oleo na frigideira. Tudo dirigido por um vento unico, com rajadas irregulares."""
import math
import numpy as np, cv2


def s5(x):
    x = min(max(x, 0.0), 1.0); return x * x * x * (x * (6 * x - 15) + 10)


def ruido(t, sem, f=0.35):
    r = np.random.default_rng(sem)
    fr = f * np.array([1.0, 1.618, 2.414, 3.303]); ph = r.uniform(0, 2 * np.pi, 4); am = np.array([1, .6, .35, .2])
    return float((am * np.sin(2 * np.pi * fr * t + ph)).sum() / am.sum())


class Vento:
    """Intensidade 0..1 com rajadas lentas e irregulares; fase do balanco com variacao de frequencia."""
    def __init__(self, sem=31):
        self.sem = sem

    def forca(self, t):
        g = 0.5 + 0.5 * ruido(t, self.sem, 0.16)
        return 0.30 + 0.70 * g ** 1.4

    def balanco(self, t, k=0):                        # -1..1, lento, nao periodico
        return 0.72 * ruido(t, self.sem + 10 + k, 0.34) + 0.28 * ruido(t, self.sem + 20 + k, 0.61)


def _campo(h, w, sigma, sem):
    f = cv2.GaussianBlur(np.random.default_rng(sem).normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    return f / (np.abs(f).max() + 1e-6)


class SombraFolhas:
    """Separa a sombra das folhas (fator S <= 1) da parede iluminada (B0 = I / S) numa regiao e,
    a cada quadro, desloca S com o vento: giro do galho em torno do pivo + tremor das folhas."""
    def __init__(self, fundo, caixa, faixas, pivo, feather, vento):
        self.x0, self.y0, self.x1, self.y1 = caixa
        reg = fundo[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        S = np.ones_like(reg)
        for a, b in faixas:                           # faixas horizontais com cor de base diferente (azulejo, friso)
            fa = reg[a - self.y0:b - self.y0]
            ng = np.stack([cv2.medianBlur(np.clip(fa[..., c], 0, 255).astype(np.uint8), 11).astype(np.float32)
                           for c in range(3)], -1)                 # mediana tira rejunte claro e escuro
            lit = np.stack([cv2.GaussianBlur(cv2.morphologyEx(ng[..., c], cv2.MORPH_CLOSE, el(61)), (0, 0), 6)
                            for c in range(3)], -1)
            S[a - self.y0:b - self.y0] = np.clip(ng / np.maximum(lit, 1), 0.25, 1.0)
        self.S = S; self.reg = reg; self.B0 = reg / S             # sem corte: B0*S devolve a imagem exata
        h, w = S.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.X, self.Y = xx, yy
        fl, ft, fb = feather                          # largura das rampas: esquerda, topo, base
        self.F = (np.clip(xx / fl, 0, 1) * np.clip(yy / ft, 0, 1) * np.clip((h - 1 - yy) / fb, 0, 1))[..., None]
        self.px, self.py = pivo[0] - self.x0, pivo[1] - self.y0
        self.dist = np.hypot(xx - self.px, yy - self.py)
        self.cf = [(_campo(h, w, 14, 100 + k), _campo(h, w, 14, 200 + k)) for k in range(4)]
        self.fr = [1.35, 1.9, 2.45, 3.1]; self.ph = np.random.default_rng(5).uniform(0, 2 * np.pi, (4, 2))
        self.amp_lenta = _campo(h, w, 70, 300) * 0.35 + 1.0     # partes do galho balancam um pouco diferente
        self.vento = vento
        ky = np.ones(h, np.float32)                    # nas linhas de rejunte entre faixas a sombra so anda na horizontal
        for a, _ in faixas[1:]:
            ky *= np.clip(np.abs(np.arange(h) + self.y0 - a) / 5.0, 0, 1)
        self.ky = cv2.GaussianBlur(ky[:, None], (0, 0), 1.5)[:, 0][:, None]

    def deslocamento(self, t):
        v = self.vento; g = v.forca(t - 0.1)
        th = math.radians(2.3 * g * v.balanco(t))                 # galho gira em torno do pivo (fora do quadro)
        c, s = math.cos(th), math.sin(th)
        rx, ry = self.X - self.px, self.Y - self.py
        dx = (c * rx - s * ry - rx) * self.amp_lenta + 3.0 * g * v.balanco(t, 3)
        dy = (s * rx + c * ry - ry) * self.amp_lenta + 1.0 * g * v.balanco(t, 4)
        af = 3.6 * g ** 1.3                                        # tremor das folhas, mais forte nas rajadas
        for k, (cx, cy) in enumerate(self.cf):
            dx += af * cx * math.sin(2 * math.pi * self.fr[k] * t + self.ph[k, 0]) / 2
            dy += af * cy * math.sin(2 * math.pi * self.fr[k] * t + self.ph[k, 1]) / 2
        return dx, dy * self.ky

    def aplicar(self, img, t):
        dx, dy = self.deslocamento(t)
        St = cv2.remap(self.S, self.X - dx, self.Y - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        nova = self.B0 * St
        out = img.copy()
        out[self.y0:self.y1, self.x0:self.x1] = nova * self.F + self.reg * (1 - self.F)
        return out


class Planta:
    """Folhas (pixels verdes) balancam com o vento, mais no alto que perto do vaso; torneira e parede ficam."""
    def __init__(self, fundo, caixa, base_y, vento, atraso=0.45):
        self.x0, self.y0, self.x1, self.y1 = caixa
        reg = fundo[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        g, b = reg[..., 1], reg[..., 2]
        folha = (g - b > 55) & (b < 110) & (g > 40)          # folhas amareladas pela luz quente
        folha = cv2.morphologyEx(folha.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.GaussianBlur(cv2.dilate(folha, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2.5)
        h, w = m.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        alt = np.clip((base_y - self.y0 - yy) / 85, 0, 1) ** 1.3
        self.peso = m * alt; self.X, self.Y = xx, yy; self.reg = reg
        self.cf = _campo(h, w, 10, 400); self.vento = vento; self.atraso = atraso
        self.npx = int((self.peso > 0.05).sum())

    def aplicar(self, img, t):
        v = self.vento; tt = t - self.atraso; g = v.forca(tt)
        dx = self.peso * (3.6 * g * v.balanco(tt, 7) + 1.0 * g * self.cf * math.sin(2 * math.pi * 1.7 * t))
        dy = self.peso * (0.8 * g * v.balanco(tt, 8))
        out = img.copy()
        out[self.y0:self.y1, self.x0:self.x1] = cv2.remap(self.reg, self.X - dx, self.Y - dy, cv2.INTER_LINEAR,
                                                          borderMode=cv2.BORDER_REPLICATE)
        return out


class BrilhoOleo:
    """Reflexos que escorrem pela poca de oleo da frigideira em direcao a borda (aplicado na textura)."""
    def __init__(self, rgba, caixa, labio):
        self.x0, self.y0, self.x1, self.y1 = caixa
        reg = rgba[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
        oleo = (r > 120) & (r - b > 60) & (g > 70) & (reg[..., 3] > 200)
        oleo = cv2.morphologyEx(oleo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.m = cv2.GaussianBlur(cv2.erode(oleo, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.5)
        h, w = self.m.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.d = np.hypot(xx - (labio[0] - self.x0), yy - (labio[1] - self.y0))
        self.ruido = _campo(h, w, 5, 500)
        self.base = reg; self.npx = int((self.m > 0.5).sum())

    def aplicar(self, tex, t, fluxo):
        if fluxo <= 0.01:
            return
        ondas = (np.sin(2 * math.pi * (self.d / 26 + 1.1 * t) + 2.0 * self.ruido) +
                 0.5 * np.sin(2 * math.pi * (self.d / 11 + 1.9 * t) + 3.0 * self.ruido))
        realce = np.clip(ondas - 0.6, 0, None)                      # so as cristas brilham (reflexo da janela)
        f = 1 + 0.16 * fluxo * self.m * realce
        reg = self.base.copy(); reg[..., :3] = np.clip(reg[..., :3] * f[..., None], 0, 255)
        tex[self.y0:self.y1, self.x0:self.x1] = reg.astype(np.uint8)
