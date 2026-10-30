"""Liquido dentro do recipiente: superficie horizontal e volume real (modelo panela3d), com cor procedural medida
na poca original da imagem (gradiente por profundidade) e brilho escorrendo para o bico."""
import math
import numpy as np, cv2


class LiquidoRecipiente:
    def __init__(self, pan, cena_rgb, mask_orig, centro=(389.5, 780.0), y_lim=872):
        self.pan = pan; self.cx, self.cy = centro
        m = mask_orig.copy(); m[y_lim:] = False
        ys, xs = np.nonzero(m)
        self.xl_o, self.xr_o = xs.min(), xs.max()
        self.top_o = np.full(cena_rgb.shape[1], np.nan, np.float32); self.bot_o = self.top_o.copy()
        for x in range(self.xl_o, self.xr_o + 1):
            c = np.nonzero(m[:, x])[0]
            if len(c):
                self.top_o[x], self.bot_o[x] = c.min(), c.max() + 1
        ok = ~np.isnan(self.top_o)
        xs_ok = np.nonzero(ok)[0]
        self.top_o = np.interp(np.arange(len(self.top_o)), xs_ok, self.top_o[ok]).astype(np.float32)
        self.bot_o = np.interp(np.arange(len(self.bot_o)), xs_ok, self.bot_o[ok]).astype(np.float32)
        self.tex = cena_rgb.astype(np.float32)
        # caixa de trabalho em coordenadas do modelo (relativas ao centro da borda)
        self.x0, self.x1, self.y0, self.y1 = -112, 112, -104, 110
        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.X, self.Y = xx, yy
        r = np.random.default_rng(12)
        self.ruido = cv2.GaussianBlur(r.normal(0, 1, xx.shape).astype(np.float32), (0, 0), sigmaX=5, sigmaY=2.5)
        self.ruido /= np.abs(self.ruido).max() + 1e-6
        self.grao = r.normal(0, 1, (8,) + xx.shape).astype(np.float32)

    def camada(self, beta, h, Q, t, Cp, gama, labio, W, H, q_ref=40.0):
        """RGBA pre-multiplicado (float32, quadro inteiro) com a poca de oleo na frigideira posada."""
        m, esp, q, s = self.pan.superficie(h, beta, self.X, self.Y)
        if m.sum() < 3:
            return None
        cols = np.nonzero(m.any(0))[0]
        top = np.full(m.shape[1], np.nan, np.float32); bot = top.copy()
        idx = np.arange(m.shape[0], dtype=np.float32)[:, None]
        mt = np.where(m, idx, np.inf).min(0); mb = np.where(m, idx, -np.inf).max(0) + 1
        top[cols], bot[cols] = mt[cols], mb[cols]
        ok = ~np.isnan(top)
        top = np.interp(np.arange(len(top)), np.nonzero(ok)[0], top[ok]).astype(np.float32)
        bot = np.interp(np.arange(len(bot)), np.nonzero(ok)[0], bot[ok]).astype(np.float32)
        # cor procedural medida na poca original: linha da superficie mais escura, corpo ambar,
        # metade de baixo mais clara (luz da janela atravessando o oleo), com variacao lenta
        w = np.clip((idx - top[None, :]) / np.maximum(bot - top, 1)[None, :], 0, 1)
        wp = np.array([0, .15, .45, .85, 1.0])
        pal = np.array([[132, 87, 18], [156, 100, 6], [146, 93, 8], [178, 121, 22], [168, 112, 18]], np.float32)
        cor = np.stack([np.interp(w, wp, pal[:, c]) for c in range(3)], -1)
        var = self.ruido[..., None] * (6 + 14 * np.clip((w - 0.55) / 0.45, 0, 1))[..., None]
        cor = cor + var * np.array([1.0, 0.8, 0.35])
        # brilho que escorre para o bico enquanto ha vazao: faixas alongadas na direcao do bico, andando para ele
        if Q > 0.3:
            lx, ly = labio
            dl = np.hypot(self.X - lx, (self.Y - ly) * 1.8)
            ang = np.arctan2((self.Y - ly) * 1.8, self.X - lx)
            hgt, wid = self.ruido.shape
            u = (ang / math.pi * 0.5 + 0.5) * (wid - 1)                     # angulo -> coluna do ruido
            v = np.mod(dl * 0.35 + 22.0 * t, hgt - 1)                       # distancia -> linha, andando para o bico
            faixa = cv2.remap(self.ruido, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
            f = 1 + 0.10 * min(Q / q_ref, 1.3) * np.clip(faixa - 0.35, 0, None) * np.exp(-dl / 140)
            cor = cor * f[..., None]
        cor = cor + self.grao[int(t * 30) % 8][..., None] * 1.6                      # granulacao igual a da imagem
        # borda: suave, e mais transparente onde a camada de oleo e fina (linha de contato)
        alfa = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.6) * np.clip(0.72 + esp / 3.0, 0, 1)
        L = np.dstack([np.clip(cor, 0, 255) * alfa[..., None], alfa * 255]).astype(np.float32)
        Mx = np.float32([[math.cos(gama), -math.sin(gama), 0], [math.sin(gama), math.cos(gama), 0]])
        o = np.array([self.x0, self.y0], np.float32)
        Mx[:, 2] = np.array(Cp, np.float32) + Mx[:, :2] @ o
        return cv2.warpAffine(L, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
