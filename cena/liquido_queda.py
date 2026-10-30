"""Liquido em queda, dirigido pela simulacao (simulacao.npz):
- Fio: cada altura do fio carrega a vazao do instante em que o oleo saiu do bico (inicio e fim descem com a
  queda); largura ~ raiz(Q); vazao baixa vira fio que se rompe em contas; frente arredondada no inicio.
- Gotas finais: formam no bico, penduram, soltam e caem.
- Poca: raio da simulacao (area molhada nao recua), espessura pelo volume, recortada pela cuba, com o oleo
  passando a borda e escorrendo no fundo da cuba."""
import math
import numpy as np, cv2


def _campo1d(n, sigma, sem):
    f = cv2.GaussianBlur(np.random.default_rng(sem).normal(0, 1, (n, 1)).astype(np.float32), (0, 0), sigma).ravel()
    return f / (np.abs(f).max() + 1e-6)


def s5(x):
    x = min(max(x, 0.0), 1.0); return x * x * x * (x * (6 * x - 15) + 10)


def colar(acc, s, ox, oy):
    """Cola sprite pre-multiplicado em posicao fracionaria."""
    H, W = acc.shape[:2]
    ix, iy = int(math.floor(ox)), int(math.floor(oy)); fx, fy = ox - ix, oy - iy
    h, w = s.shape[:2]
    s2 = cv2.warpAffine(s, np.float32([[1, 0, fx], [0, 1, fy]]), (w + 2, h + 2), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x0, y0 = max(ix, 0), max(iy, 0); x1, y1 = min(ix + w + 2, W), min(iy + h + 2, H)
    if x1 <= x0 or y1 <= y0:
        return
    reg = s2[y0 - iy:y1 - iy, x0 - ix:x1 - ix]; al = reg[..., 3:4] / 255
    acc[y0:y1, x0:x1] = reg + acc[y0:y1, x0:x1] * (1 - al)


class Fio:
    def __init__(self, spr, fy0, fx0, labio, y_corte, pouso_y, sim, q_ref=40.0, g=8900.0, v0=230.0):
        self.spr = spr; self.fy0 = fy0; self.cx = labio[0] - fx0; self.labio = np.array(labio, float)
        self.D0 = y_corte - labio[1]; self.pouso = pouso_y; self.G = g; self.V0 = v0
        self.T, self.Q = sim['T'], sim['Q']; self.q_ref = q_ref
        self.brilho = _campo1d(8000, 9, 3); self.pulso = _campo1d(8000, 14, 4)
        self.vD0 = math.sqrt(v0 * v0 + 2 * g * self.D0)
        self.grao = np.random.default_rng(31).normal(0, 1, (6, 400, spr.shape[1])).astype(np.float32)
        gota = spr[-26:].copy(); gota *= np.clip(np.linspace(-0.3, 1.7, gota.shape[0]), 0, 1)[:, None, None]
        self.gota = gota
        i_on = np.argmax(self.Q > 0.3); self.t_on = self.T[i_on]

    def q(self, tt):
        return np.interp(tt, self.T, self.Q, left=0.0, right=0.0)

    def vel(self, dy):
        return np.sqrt(self.V0 ** 2 + 2 * self.G * np.maximum(dy, 0))

    def desenhar(self, acc, t, hist):
        lab, _ = hist(t)
        L = int(math.ceil(self.pouso - lab[1])) + 2
        w_out = self.spr.shape[1]
        dy = np.arange(L, dtype=np.float32)
        tau = (-self.V0 + np.sqrt(self.V0 ** 2 + 2 * self.G * dy)) / self.G
        t_emit = t - tau
        qe = self.q(t_emit)                                                     # vazao de quando o oleo saiu
        if qe.max() < 0.3:
            return None
        amostra = np.arange(0, L, 4)
        past = np.array([hist(t - tau[k]) for k in amostra])
        px = np.interp(dy, amostra, past[:, 0, 0] + past[:, 1, 0] * tau[amostra]) - lab[0]
        esc = np.sqrt(np.clip(qe / self.q_ref, 0, 1.4))                           # largura ~ raiz(Q)
        esc = np.maximum(esc, 0.22 * (qe > 0.3))                                 # fio fino minimo visivel
        rel = np.clip(dy / max(self.D0, 1), 0, 1)
        afina = np.sqrt(self.vD0 / self.vel(dy))
        k = (np.arange(L) + int(t * 900)) % len(self.pulso)
        varic = 1 + 0.06 * self.pulso[k] * np.clip((dy - self.D0) / 60, 0, 1)
        m = np.clip((dy - self.D0 + 6) / 12, 0, 1)
        larg = (1 - m) * esc + m * esc * afina * varic
        larg = np.maximum(larg, 1e-3)
        src_row = (self.labio[1] - self.fy0) + np.minimum(dy, self.D0)
        xx = np.arange(w_out, dtype=np.float32)[None, :].repeat(L, 0)
        fx_ = lab[0] - math.floor(lab[0]); fy_ = lab[1] - math.floor(lab[1])
        src_x = self.cx + (xx - self.cx - px[:, None] - fx_) / larg[:, None]
        src_y = np.repeat((src_row - fy_)[:, None], w_out, 1)
        s = cv2.remap(self.spr, src_x.astype(np.float32), src_y.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        u = (xx - self.cx - px[:, None] - fx_) / np.maximum(3.5 * larg[:, None], 0.6)
        vol = (1 - 0.28 * np.clip(np.abs(u), 0, 1) ** 2) + 0.45 * np.exp(-((u - 0.38) / 0.16) ** 2)
        s[..., :3] *= (1 + (vol - 1) * m[:, None])[..., None]
        # vazao baixa: o fio fino se rompe em contas que descem com o oleo (fase = instante de saida)
        f_b = np.clip(qe / 0.05, 1, 80)                                         # contas/s (gota de ~0,05 mL)
        fase = np.mod(np.cumsum(np.r_[0, np.diff(t_emit)] * -f_b) + t * 30, 1.0)
        contas = np.clip(1.8 * (0.5 + 0.5 * np.cos(2 * np.pi * fase)) - 0.55, 0, 1)
        quebra = np.clip((3.0 - qe) / 2.2, 0, 1) * (dy > 30)                      # so rompe longe do bico
        gate = np.clip((qe - 0.3) / 0.6, 0, 1) * (1 - quebra + quebra * contas)
        s *= gate[:, None, None]
        kb = (np.arange(L) + int(t * 620)) % len(self.brilho)
        s[..., :3] *= (1 + 0.08 * self.brilho[kb] * np.clip(rel * 5, 0, 1))[:, None, None]
        gr = self.grao[int(t * 30) % 6][:L] if L <= 400 else 0
        s[..., :3] += (gr * 1.5)[..., None] * (s[..., 3:4] / 255) * m[:, None, None]
        chegou = qe[-1] > 0.3
        if chegou:
            s[-3:] *= np.array([0.66, 0.33, 0.1])[:, None, None]
        ox, oy = int(math.floor(lab[0]) - self.cx), int(math.floor(lab[1]))
        H, W = acc.shape[:2]
        x0, y0 = max(ox, 0), max(oy, 0); x1, y1 = min(ox + w_out, W), min(oy + L, H)
        reg = s[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; al = reg[..., 3:4] / 255
        acc[y0:y1, x0:x1] = reg + acc[y0:y1, x0:x1] * (1 - al)
        # frente do primeiro despejo: gota arredondada na ponta, antes de chegar ao tampo
        if not chegou and t > self.t_on:
            vivo = np.nonzero(qe > 0.3)[0]
            if len(vivo):
                yf = vivo.max(); dsc = float(esc[yf] * afina[yf])
                g = cv2.resize(self.gota, (self.gota.shape[1], max(int(self.gota.shape[0] * (0.6 + 0.6 * dsc)), 3)))
                g = cv2.resize(g, (max(int(g.shape[1] * max(dsc * 1.3, 0.5)), 3), g.shape[0]))
                colar(acc, g, lab[0] + px[yf] - g.shape[1] / 2 + 0.5, lab[1] + yf - g.shape[0] * 0.7)
        return (lab[0] + px[-1], self.pouso, float(larg[-1]) * 7.0) if chegou else None


class Gotas:
    """Gotas que pingam do bico depois que o fio acaba: formam (penduradas), soltam e caem."""
    def __init__(self, gota_spr, pouso, inicios, g=8900.0):
        self.g = gota_spr; self.pouso = pouso; self.ini = inicios; self.G = g

    def desenhar(self, acc, t, hist):
        pousos = []
        for t0 in self.ini:
            tf = t0 + 0.30                                                          # 0,30 s pendurada crescendo
            if t < t0:
                continue
            if t < tf:
                lab, _ = hist(t); e = s5((t - t0) / 0.30)
                g = cv2.resize(self.g, (max(int(self.g.shape[1] * (0.35 + 0.35 * e)), 3), max(int(self.g.shape[0] * (0.25 + 0.55 * e)), 3)))
                colar(acc, g * (0.5 + 0.5 * e), lab[0] - g.shape[1] / 2 + 0.5, lab[1] - 2)
                continue
            lab0, v0 = hist(tf); tau = t - tf
            y = lab0[1] + 14 + 40 * tau + 0.5 * self.G * tau * tau
            if y > self.pouso + 2:
                if y - (40 + self.G * tau) / 30 <= self.pouso + 2:
                    pousos.append(t)                                                  # pousou neste quadro
                continue
            vel = 40 + self.G * tau
            rastro = 1 + min(vel / 30 / self.g.shape[0] * 0.5, 1.4)
            g = cv2.resize(self.g, (max(int(self.g.shape[1] * 0.7), 3), max(int(self.g.shape[0] * 0.8 * rastro), 3)))
            colar(acc, g, lab0[0] + v0[0] * tau - g.shape[1] / 2 + 0.5, y - g.shape[0] * 0.7)
        return pousos


def mascara_cuba(poligono, W=1080, H=1920):
    m = np.zeros((H, W), np.uint8)
    if poligono:
        cv2.fillPoly(m, [np.array(poligono, np.int32)], 1)
    return m


class PocaSim:
    def __init__(self, fundo, centro, achat, sim, caixa, cuba=None):
        cuba = cuba or {}
        self.canto = cuba.get('canto'); self.dirf = cuba.get('direcao_fundo', (1.0, 0.0)); self.pt_ponte = cuba.get('ponte')
        self.cx, self.cy = centro; self.achat = achat
        self.x0, self.y0, self.x1, self.y1 = caixa
        self.base = fundo[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        h, w = self.base.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.X, self.Y = xx + self.x0, yy + self.y0
        self.dx = self.X - self.cx; self.dy = (self.Y - self.cy) / achat
        self.rho = np.hypot(self.dx, self.dy); self.th = np.arctan2(self.dy, self.dx)
        r = np.random.default_rng(9); self.lob = [(r.uniform(0, 2 * np.pi), a, n) for n, a in ((2, .07), (3, .05), (5, .035), (7, .02))]
        cuba = mascara_cuba(cuba.get('poligono'))[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.cuba = cv2.GaussianBlur(cuba, (0, 0), 0.8)
        self.T, self.R, self.Vp, self.Qo, self.dren = sim['T'], sim['R'], sim['Vp'], sim['Qout'], sim['drenado']
        self.grao = np.random.default_rng(41).normal(0, 1, (6, h, w)).astype(np.float32)
        self.borda_f = _campo1d(400, 6, 77)

    def em(self, t, arr):
        return float(np.interp(t, self.T, arr))

    def aplicar(self, img, t, impacto, larg_fio, camada=None, ondas_extra=()):
        R = self.em(t, self.R)
        if R < 0.5:
            return img
        Vp = self.em(t, self.Vp)
        area_cm2 = math.pi * R * R * 0.0121
        esp_mm = 10 * Vp / max(area_cm2, 1e-3)
        dens = float(np.clip(esp_mm / 2.2, 0.25, 1.0))                              # oleo mais fino = mais claro
        borda = np.ones_like(self.th)
        for ph, a, n in self.lob:
            borda += a * np.sin(n * self.th + ph + 0.12 * n * math.sin(0.35 * t + ph))
        Rth = R * borda
        alfa = np.clip((Rth - self.rho) / 1.8, 0, 1) * (1 - self.cuba)
        rn = self.rho / np.maximum(Rth, 1)
        esp = np.clip(1 - rn ** 2, 0, 1)
        g = self.base
        ix, iy = impacto if impacto is not None else (self.cx, self.cy)
        rr = np.hypot(self.X - ix, (self.Y - iy) / self.achat)
        irreg = 1.6 * np.sin(3 * np.arctan2(self.Y - iy, self.X - ix) + 0.7 * t)
        forte = 1.0 if impacto is not None else 0.0
        ond = np.cos(2 * math.pi * (rr / 8.0 - 2.4 * t) + irreg) * np.exp(-rr / 10.0) * 0.6 * forte
        for (tx, ty, t0) in ondas_extra:                                            # gota pingando: anel que se abre
            dtt = t - t0
            if 0 <= dtt < 0.8:
                r2 = np.hypot(self.X - tx, (self.Y - ty) / self.achat)
                ond = ond + 0.9 * np.cos(2 * math.pi * (r2 - 55 * dtt) / 7.0) * np.exp(-((r2 - 55 * dtt) / 6.0) ** 2) * (1 - dtt / 0.8)
        ond = ond[..., None]
        cor = g * np.array([0.80, 0.62, 0.34]) + np.array([255, 160, 20]) * (0.06 + 0.14 * esp * dens)[..., None]
        tras = np.clip(-self.dy / np.maximum(Rth, 1), 0, 1)[..., None]
        fres = 0.16 + 0.12 * tras
        cor = cor * (1 - fres) + np.array([244, 216, 174]) * fres * (1 + 0.35 * ond)
        if camada is not None and impacto is not None:
            H, W = camada.shape[:2]
            src_y = np.clip(iy - (self.Y - iy) * 2.6 - 2, 0, H - 1).astype(np.float32)
            src_x = (self.X + 0.8 * np.sin(2 * math.pi * (self.Y / 6.0 - 1.5 * t))).astype(np.float32)
            refl = cv2.GaussianBlur(cv2.remap(camada, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), (0, 0), 0.8)
            kk = (0.55 * (self.Y > iy + 1))[..., None] * np.exp(-np.maximum(self.Y - iy, 0) / 30.0)[..., None]
            cor = cor * (1 - refl[..., 3:4] / 255 * kk) + refl[..., :3] * kk
        menisco = np.exp(-((self.rho - Rth) / 1.1) ** 2)[..., None]
        cor = cor + menisco * (0.12 + 0.40 * tras) * np.array([255, 232, 190]) * 0.5
        if impacto is not None:
            ra = np.hypot(self.X - ix, (self.Y - iy) / 0.42 / self.achat * self.achat)
            anel = (np.exp(-((ra - 0.9 * larg_fio) / 1.6) ** 2) * (0.8 + 0.2 * math.sin(2 * math.pi * 3.1 * t)))[..., None]
            cor = cor * (1 - 0.25 * anel) + anel * np.array([235, 175, 50]) * 0.25
        cor = cv2.GaussianBlur(cor.astype(np.float32), (0, 0), 0.7)                 # mesmo desfoque do granito
        cor = cor + self.grao[int(t * 30) % 6][..., None] * 1.8
        a3 = alfa[..., None]
        out = g * (1 - a3) + cor * a3
        # oleo entrando na cuba: pelicula na borda e filete no fundo, crescendo com o volume escoado
        dren = self.em(t, self.dren); qo = self.em(t, self.Qo)
        if dren > 0.05 and self.canto is not None:
            # pelicula de oleo entrando pelo canto da cuba e se abrindo no fundo, rente a borda da frente
            Cx, Cy = self.canto; dxu, dyu = self.dirf
            al_ = (self.X - Cx) * dxu + (self.Y - Cy) * dyu                  # ao longo do fundo
            tr_ = -(self.X - Cx) * dyu + (self.Y - Cy) * dxu                 # atravessado (negativo = para dentro da cuba)
            comp = 10 + 230 * (1 - math.exp(-dren / 40.0))
            qn = math.sqrt(min(qo, 40) / 40)
            irr = np.interp(al_, np.arange(400) - 20, self.borda_f, left=0, right=0)        # margem irregular, muda devagar
            larg = (2.5 + (4 + 8 * qn) * np.clip(1 - al_ / max(comp, 1), 0, 1) ** 0.7) * (1 + 0.30 * irr * math.cos(0.7 * t) + 0.15 * irr)
            dentro = np.clip((larg + tr_) / 1.5, 0, 1) * np.clip((1.0 - tr_) / 1.0, 0, 1)
            fil = dentro * np.clip(al_ / 3 + 1, 0, 1) * np.clip((comp - al_) / 10, 0, 1) * self.cuba
            fil = fil * (0.55 + 0.45 * np.clip(1 - al_ / max(comp, 1), 0, 1))
            reflexo = np.exp(-((tr_ + 0.45 * larg) / 0.9) ** 2) * fil * (0.6 + 0.4 * np.sin(al_ / 9 - 5 * t))
            cor_f = out * np.array([0.80, 0.60, 0.32]) + np.array([255, 165, 35]) * 0.12 + reflexo[..., None] * np.array([255, 238, 200]) * 0.45
            out = out * (1 - fil[..., None] * 0.8) + cor_f * fil[..., None] * 0.8
            # oleo passando por cima da borda no canto
            px_, py_ = self.pt_ponte
            ponte = np.exp(-(np.hypot(self.X - px_, (self.Y - py_) * 1.3) / (4.0 + 3.0 * qn)) ** 2)
            out = out * (1 - 0.55 * ponte[..., None]) + (g * np.array([0.8, 0.62, 0.34]) + np.array([250, 175, 50]) * 0.3) * 0.55 * ponte[..., None]
        img[self.y0:self.y1, self.x0:self.x1] = out
        return img
