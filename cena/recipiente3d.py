"""Recipiente (frigideira, panela, balde) em 3D simplificado: tronco de cone (fundo plano + parede inclinada), camera ortografica
com elevacao E, frigideira inclinada BETA em direcao a camera (eixo = cabo, horizontal na tela).
Coordenadas locais (p, q, s): p = horizontal na tela, q = no plano da borda apontando para o bico (perto e para baixo),
s = normal da borda (para fora da frigideira). Origem = centro da borda. Unidades em px (1 px ~ 1,1 mm).
Altura no mundo:  y = -q sin(beta) + s cos(beta)
Tela (relativa ao centro da borda, y para baixo):  x = p ;  y = q sin(beta+E) - s cos(beta+E)"""
import math
import numpy as np

MM_PX = 1.1
ML_PX3 = (MM_PX / 10) ** 3          # mL por px^3


class Panela:
    def __init__(self, Rr=102.5, Rf=92.0, Dp=50.0, E=21.0, passo=2.0):
        self.Rr, self.Rf, self.Dp, self.E = Rr, Rf, Dp, math.radians(E)
        g = np.arange(-Rr, Rr + passo, passo); s = np.arange(-Dp + passo / 2, 0, passo)
        P, Q, S = np.meshgrid(g, g, s, indexing='ij')
        dentro = np.hypot(P, Q) <= self.rho(S)
        self.Q, self.S = Q[dentro], S[dentro]; self.dV = passo ** 3

    def rho(self, s):
        return self.Rf + (self.Rr - self.Rf) * (s + self.Dp) / self.Dp

    def y_bico(self, beta):
        return -self.Rr * math.sin(beta)

    def volume(self, h, beta):                                    # px^3 abaixo da altura h
        y = -self.Q * math.sin(beta) + self.S * math.cos(beta)
        return float((y < h).sum()) * self.dV

    def altura(self, V, beta):
        lo, hi = -self.Rr * math.sin(beta) - self.Dp * math.cos(beta) - 1, self.Rr * math.sin(beta) + 1
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.volume(mid, beta) < V:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def capacidade(self, beta):
        return self.volume(self.y_bico(beta), beta)

    def superficie(self, h, beta, X, Y):
        """Para pixels de tela (X, Y relativos ao centro da borda): mascara da superficie do oleo visivel,
        espessura aproximada (px) e coordenadas (p, q) no plano da frigideira."""
        E = self.E
        q = (Y * math.cos(beta) + h * math.cos(beta + E)) / math.sin(E)
        s = (h + q * math.sin(beta)) / math.cos(beta)
        r = np.hypot(X, q)
        dentro_cone = (s <= 0.5) & (s >= -self.Dp) & (r <= self.rho(np.clip(s, -self.Dp, 0)))
        abertura = (X / self.Rr) ** 2 + (Y / (self.Rr * math.sin(beta + E))) ** 2 <= 1.0
        m = dentro_cone & abertura
        # espessura: distancia vertical ate o fundo ou a parede (aprox.: menor folga radial ou ate o fundo)
        folga_r = (self.rho(np.clip(s, -self.Dp, 0)) - r) * math.tan(math.radians(60))
        folga_f = (s + self.Dp) / max(math.cos(beta), 1e-3)
        esp = np.where(m, np.clip(np.minimum(folga_r, folga_f), 0, None), 0)
        return m, esp, q, s
