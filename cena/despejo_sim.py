#!/usr/bin/env python3
"""Simulacao do despejo (pre-calculada, antes do render):
inclinacao do recipiente (uma 'mao' que controla o fio pelo olho) -> volume e nivel (modelo 3D) -> vazao pelo bico
(vertedouro, Q = Cw * carga^1.5, calibrada pelo fio da imagem) -> queda -> poca no tampo (corrente gravitacional
viscosa, Huppert 1982; area molhada nao recua) -> transbordo para a cuba (poligono do cfg).
Uso: python3 -I despejo_sim.py --cfg c.json --dir D      (le D/calib.json, grava D/simulacao.npz)"""
import argparse, json, math, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipiente3d as R3

G_PX = 8900.0                     # px/s2 (1 px ~ 1,1 mm)
V0_PX = 230.0                     # velocidade de saida pelo bico (px/s)
MM = R3.MM_PX / 1000.0


def s5(x):
    x = min(max(x, 0.0), 1.0); return x * x * x * (x * (6 * x - 15) + 10)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cfg', required=True); ap.add_argument('--dir', required=True)
    a = ap.parse_args(); cfg = json.load(open(a.cfg)); rc, dp, po = cfg['recipiente'], cfg['despejo'], cfg['pouso']
    W, H = cfg['quadro']; DUR = cfg['duracao']; FPS_SIM = dp['fps_sim']
    pan = R3.Panela(Rr=rc['Rr'], Rf=rc['Rf'], Dp=rc['Dp'], E=rc['camera_elevacao'])
    cal = json.load(open(f'{a.dir}/calib.json')); b0, h0 = cal['beta0'], cal['h0']
    carga_img = h0 - pan.y_bico(b0)
    Q_IMG = dp['q_img']
    CW = Q_IMG / max(carga_img, 0.5) ** 1.5
    V_INI = dp['v_ini_ml']
    n = int(DUR * FPS_SIM); T = np.arange(n) / FPS_SIM

    def ruido_lento(t, sem):
        r = np.random.default_rng(sem); ph = r.uniform(0, 2 * np.pi, 3)
        return float(np.mean([np.sin(2 * np.pi * f * t + p) for f, p in zip((0.31, 0.53, 0.89), ph)]))

    def alvo(t):
        return dp['vazao_alvo'] * (1 + 0.12 * ruido_lento(t, 21))

    t0, tl, bmax = dp['t_comeca'], dp['t_levanta'], dp['beta_max']
    V = V_INI / R3.ML_PX3; b = dp['beta_ini']; db = 0.0; Bc = np.zeros(n)
    for i, t in enumerate(T):
        br = math.radians(b); h = pan.altura(V, br); carga = h - pan.y_bico(br)
        Q = CW * max(carga - 0.15, 0.0) ** 1.5
        if t < t0:
            alvo_db = 0.0
        elif t < tl:
            H_alvo = (alvo(t) / CW) ** (2 / 3) + 0.15
            alvo_db = float(np.clip(14.0 * (H_alvo - carga), -6.0, 9.0)) * s5((t - t0) / 0.5)
            if b >= bmax:
                alvo_db = min(alvo_db, 0.0)
        else:
            alvo_db = -8.0 * s5((t - tl) / 0.4)
        db += (alvo_db - db) * min(1.0, 3.5 / FPS_SIM * 4)
        b += db / FPS_SIM
        V -= min(Q / R3.ML_PX3 / FPS_SIM, V)
        Bc[i] = b
    k = cv2.getGaussianKernel(int(0.5 * FPS_SIM) | 1, 0.10 * FPS_SIM).ravel()
    Bs = np.convolve(np.pad(Bc, (len(k) // 2, len(k) // 2), mode='edge'), k, mode='valid')
    V = V_INI / R3.ML_PX3
    B, Vt, Ht, Qt = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i, t in enumerate(T):
        br = math.radians(Bs[i]); h = pan.altura(V, br); carga = h - pan.y_bico(br)
        Q = CW * max(carga - 0.15, 0.0) ** 1.5 if V > 0 else 0.0
        dV = min(Q / R3.ML_PX3 / FPS_SIM, V); V -= dV
        B[i], Vt[i], Ht[i], Qt[i] = br, V * R3.ML_PX3, h, dV * R3.ML_PX3 * FPS_SIM
    POUSO = po['y']; labio_y = cfg['fio']['labio'][1]
    t_queda = (-V0_PX + math.sqrt(V0_PX ** 2 + 2 * G_PX * (POUSO - labio_y))) / G_PX
    atraso = int(round(t_queda * FPS_SIM))
    Qarr = np.concatenate([np.zeros(atraso), Qt[:-atraso]])
    i_imp = int(np.argmax(Qarr > 0.5)); t_imp = T[i_imp]
    bacia = np.zeros((H, W), np.uint8)
    if cfg.get('cuba', {}).get('poligono'):
        cv2.fillPoly(bacia, [np.array(cfg['cuba']['poligono'], np.int32)], 1)
    bacia = bacia.astype(bool)
    cx, cy = po['centro_poca']; ACH = po['achat']; NU = po['nu']; KAPPA = po['kappa']
    sx0, sy0, sx1, sy1 = po['caixa_sobreposicao']
    yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
    Vp, Rt, Vpt, Qout, Vdren = 0.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    drenado = 0.0; R = 0.0
    for i, t in enumerate(T):
        Vp += Qarr[i] / FPS_SIM
        tau = max(t - t_imp, 1e-3)
        if Vp > 0:
            R = max(R, 0.8 * (9.81 / (3 * NU)) ** 0.125 * (Vp * 1e-6) ** 0.375 * tau ** 0.125 / MM)
        sob = 0.0
        if R > 40:
            el = ((xx - cx) / R) ** 2 + ((yy - cy) / (R * ACH)) ** 2 <= 1
            sob = float((el & bacia[sy0:sy1, sx0:sx1]).sum())
        residuo = math.pi * R * R * (MM * 100) ** 2 * 0.08
        q_out = min(KAPPA * sob ** 0.75, max(Vp - residuo, 0.0) * FPS_SIM)
        Vp -= q_out / FPS_SIM; drenado += q_out / FPS_SIM
        Rt[i], Vpt[i], Qout[i], Vdren[i] = R, Vp, q_out, drenado
    np.savez(f'{a.dir}/simulacao.npz', T=T, beta=B, V=Vt, h=Ht, Q=Qt, Qarr=Qarr, R=Rt, Vp=Vpt, Qout=Qout,
             drenado=Vdren, t_imp=t_imp, t_queda=t_queda, cw=CW, q_img=Q_IMG)
    on = T[np.argmax(Qt > 0.5)]; fim = T[len(Qt) - 1 - np.argmax(Qt[::-1] > 0.5)]
    i1 = np.argmax(Qout > 0.5)
    print(f'despejo de {on:.2f} a {fim:.2f} s, pico {Qt.max():.0f} mL/s; {V_INI:.0f} mL -> sobram {Vt[-1]:.0f} mL; '
          f'beta {math.degrees(B.min()):.1f} a {math.degrees(B.max()):.1f} graus')
    print(f'impacto em {t_imp:.2f} s; poca chega a cuba em {T[i1]:.2f} s; raio final {Rt[-1]:.0f} px; '
          f'ficam {Vpt[-1]:.0f} mL no tampo; {Vdren[-1]:.0f} mL vao para a cuba')


if __name__ == '__main__':
    main()
