#!/usr/bin/env python3
"""Desenho de som PROVISORIO de cena com liquido (sintetico, dirigido pela simulacao e pelo vento).
Stems: ambiente (sala + folhas + passaros distantes) e foley (liquido no tampo, bolhas, filete na cuba, gotas).
Normaliza: mix a -21 LUFS (para mixar sob a narracao) e mix_reels a -14 LUFS / -1 dBTP; stems com o mesmo ganho do mix.
Uso: python3 -I som_cena.py --cfg c.json --dir D --eventos v_eventos.json"""
import argparse, json, math, os, subprocess, sys
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ambiente as AMB

SR = 48000
rng = np.random.default_rng(2026)


def rosa(n):
    """Ruido rosa (1/f) por filtragem de ruido branco (Voss-McCartney aproximado por IIR)."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, rng.normal(0, 1, n))


def banda(x, lo, hi, ordem=4):
    sos = signal.butter(ordem, [lo, hi], btype='band', fs=SR, output='sos'); return signal.sosfilt(sos, x)


def passa_baixa(x, fc, ordem=4):
    sos = signal.butter(ordem, fc, btype='low', fs=SR, output='sos'); return signal.sosfilt(sos, x)


def env(t_audio, T, v):
    return np.interp(t_audio, T, v, left=0, right=0)


def reverb_curta(x, rt=0.35, mix=0.25):
    n = int(rt * SR); ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (rt * SR / 6.9))
    ir = passa_baixa(ir, 4000); ir /= np.sqrt((ir ** 2).sum())
    return (1 - mix) * x + mix * signal.fftconvolve(x, ir)[:len(x)]


def main(sim_p, ev_p, saida, dur, cfg):
    S = np.load(sim_p); ev = json.load(open(ev_p))
    n = int(dur * SR); ta = np.arange(n) / SR
    T = S['T']
    # ---------- ambiente
    sala = passa_baixa(rosa(n), 900) * 0.006                                   # ar da cozinha
    sala += 0.0012 * np.sin(2 * np.pi * 120 * ta) * (1 + 0.2 * np.sin(2 * np.pi * 0.3 * ta))   # zumbido leve (rede 60 Hz)
    vento = AMB.Vento()
    fz = np.array([vento.forca(t) for t in np.arange(0, dur + 0.05, 0.02)])
    folhas = banda(rosa(n), 1800, 7000) * np.interp(ta, np.arange(len(fz)) * 0.02, fz) ** 1.6 * 0.010
    folhas = passa_baixa(folhas, 5000)                                         # vindo de fora, pela janela
    passaros = np.zeros(n)
    for t0, f0, k in cfg['som']['passaros']:                                    # canto curto, distante
        for j in range(k):
            ti = t0 + j * rng.uniform(0.11, 0.17); ln = rng.uniform(0.05, 0.09)
            i0 = int(ti * SR); m = int(ln * SR); tt = np.arange(m) / SR
            f = f0 * (1 + 0.25 * np.sin(np.pi * tt / ln)) * (1 + 0.04 * np.sin(2 * np.pi * 38 * tt))
            ph = 2 * np.pi * np.cumsum(f) / SR
            e = np.sin(np.pi * tt / ln) ** 2
            if i0 + m < n:
                passaros[i0:i0 + m] += (np.sin(ph) + 0.25 * np.sin(2 * ph)) * e * 0.006
    passaros = reverb_curta(passa_baixa(passaros, 6000), 0.6, 0.45)
    # ---------- foley do oleo
    Qa = env(ta, T, S['Qarr'])                                                 # vazao que chega ao tampo (mL/s)
    qn = np.sqrt(np.clip(Qa / 40.0, 0, 1.4))
    mod = 1 + 0.25 * signal.sosfilt(signal.butter(2, 12, fs=SR, output='sos'), rng.normal(0, 1, n)) * 6
    chiado = banda(rosa(n), 240, 1800) * qn * np.clip(mod, 0.5, 1.6) * 0.030  # fio batendo na poca: chiado macio e grave
    bolhas = np.zeros(n)
    taxa = 11 * qn                                                             # bolhas/s (oleo viscoso: poucas e graves)
    t = 0.0
    while t < dur:
        r = float(np.interp(t, ta[::480], taxa[::480]))
        if r < 0.3:
            t += 0.02; continue
        t += rng.exponential(1 / r)
        f0 = rng.uniform(210, 680); tau = rng.uniform(0.014, 0.040); m = int(6 * tau * SR)
        i0 = int(t * SR)
        if i0 + m >= n:
            break
        tt = np.arange(m) / SR
        bolhas[i0:i0 + m] += np.sin(2 * np.pi * f0 * (1 + 0.25 * tt / tau) * tt) * np.exp(-tt / tau) * rng.uniform(0.3, 1.0) * 0.010
    Qo = env(ta, T, S['Qout'])
    filete = banda(rosa(n), 1300, 5200) * np.sqrt(np.clip(Qo / 40.0, 0, 1)) * 0.012
    for f_m, q_m in ((720, 18), (1330, 22), (2150, 25)):                        # cuba de inox ressoa um pouco
        b, a_ = signal.iirpeak(f_m, q_m, fs=SR); filete += signal.lfilter(b, a_, filete) * 0.6

    def plip(tq, ganho):
        m = int(0.12 * SR); tt = np.arange(m) / SR; i0 = int(tq * SR)
        if i0 + m >= n:
            return np.zeros(0), 0
        s = (np.sin(2 * np.pi * 420 * tt * (1 - 0.3 * tt / 0.12)) * np.exp(-tt / 0.022) +
             0.3 * banda(rng.normal(0, 1, m), 500, 2200) * np.exp(-tt / 0.006))
        return s * ganho, i0
    pings = np.zeros(n)
    s_, i0 = plip(float(ev['t_impacto']), 0.05)                                 # primeiro toque do oleo no tampo
    pings[i0:i0 + len(s_)] += s_
    for tg in ev.get('gotas_pouso', []):
        s_, i0 = plip(tg, 0.035)
        pings[i0:i0 + len(s_)] += s_
    foley = reverb_curta(chiado + passa_baixa(bolhas, 1800) + filete + pings, 0.3, 0.18)
    # estereo: oleo um pouco a esquerda (x ~ 393/1080), passaros e folhas a direita (janela), sala larga
    def pan(x, p):
        return np.stack([x * math.cos((p + 1) * math.pi / 4), x * math.sin((p + 1) * math.pi / 4)], 1)
    sala_st = np.stack([sala, passa_baixa(rosa(n), 900) * 0.006 + 0.0012 * np.sin(2 * np.pi * 120 * ta)], 1)
    pj = cfg['som']['pan_janela']
    amb_st = sala_st + pan(folhas, pj) + pan(passaros, pj + 0.05)
    foley_st = pan(foley, cfg['som']['pan_liquido'])
    fade = np.clip(np.minimum(ta / 0.25, (dur - ta) / 0.35), 0, 1)[:, None]
    os.makedirs(saida, exist_ok=True)
    from scipy.io import wavfile
    for nome, x in (('ambiente', amb_st * fade), ('foley_oleo', foley_st * fade), ('mix', (amb_st + foley_st) * fade)):
        wavfile.write(os.path.join(saida, f'{nome}.wav'), SR, np.clip(x, -1, 1).astype(np.float32))
    print('stems em', saida, '| pico mix', round(20 * math.log10(np.abs(amb_st + foley_st).max() + 1e-9), 1), 'dBFS')


def integrada(arq):
    e = subprocess.run(['ffmpeg', '-hide_banner', '-i', arq, '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True).stderr
    return float(e[e.rfind('Summary:'):].split('I:')[1].split('LUFS')[0])


def normalizar(ent, sai, alvo, tp):
    j = subprocess.run(['ffmpeg', '-hide_banner', '-i', ent, '-af', f'loudnorm=I={alvo}:TP={tp}:LRA=11:print_format=json',
                        '-f', 'null', '-'], capture_output=True, text=True).stderr
    m = json.loads(j[j.rfind('{'):j.rfind('}') + 1])
    filtro = (f"loudnorm=I={alvo}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
              f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
              f"aresample=48000,alimiter=limit={10 ** ((tp - 0.4) / 20):.3f}:attack=1:release=50:level=false")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', ent, '-af', filtro, '-c:a', 'pcm_s24le', sai], check=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cfg', required=True); ap.add_argument('--dir', required=True)
    ap.add_argument('--eventos', required=True)
    a = ap.parse_args(); cfg = json.load(open(a.cfg)); out = f'{a.dir}/som'
    main(f'{a.dir}/simulacao.npz', a.eventos, out, cfg['duracao'], cfg)
    normalizar(f'{out}/mix.wav', f'{out}/mix_-21LUFS.wav', -21, -2)
    normalizar(f'{out}/mix.wav', f'{out}/mix_reels.wav', -14, -1)
    ganho = integrada(f'{out}/mix_-21LUFS.wav') - integrada(f'{out}/mix.wav')       # mesmo ganho nos stems: preserva o equilibrio
    for nome in ('ambiente', 'foley_oleo'):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{out}/{nome}.wav', '-af',
                        f'volume={ganho:.2f}dB,alimiter=limit=0.76:attack=1:release=50:level=false',
                        '-c:a', 'pcm_s24le', f'{out}/{nome}_stem.wav'], check=True)
    print('som normalizado em', out)
