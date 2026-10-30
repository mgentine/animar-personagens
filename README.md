# animar-personagens

Código da skill **animar-personagens** do Claude: marionete 2D por esqueleto para animar personagens ilustrados de imagens estáticas, com líquido despejado, ambiente (sombra de folhagem, planta, vento) e som provisório para cenas de Reels. Sem IA generativa: o personagem é recortado, o fundo é reconstruído e a imagem é deformada por uma malha.

As instruções de uso ficam na própria skill. Este repositório só guarda o código, numa versão marcada por tag e conferida por hash.

## Estrutura

| Caminho | O que é |
|---|---|
| `anima.py` | Recorte, fundo, esqueleto, malha, ações (acenar, apontar, falar, piscar, pular...) e render |
| `cena/prep_cena.py` | Preparo de cena com líquido: camadas, ajustes do esqueleto, recipiente sem líquido, calibração |
| `cena/recipiente3d.py` | Modelo 3D do recipiente (tronco de cone, câmera ortográfica) |
| `cena/despejo_sim.py` | Simulação do despejo: vazão pelo bico, queda, poça e escoamento |
| `cena/liquido_recipiente.py` | Líquido dentro do recipiente, com superfície horizontal |
| `cena/liquido_queda.py` | Fio, contas, gotas, poça e película de escoamento |
| `cena/ambiente.py` | Vento, sombra de folhagem e planta |
| `cena/cena_despejo.py` | Render da cena pelo arquivo de configuração |
| `cena/som_cena.py` | Som provisório sincronizado e normalização |
| `cena/previa_reels.py` | Prévia com selos e legenda de teste |
| `exemplos/cena_oleo.json` | Configuração da cena de referência (cozinheira despejando óleo) |
| `SHA256SUMS` | Hashes dos arquivos da versão |

## Instalação

```bash
git clone --depth 1 --branch v1.0.0 https://github.com/mgentine/animar-personagens.git ~/anima
cd ~/anima && sha256sum -c SHA256SUMS
```

Dependências e modelos: ver a seção de instalação da skill.

Testado em 30/10/2026 com Python 3.13, OpenCV 5.0, NumPy 2.5, mediapipe 0.10.32, onnxruntime 1.29, scikit-image 0.26, SciPy 1.18, moderngl sobre EGL/llvmpipe e ffmpeg 6.1.
