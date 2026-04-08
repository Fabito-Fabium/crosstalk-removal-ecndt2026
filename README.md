# Modeling and Removal of Electronic Crosstalk Effects in Ultrasound Signals by Deconvolution (ENCDT 2026)

by 
[Fabio Z. Y. Wang](https://orcid.org/0009-0008-9170-2687),
[Tatiana de A. Prado](https://orcid.org/0000-0002-4876-2974),
[Thiago E. Kalid](https://orcid.org/0000-0002-2035-5349),
[Gustavo P. Pires](https://orcid.org/0009-0008-3474-6077),
[Glauber Brante](https://orcid.org/0000-0001-6006-4274),
[Daniel R. Pipa](https://orcid.org/0000-0002-9398-332X),
[Thiago A. R. Passarin](https://orcid.org/0000-0003-1001-5911),.

 <br>
This repository contains the data and source code used to produce the results presented in:

> PLACEHOLDER
 
<br>
 
|                              | Info |
|------------------------------|------|
| Version of record            |   [``]()   |
| Open-access preprint |   [``]()   | 
| Archive of this repository   |   [`https:/doi.org/10.5281/zenodo.19410832`](https:/doi.org/10.5281/zenodo.19410832)   | 
| Reproducing our results | [`REPRODUCING.md`](REPRODUCING.md) |

## Abstract

The performance of an ultrasound array transducer is typically impaired by element-to-element interference, also known as crosstalk. The various approaches for reducing crosstalk generally involve modifying the transducer design or controlling the excitation waveforms. In this paper, we propose a method for modeling and removing electronic reception crosstalk in post-processing, which can serve as an alternative, or even a complement, to existing approaches. The modeling consists of estimating $N\cdot(N-1)$ Finite Impulse Responses (FIRs) that represent the effect that each $(N-1)$-th channel of the Phased Array (PA) has on the other channels. To obtain this modeling, we have performed controlled acquisitions that capture supposedly ideal signals (free from crosstalk) on one channel at a time, physically blocking the remainder channels. As a consequence, blocked channels will only capture electronic crosstalk. After the modeling stage, crosstalk removal from arbitrary datasets acquired with the modeled PA is performed via a Tikhonov-regularized inverse problem using the estimated FIRs. Experimental results show that the proposed method is able to improve the Signal-to-Interference-plus-Noise Ratio (SINR) by up to 10 dB.

## License
All Python source code (`.py`) is made available
under the MIT license. You can freely use and modify the code, without
warranty, so long as you provide attribution to the authors. See
`LICENSE-MIT.txt` for the full license text.

Figures and data
produced as part of this research are available under the [Creative Commons
Attribution 4.0 License (CC-BY)][cc-by]. See `LICENSE-CC-BY.txt` for the full
license text.

[cc-by]: https://creativecommons.org/licenses/by/4.0/


