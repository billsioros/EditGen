# EditGen: Harnessing Cross Attention Control for Instruction-Based auto-regressive Audio Editing

<p align="center">
  <img src="https://github.com/billsioros/EditGen/blob/master/docs/EditGen.png?raw=true" alt="EditGen"/>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2507.11096">
        <img src="https://img.shields.io/badge/arXiv-2507.11096-b31b1b.svg" alt="arXiv" />
    </a>
    <a href="https://github.com/google/prompt-to-prompt/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License" />
    </a>
    <a href="https://www.python.org/downloads/release/python-3921/">
        <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" />
    </a>
</p>

> Accompanying code for the paper [**EditGen: Harnessing Cross Attention Control for Instruction-Based auto-regressive Audio Editing**](https://arxiv.org/abs/2507.11096)

In this study, we investigate leveraging cross-attention control for efficient audio editing within auto-regressive models. Inspired by image editing methodologies, we develop a Prompt-to-Prompt-like approach that guides edits through cross and self-attention mechanisms. Integrating a diffusion-based strategy, influenced by Auffusion, we extend the model's functionality to support refinement edits, establishing a baseline for prompt-guided audio editing. Additionally, we introduce an alternative approach by incorporating MUSICGEN, a pre-trained frozen auto-regressive model, and propose three editing mechanisms, based on Replacement, Reweighting, and Refinement of the attention scores. We employ commonly-used music-specific evaluation metrics and a human study, to gauge time-varying controllability, adherence to global text cues, and overall audio realism. The automatic and human evaluations indicate that the proposed combination of prompt-to-prompt guidance with autoregressive generation models significantly outperforms the diffusion-based baseline in terms of melody, dynamics, and tempo of the generated audio.

<p align="center">
  <img src="https://github.com/billsioros/EditGen/blob/master/docs/EditGen.drawio.png?raw=true" alt="EditGen"/>
</p>

## :scroll: Citation

```bibtex
@misc{EditGen,
  author = {Vassilis Sioros},
  title = {EditGen: Harnessing Cross Attention Control for Instruction-Based auto-regressive Audio Editing},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/billsioros/EditGen}}
}
```

> [!NOTE]
> The files under `src/auffusion` where taken directly from the [`Auffusion`](https://github.com/happylittlecat2333/Auffusion) project for the purposes of comparing the two models and fall under the [Apache 2.0 license](https://github.com/happylittlecat2333/Auffusion/blob/main/LICENSE).
