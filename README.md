# Attack MNIST Recognition

This experiment attacks the CNN selected for the earlier MNIST Recognition assignment. It classified **9,895 of 10,000 test images correctly (98.95%)** before attack. The main experiment compares FGSM, PGD, and MI-FGSM. A separate extension tests whether the same untargeted perturbations transfer between the CNN, an MLP, and a patch Transformer.

## Data and threat model

The CNN has 101,129 parameters. Its weights came from the prior seed-42 holdout experiment: 55,000 images for training and 5,000 for checkpoint selection from MNIST's official training split. The test split was not used for fitting or checkpoint selection, but the same public test split was used in the previous assignment. The training-only pixel mean and standard deviation are 0.1306704395 and 0.3081222811. `train.py` implements the winning CNN recipe (AdamW, learning rate 0.003, weight decay 0, 5 epochs, batch size 256), and `cnn-reference.pt` contains the exact inherited weights used for these results, repackaged with the normalization constants. The [original training protocol and search](https://github.com/KStone1704/ai-hw-fall-2026-nn/tree/8c88686ec84aabb67f6f15ef60f6b37f717b8103/matched_hpo) records the six-trial CNN selection. No model was fine-tuned after attack evaluation.

The main experiment uses **untargeted, white-box, digital** attacks: each attack increases cross-entropy on the true label with respect to the input while model weights stay fixed. Any wrong prediction is a success; no wrong class is specified. The L-infinity budget applies to raw `[0,1]` pixels; the model normalizes internally. Every new pixel is constrained to `[0,1]` and within `epsilon` of its original value.

- **FGSM:** one sign-gradient step of size `epsilon`.
- **PGD (iterative FGSM variant):** three seeded random starts, ten projected sign steps per start at `epsilon/4`. A test item counts as correct only if it survives **all three** starts; any successful restart counts as an attack success.
- **MI-FGSM:** one clean start, ten projected steps at `epsilon/4`, momentum decay 1, and L1-normalized gradients per image.

## Full test results

Attack success rate (ASR) is the fraction of the **9,895 originally correct** images fooled by the specified attack. Adversarial accuracy uses **all 10,000** images. Counts and maximum observed perturbation for every setting are in [`results.json`](results.json).

| Epsilon | Method | Correct after attack | Adversarial accuracy | ASR |
| ---: | --- | ---: | ---: | ---: |
| 0.05 | FGSM | 9,425 | 94.25% | 4.75% |
| 0.05 | PGD (3 starts) | 9,177 | 91.77% | 7.26% |
| 0.05 | MI-FGSM | 9,185 | 91.85% | 7.18% |
| 0.15 | FGSM | 5,981 | 59.81% | 39.56% |
| 0.15 | PGD (3 starts) | 646 | 6.46% | 93.47% |
| 0.15 | MI-FGSM | 946 | 9.46% | 90.44% |
| 0.30 | FGSM | 1,036 | 10.36% | 89.53% |
| 0.30 | PGD (3 starts) | 0 | 0.00% | 100.00% |
| 0.30 | MI-FGSM | 0 | 0.00% | 100.00% |

At epsilon 0.15, reporting FGSM alone would overstate this model's robustness against the iterative attacks. PGD used a larger gradient budget than MI-FGSM because of its three restarts, so this comparison does not isolate momentum. A 0.30 maximum change is substantial on MNIST, and a norm bound does not establish perceptual similarity.

## Extension: does the attack transfer?

I used three fixed, validation-selected models from the earlier MNIST experiment and the **9,574 test images all three originally classified correctly**. This common denominator differs from the CNN-only denominator above. Each source generated its perturbations using only its own gradients; the other models scored the resulting images afterwards. I-FGSM and MI-FGSM both used 10 steps from the clean image, step size `epsilon/4`, and no restarts. They differ only in momentum (0 vs. 1). I tested raw pixel budgets 0.05 and 0.15 and a seeded random-sign control at the same budget. [`transfer/results.json`](transfer/results.json) records both methods and budgets, counts, maximum perturbations, checkpoint hashes, and the following diagnostic results.

MI-FGSM at epsilon 0.15, percentage of the common set misclassified:

| Perturbation source | MLP target | CNN target | Transformer target |
| --- | ---: | ---: | ---: |
| MLP | **97.57%** | 2.55% | 42.46% |
| CNN | 6.31% | **90.12%** | 34.93% |
| Transformer | 8.78% | 1.54% | **99.21%** |
| Random sign, clipped to `[0,1]` | 0.89% | 0.21% | 25.49% |

The diagonal is white-box; the off-diagonal measures transfer without target queries during generation. The CNN/Transformer clean input loss gradients had mean cosine similarity **0.075**, yet transfer was asymmetric. For the 3,263 CNN-generated examples that fooled both models, 2,691 (82.5%) produced *different* wrong labels. At epsilon 0.05, CNN → Transformer fell to 2.31% (random control: 1.30%). At epsilon 0.15, momentum raised that transfer rate from 32.69% to 34.93% while reducing CNN white-box ASR on the common set from 91.95% to 90.12%.

### Why did the Transformer also fail under random noise?

About 81% of clean pixels were exactly zero. Clipping random ±0.15 noise to `[0,1]` removes negative changes at those pixels but keeps positive ones, so the control systematically brightens them. I separated the components of the existing CNN attack, without reoptimizing either component:

| Input at epsilon 0.15 | MLP error | CNN error | Transformer error |
| --- | ---: | ---: | ---: |
| Complete CNN attack | 6.31% | 90.12% | 34.93% |
| Keep the changes only on original zero pixels | 2.28% | 23.54% | 27.19% |
| Keep the changes only on original nonzero pixels | 1.60% | 12.16% | 2.16% |
| No attack: set original zero pixels uniformly to 0.075 | 0.68% | 0.07% | 22.10% |

This trained Transformer was particularly sensitive to brightening originally black pixels. That sensitivity is not proof that it extrapolates worse in all off-manifold directions. The study does not estimate distance to a data manifold, match perturbation energy, or isolate architecture from training. The results apply to these checkpoints, images, and budgets; they are not robustness guarantees.

## Reproduce

Use Python with `torch` and `torchvision` installed (`python -m pip install -r requirements.txt`). Results in this repository were generated on CPU with PyTorch 2.14.1 and TorchVision 0.29.1; numerical details may vary across versions and devices. From the repository root:

```bash
python evaluate.py --data data --download --output results-reproduced.json
python -m unittest -v test_attacks
python train.py --data data --download --output trained-cnn.pt
python evaluate.py --data data --checkpoint trained-cnn.pt --output results-retrained.json
python transfer/run_transfer.py --data data --download
```

The first evaluation reuses the exact recorded checkpoint. Training reproduces the recipe, not necessarily bit-identical weights across hardware. The transfer command uses the three original weights in `transfer/checkpoints/` and writes `transfer/results-reproduced.json`. The CNN in both studies has the same weights; the root checkpoint merely repackages the original. The earlier model selection and training code are linked above. No raw data or training cache is included.

References: [MNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html), [FGSM](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html), [Madry et al., PGD](https://arxiv.org/abs/1706.06083), [Dong et al., MI-FGSM](https://arxiv.org/abs/1710.06081), [Demontis et al., transferability](https://www.usenix.org/conference/usenixsecurity19/presentation/demontis), and [Stutz et al., on- and off-manifold attacks](https://openaccess.thecvf.com/content_CVPR_2019/html/Stutz_Disentangling_Adversarial_Robustness_and_Generalization_CVPR_2019_paper.html).

This project was completed with assistance from OpenAI Codex.
