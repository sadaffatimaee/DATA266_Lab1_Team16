# Task 3: CycleGAN photo to Monet style transfer

Poushali Purkayastha, Team 16

## What was built

A CycleGAN with two generators and two discriminators, trained from scratch on unpaired photos and Monet paintings. Domain A is photos and domain B is Monet paintings, so G_AB turns a photo into a Monet-style image (the Kaggle direction) and G_BA turns a Monet into a photo-style image. Training uses least-squares adversarial loss, cycle-consistency loss, and identity loss. The code is in src/, split into data.py, models.py, train.py, translate.py, metrics.py, audit.py, and run.py. evaluate_local.py at the folder root recomputes every metric from saved outputs.

Final numbers are in full_metrics_report.csv. Sections marked "to fill" are completed from the final GPU run, the Kaggle submission, and the human audit.

## Data

- Source: the class Kaggle competition data, two flat folders of 256 x 256 JPEGs, monet_jpg and photo_jpg. The local pipeline test used the public monet2photo release from the CycleGAN authors, which has the same 7,038 photos and 1,193 Monets.
- Domains are unpaired. No test pairings exist and none are used.
- Holdout: 300 photos and 30 Monets, chosen with seed 20, are kept out of training and used only for cycle-consistency, identity, LPIPS, and content-preservation metrics and for the human audit inputs. The file names are in data_processed/full/holdout.json.
- Training resolution 128 x 128 (bicubic resize, cached as uint8 arrays that are not committed). Inference at 256 x 256, which the fully convolutional generator supports directly, so the Kaggle submission and pred_A2B are at the competition's native size.
- Augmentation: random horizontal flip per image.

## Architecture and why

| Component | Choice | Why |
| --- | --- | --- |
| Generators G_AB and G_BA | ResNet generator: 7x7 conv to 64 channels, two stride-2 downsampling convs to 256 channels, 6 residual blocks, two transposed convs back up, 7x7 conv to RGB with tanh; reflection padding; instance normalization | the CycleGAN paper's 128 px generator; residual blocks keep the layout of the input while the style changes, instance norm suits per-image style transfer |
| Discriminators D_A and D_B | 70x70 PatchGAN: four stride-2 and stride-1 4x4 convs from 64 to 512 channels, LeakyReLU 0.2, instance norm, one-channel patch output | judges local texture patches instead of the whole image, which is what brush-stroke style lives in; fewer parameters than a full-image discriminator |
| Adversarial loss | least-squares (LSGAN) | more stable gradients than the sigmoid cross-entropy loss and less mode collapse in the paper's ablations |
| Cycle-consistency loss | L1 between the input and its reconstruction in both directions, weight 10 | the constraint that makes unpaired training work: content must survive a round trip |
| Identity loss | L1 between G_AB(monet) and monet and between G_BA(photo) and photo, weight 5 | keeps colour composition when the input already belongs to the target domain, which stops G_AB from tinting every photo |
| Image pool | 50 previously generated images per discriminator | discriminators see a history of fakes, which reduces oscillation |
| Initialization | normal(0, 0.02) on every conv | the paper's choice |

Parameter counts per network: to fill from full_metrics_report.csv.

## Hyperparameters and why

| Hyperparameter | Value | Why |
| --- | --- | --- |
| Training image size | 128 | keeps one lab-slot training run under 20 minutes while sharing the GPU |
| Batch size | 4 | instance norm works per image so a small batch is fine; 4 improves GPU utilization over the paper's 1 |
| Steps per epoch, epochs | 300, 24 | 7,200 generator updates, about 28,800 photos and 96 passes over the 300 Monets; sized from a throughput benchmark so the run fits a shared lab slot |
| Optimizer | Adam, learning rate 2e-4, beta1 0.5 | the paper's settings; beta1 0.5 is standard for GANs |
| Learning-rate schedule | constant for 12 epochs, then linear decay to zero at the end | the paper's schedule scaled to the epoch budget |
| lambda_cycle, lambda_identity | 10, 5 | the paper's ratio (identity at half the cycle weight) |
| Gradient clipping | 10 | mostly a measurement of gradient norms, only bites on a genuine spike |
| Mixed precision | on GPU, bfloat16 if supported else float16 with loss scaling; automatically disabled after 5 consecutive non-finite losses | speed, with a guard against fp16 instabilities |
| Seed | 20 | fixes the holdout, sampling, initialization, and the metric subsampling |

## How each metric is computed

| Metric | Computation |
| --- | --- |
| FID, both directions | Inception-v3 (torchvision ImageNet weights, 2048-d pool features at 299 px) on up to 2,000 real and 2,000 generated images per domain; Frechet distance between the two Gaussians |
| KID, both directions | unbiased MMD with a cubic polynomial kernel on the same features, mean and std over 100 subsets of 1,000 |
| Precision, recall, density, coverage | k-nearest-neighbour manifolds (k = 3) on the Inception features, following Kynkaanniemi et al. and Naeem et al. |
| Cycle-reconstruction L1 | mean absolute error between a holdout image and its round trip, in [0, 1] pixel scale, both directions |
| Identity L1 | mean absolute error between a holdout image and the generator of its own domain applied to it |
| LPIPS | AlexNet LPIPS between input and translation, and between input and reconstruction, both directions on the holdout images |
| Content-preservation cosine similarity | cosine between the Inception features of a holdout input and of its translation |
| Memorization distance and MiFID-like | mean over generated Monets of the minimum cosine distance to real Monet features; FID divided by that distance when it falls below epsilon 0.1, mirroring Kaggle's MiFID penalty |
| Loss curves, gradient norms, NaN count | logged every step during training; plots in outputs/full/loss_curves.png |
| Parameter count, training time, images/sec, peak memory | from the training run; images/sec counts both domains |
| Human audit | 30 fixed holdout photos, translated by each team member's model, shuffled and blinded; both raters score style, content, and artifacts from 1 to 5; Cohen's kappa, linear-weighted kappa, and percent agreement per criterion |
| Kaggle score and rank | from the leaderboard after uploading outputs/full/kaggle/images.zip |

## Results

To fill from the final run: the metrics table from full_metrics_report.csv, the loss curves and learning-rate plot, and the sample grids in outputs/full/samples/ (rows: real photo, fake Monet, reconstructed photo, real Monet, fake photo, reconstructed Monet).

## Visual quality and cycle-consistency verification

To fill: a short assessment of pred_A2B_preview and pred_B2A_preview, whether the reconstructions in the sample grids match their inputs, and what the cycle L1 and LPIPS reconstruction numbers say.

## Training stability and loss behaviour

To fill: convergence of generator and discriminator losses, any oscillation, gradient-norm spikes, NaN events, and whether mixed precision stayed on.

## Kaggle submission

To fill: submission date, public and private score, leaderboard rank. The submission is the unedited content of outputs/full/pred_A2B, produced by checkpoints/full/generators.pt.

## Human audit

To fill from outputs/full/audit/audit_results.json: mean style, content, and artifact scores per model and the inter-rater agreement.

## Shortcomings and future work

To fill.

## Hardware and run identity

To fill from the manifest: GPU model, run_id, git commit, raw log path, manifest path.

## How this model differs from Sadaf's

Per the team plan, Sadaf's CycleGAN uses a U-Net generator, a smaller PatchGAN, the sigmoid cross-entropy adversarial loss, no identity loss, and a different cycle weight. Mine uses the paper's ResNet generator, the 70x70 PatchGAN, least-squares loss, and identity loss. The comparison table in the report sets the two side by side.
