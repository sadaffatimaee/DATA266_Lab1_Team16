# Task 3: CycleGAN photo to Monet style transfer

Poushali Purkayastha, Team 16

## What was built

A CycleGAN with two generators and two discriminators, trained from scratch on unpaired photos and Monet paintings. Domain A is photos and domain B is Monet paintings, so G_AB turns a photo into a Monet-style image (the Kaggle direction) and G_BA turns a Monet into a photo-style image. Training uses least-squares adversarial loss, cycle-consistency loss, and identity loss. The code is in src/, split into data.py, models.py, train.py, translate.py, metrics.py, kaggle_eval.py, audit.py, and run.py. evaluate_local.py at the folder root recomputes every metric from saved outputs, and update_report_rows.py merges audit and Kaggle results into the report without recomputing.

Final numbers are in full_metrics_report.csv (metrics_report.csv is an identical copy) and come from run `full_20260928_180436` on an NVIDIA GeForce RTX 5090 in the GPU lab.

## Data

- Source: the class Kaggle competition data, downloaded with the Kaggle API on the lab machine: 300 Monet paintings and 7,038 photos, all 256 x 256 JPEGs. The pipeline was tested beforehand on the public monet2photo release from the CycleGAN authors, which has the same photos and a larger Monet set.
- Domains are unpaired. No test pairings exist and none are used.
- Holdout: 300 photos and 30 Monets, chosen with seed 20, are kept out of training and used only for the cycle-consistency, identity, LPIPS, and content-preservation metrics and for the human audit inputs. Training uses the remaining 6,738 photos and 270 Monets. File names are in data_processed/full/holdout.json.
- Training resolution 128 x 128 (bicubic resize, cached as uint8 arrays that are not committed). Inference at 256 x 256, which the fully convolutional generator supports directly, so the Kaggle submission and pred_A2B are at the competition's native size. Random horizontal flip per image is the only augmentation.

## Architecture and why

| Component | Choice | Why |
| --- | --- | --- |
| Generators G_AB and G_BA | ResNet generator: 7x7 conv to 64 channels, two stride-2 downsampling convs to 256 channels, 6 residual blocks, two transposed convs back up, 7x7 conv to RGB with tanh; reflection padding; instance normalization; 7,837,699 parameters each | the CycleGAN paper's 128 px generator; residual blocks keep the layout of the input while the style changes, instance norm suits per-image style transfer |
| Discriminators D_A and D_B | 70x70 PatchGAN: 4x4 convs from 64 to 512 channels with stride 2, 2, 1, LeakyReLU 0.2, instance norm, one-channel patch output; 2,764,737 parameters each | judges local texture patches instead of the whole image, which is what brush-stroke style lives in; fewer parameters than a full-image discriminator |
| Adversarial loss | least-squares (LSGAN) | more stable gradients than sigmoid cross-entropy and less mode collapse in the paper's ablations |
| Cycle-consistency loss | L1 between the input and its reconstruction in both directions, weight 10 | the constraint that makes unpaired training work: content must survive a round trip |
| Identity loss | L1 between G_AB(monet) and monet and between G_BA(photo) and photo, weight 5 | keeps colour composition when the input already belongs to the target domain, which stops G_AB from tinting every photo |
| Image pool | 50 previously generated images per discriminator | discriminators see a history of fakes, which reduces oscillation |
| Initialization | normal(0, 0.02) on every conv | the paper's choice |

Total parameters: 21,204,872.

## Hyperparameters and why

| Hyperparameter | Value | Why |
| --- | --- | --- |
| Training image size | 128 | keeps one training run short enough to share a lab slot; the RTX 5090 finished it in under 9 minutes |
| Batch size | 4 | instance norm works per image so a small batch is fine; 4 improves GPU utilization over the paper's 1 |
| Steps per epoch, epochs | 300, 24 | 7,200 generator updates, about 28,800 photos and 107 passes over the 270 training Monets |
| Optimizer | Adam, learning rate 2e-4, beta1 0.5 | the paper's settings; beta1 0.5 is standard for GANs |
| Learning-rate schedule | constant for 12 epochs, then linear decay to 1.5e-5 at epoch 24 | the paper's schedule scaled to the epoch budget |
| lambda_cycle, lambda_identity | 10, 5 | the paper's ratio, identity at half the cycle weight |
| Gradient clipping | 10 | mostly a measurement of gradient norms; it clipped the 9 steps whose norm exceeded 100 |
| Mixed precision | bfloat16 autocast, which stayed on for the whole run; automatic fallback after 5 consecutive non-finite losses never triggered | speed, with a guard against instabilities |
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
| Memorization distance | mean over generated Monets of the minimum cosine distance to real Monet features |
| Competition FID and MiFID | src/kaggle_eval.py on all 7,038 generated Monet-style images: FID against the competition's real_stats.npz mean and covariance, MiFID as the mean cosine distance between generated and real Inception features after subsampling both to 300, the definitions on the competition's Overview page; the leaderboard score is their average. A check confirmed our Inception features match the course's stored features with cosine 0.9999 |
| Loss curves, gradient norms, NaN count | logged every step during training; plots in outputs/full/loss_curves.png and lr_schedule.png |
| Parameter count, training time, images/sec, peak memory | from the training run; images/sec counts both domains |
| Human audit | 30 fixed holdout photos, translated by each team member's model, shuffled and blinded; both raters score style, content, and artifacts from 1 to 5; Cohen's kappa, linear-weighted kappa, and percent agreement per criterion |
| Kaggle score and rank | from the leaderboard after uploading submission.csv |

## Results

All values are from full_metrics_report.csv.

| Metric | photo to Monet | Monet to photo |
| --- | --- | --- |
| FID | 98.79 | 100.08 |
| KID mean (std) | 0.0215 (0.0011) | 0.0321 (0.0013) |
| Precision / recall | 0.370 / 0.470 | 0.463 / 0.293 |
| Density / coverage | 0.324 / 0.713 | 0.516 / 0.151 |
| Images compared, real and fake | 300, 2,000 | 2,000, 300 |
| Cycle-reconstruction L1 | 0.0641 | 0.0806 |
| Identity L1 | 0.0646 (G_BA on photos) | 0.0766 (G_AB on Monets) |
| LPIPS input vs translation | 0.363 | 0.240 |
| LPIPS input vs reconstruction | 0.221 | 0.291 |
| Content cosine similarity | 0.712 | 0.786 |

| Competition metric, all 7,038 generated images | Value |
| --- | --- |
| FID against real_stats.npz | 97.29 |
| MiFID, mean cosine distance | 0.4197 |
| Leaderboard score, (FID + MiFID) / 2 | 48.855 |
| Memorization distance to real Monets | 0.261 |
| Kaggle public score, 2026-09-28 | 48.85532 |
| Kaggle rank, 2026-09-28 | 1 (private score to fill after the competition closes) |

| Training and inference | Value |
| --- | --- |
| Final-epoch mean losses: G total / D_A / D_B | 4.628 / 0.166 / 0.075 |
| Final-epoch mean adversarial G_AB / G_BA, cycle, identity | 0.659 / 0.445, 0.232, 0.241 |
| Gradient norm G mean / max | 34.3 / 134.9 |
| Gradient norm D mean / max | 15.7 / 131.1 |
| NaN or Inf losses | 0 |
| Epochs, optimizer steps, batch | 24, 7,200, 4 |
| Training time, images/sec | 523.8 s, 110.0 |
| Peak training memory | 1,299 MB, CUDA max memory allocated |
| Inference images/sec at 256 px, photo to Monet / Monet to photo | 255.9 / 246.6 |
| Submission images | 7,038 |

Evidence: outputs/full/loss_curves.png, outputs/full/lr_schedule.png, outputs/full/history.json (every step), outputs/full/samples/epoch_001.jpg to epoch_024.jpg (rows: real photo, fake Monet, reconstructed photo, real Monet, fake photo, reconstructed Monet), outputs/full/pred_A2B_preview and pred_B2A_preview (first 60 translations of each direction), outputs/full/kaggle/official_scores.json and kaggle_results.json.

## Visual quality and cycle-consistency verification

Photo to Monet is the stronger direction. In the epoch-24 sample grid the fake Monets take on the flattened tonal range, muted greens and blues, and soft edges of the paintings while keeping the layout of the photo; the previews of countryside and coastal photos read as Monet-like at thumbnail size. Two artifacts are visible at full size and are analysed in failure_analysis.md: a fine checkerboard texture that comes from the transposed-convolution upsampling and is amplified by inferring at 256 px with a model trained at 128 px, and blotchy purple-blue speckle in flat sky regions.

Cycle consistency holds. The reconstructed photos in row 3 of every sample grid are near copies of the inputs, and the numbers agree: cycle L1 of 0.064 on the photo side and 0.081 on the Monet side, in [0, 1] pixel units, with LPIPS between input and reconstruction of 0.22 and 0.29. Content is preserved through the translation itself as well: the Inception features of a photo and its Monet version have cosine similarity 0.71, and LPIPS between input and translation is 0.36, which is the expected range for a style change that keeps the scene.

Monet to photo is weaker. The fake photos in row 5 keep visible brush texture and mostly sharpen and desaturate the painting rather than photograph it; LPIPS between input and translation is only 0.24 and coverage of the real-photo manifold is 0.15, so G_BA changes its inputs less than G_AB does. With only 270 training Monets against 6,738 photos, and identity loss holding G_BA close to the identity on photos, this direction had far less to learn from.

## Training stability and loss behaviour

Training was stable from start to finish. The cycle loss fell monotonically from 0.57 in epoch 1 to 0.23 in epoch 24 and the identity loss from 0.53 to 0.24. Both discriminator losses fell steadily, D_A from 0.40 to 0.17 and D_B from 0.46 to 0.075. The Monet discriminator D_B ended up ahead: during the decay phase G_AB's adversarial loss rose from 0.47 at epoch 4 to 0.66 at epoch 24 while D_B's loss kept falling, which is the usual sign that the discriminator on the small 270-image domain wins as the learning rate decays. It did not collapse the generator, because the cycle and identity terms kept improving and the samples kept getting better, but it is the reason a longer run with a lower discriminator learning rate would be the next experiment.

Gradient norms averaged 34 for the generators and 16 for the discriminators. Nine steps out of 7,200 exceeded 100 (seven generator, two discriminator) and were clipped at 10; none produced a non-finite loss, mixed precision stayed on for the whole run, and no loss spike is visible in the curves. The learning-rate plot shows the constant phase to epoch 12 and the linear decay after it.

An abandoned second attempt at 60 epochs (raw log Poushali_Purkayastha_task3_gan_full_20260928_184455.log) was started in the lab and stopped at epoch 23 when the session ended. None of its outputs are used; every number here comes from the completed 24-epoch run.

## Kaggle submission

submission.csv holds ID, FID, and MiFID as the class competition requires, with the two numbers from kaggle_eval.py on all 7,038 generated images against the competition's real_stats.npz. It was submitted on 2026-09-28 under team PairProgramming_Team_16 with the description "Poushali CycleGAN ResNet-6 128px 24 epochs" and scored 48.85532, the average of FID 97.29 and MiFID 0.42; lower is better. On the day of submission it was the best score on the board and the team ranked first. The generated images themselves are the unedited output of checkpoints/full/generators.pt; the full set is archived as images.zip on the team's Google Drive and the first 60 are committed in outputs/full/pred_A2B_preview. Sadaf's submission scored 66.90 with her U-Net CycleGAN.

## Human audit

To fill from outputs/full/audit/audit_results.json once both raters have scored the blinded set: mean style, content, and artifact scores per model and the inter-rater agreement. The set is built with src/audit.py from the 30 fixed holdout photos, translated by both members' generators.

## Shortcomings and future work

- Training at 128 px and inferring at 256 px was the right call for a shared lab slot but is the source of the checkerboard texture; training at 256 px, or replacing transposed convolutions with upsample-and-convolution, is the first change to make.
- 24 epochs is short for a CycleGAN. The losses were still improving and the discriminator on the Monet side was starting to dominate, so a 60-epoch run with a halved discriminator learning rate should lower FID by several points; only the best Kaggle submission counts, so it is a free improvement.
- Identity loss protects colours but holds the Monet-to-photo generator too close to the identity; a lower identity weight on that direction, or none, would give real-looking photos at the cost of some colour drift.
- FID against 300 reference paintings is a noisy, upward-biased estimate for every team, which is why KID (0.021) and the manifold metrics are reported alongside it.
- The human audit is still pending and is the only quality measure here that is not derived from Inception features.

## Hardware and run identity

- Device: NVIDIA GeForce RTX 5090, 32 GB, driver 610.60, CUDA 12.8, torch 2.11.0+cu128, bfloat16 autocast. Machine: Windows 11, 24-core Intel CPU, 68 GB RAM, GPU lab.
- Run id: full_20260928_180436, executed through src/task3_cyclegan.ipynb with configs/full.yaml and TASK3_STAGE=all on 2026-09-28; the metrics stage was rerun as full_metrics_20260928_182929 and full_metrics_20260928_183952 to add the competition scores.
- Git commit at run time: ec1788f.
- Raw log: reproducibility/raw_logs/Poushali_Purkayastha_task3_gan_full_20260928_180436.log
- Manifest: reproducibility/manifests/Poushali_Purkayastha_task3_gan_full_20260928_180436.json

## Checkpoints

checkpoints/full/generators.pt holds both generators in fp16 at epoch 24 and produced every translated image, every metric, and the Kaggle submission. The full training state with discriminators and optimizers (last.pt, 24 epochs) is in the Drive backup, not in git.

## How this model differs from Sadaf's

Sadaf's CycleGAN uses U-Net generators with 8 downsampling levels and 48 base channels (30.6M parameters each), 2-layer PatchGAN discriminators, sigmoid cross-entropy adversarial loss, cycle weight 8, no identity loss, 256 px training at batch size 1 for 20,000 steps on a Tesla T4 in 33 minutes. Mine uses the paper's ResNet-6 generators (7.8M each), the 70x70 PatchGAN, least-squares loss, identity loss at weight 5, 128 px training at batch size 4 for 7,200 steps in 9 minutes on the RTX 5090. On the competition metric mine scores 48.86 against her 66.90; her cycle L1 is lower (0.044 against 0.064) but her FID in the Kaggle direction is higher (123.3 against 97.3). The report's comparison table sets the two side by side.
