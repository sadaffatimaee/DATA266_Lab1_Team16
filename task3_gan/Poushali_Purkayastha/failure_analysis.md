# Task 3: failure and artifact analysis

Poushali Purkayastha, Team 16

Cases are taken from outputs/full/pred_A2B_preview, outputs/full/pred_B2A_preview, and the sample grids in outputs/full/samples/, all produced by checkpoints/full/generators.pt from run full_20260928_180436. Each case names the file so it can be opened.

Failure types used: checkerboard or tiling artifacts, colour speckle in flat regions, unchanged output (near identity), loss of content, hallucinated texture, mode collapse, oversaturation.

## Case 1: checkerboard texture across the whole image

- Direction and input file: photo to Monet, photo 000ded5c41.jpg (a beach at sunset with two people)
- Image: outputs/full/pred_A2B_preview/000ded5c41.jpg
- Failure type: checkerboard or tiling artifacts
- Observation: at full size the translation is covered by a regular fine-grained grid of alternating light and dark pixels, strongest in the sky and the wet sand, which no Monet has. The grid has the period of the two transposed-convolution upsampling layers in the generator, and it is stronger in the 256 px translations than in the 128 px sample grids because the generator was trained at 128 px and run at twice that resolution, so its learned stroke scale is doubled and the upsampling kernel overlap shows. The 70x70 PatchGAN cannot penalize a texture finer than its receptive field, so training never removed it. It costs FID directly, since Inception features are sensitive to high-frequency texture.

## Case 2: colour speckle in flat regions

- Direction and input file: photo to Monet, the first column of outputs/full/samples/epoch_024.jpg (a church spire against an overcast sky) and the sky of outputs/full/pred_A2B_preview/00068bc07f.jpg (a hillside at sunset)
- Image: outputs/full/samples/epoch_024.jpg, row 2 column 1; outputs/full/pred_A2B_preview/00068bc07f.jpg
- Failure type: colour speckle, hallucinated texture
- Observation: where the photo is a flat, nearly uniform region, the fake Monet fills it with blotches of purple, blue, and orange that are not in the source and not in Monet's skies either. The discriminator learned that real Monets have colour variation everywhere, so the cheapest way for the generator to satisfy it on a flat sky is to scatter colour, and the cycle loss does not object because G_BA can remove the speckle again on the way back (the reconstruction in row 3 is clean). The identity loss limits how far the mean colour drifts but not the local variance. Textured inputs such as the forest and the ravine in the same grid do not show it, because the source already carries the variation.

## Case 3: Monet to photo output stays a painting

- Direction and input file: Monet to photo, rows 4 and 5 of outputs/full/samples/epoch_024.jpg and outputs/full/pred_B2A_preview/000c1e3bff.jpg
- Image: outputs/full/pred_B2A_preview/000c1e3bff.jpg
- Failure type: unchanged output, near identity
- Observation: the "photo" versions of the Monets are sharper and slightly less saturated but keep the brush strokes, the flat perspective, and the painted sky; nobody would mistake them for photographs. The metrics say the same: LPIPS between a Monet and its translation is 0.24 against 0.36 in the other direction, and the translations cover only 15% of the real-photo manifold (coverage 0.151, recall 0.293). Two causes: the Monet domain has 270 training images against 6,738 photos, so G_BA saw each painting 107 times and had little variety to learn a photographic prior from, and the identity loss at weight 5 rewards G_BA for leaving photo-like inputs alone, which pulls it toward small edits on everything. This direction is not the Kaggle direction, so it did not affect the leaderboard, but it counts against the cycle model as a whole.

## Training-stability failures

There was no instability in the sense of divergence: no NaN or Inf loss in 7,200 steps, mixed precision stayed on, and the loss curves in outputs/full/loss_curves.png show no spikes. Two observations from the raw log are still worth recording:

- Nine steps had a gradient norm above 100 (seven generator, two discriminator, maximum 134.9), all within the first three epochs, and were clipped to 10. After epoch 3 the generator gradient norm settled around 35 and the discriminator around 15.
- The Monet discriminator D_B gained the upper hand during the decay phase. Its loss fell from 0.46 in epoch 1 to 0.075 in epoch 24 while the adversarial loss of G_AB rose from 0.47 (epoch 4) to 0.66 (epoch 24). The generator kept improving on cycle and identity, so this is a warning rather than a failure, but a longer run at the same settings would likely see G_AB stall against a discriminator it can no longer fool.

## What I would change

1. For the checkerboard: train at 256 px, or keep 128 px training but replace each transposed convolution with bilinear upsampling followed by a 3x3 convolution, which removes the overlap that creates the grid. Test: FID and KID in the photo-to-Monet direction on the same 300 holdout photos, expecting a drop of several FID points, and a visual check of a flat-sky image.
2. For the colour speckle: add a small total-variation penalty on the generated image, or increase the image pool and train longer so the discriminator stops rewarding noise. Test: KID on the 50 flattest holdout photos, chosen by lowest pixel variance, before and after.
3. For the weak Monet-to-photo direction: drop the identity term for G_BA only, or lower it to 1, and oversample the 270 Monets with random crops at load size 286 as the paper does. Test: coverage and recall in the Monet-to-photo direction, currently 0.151 and 0.293, and cycle L1 on the Monet side, which must not rise above 0.10.
4. For the discriminator imbalance: halve the discriminator learning rate to 1e-4 or update the discriminators every second step. Test: the gap between D_B loss and G_AB adversarial loss over a 60-epoch run.
