# Task 3 failure analysis, Sadaf Fatima Syeda

These cases come from my submitted run (20,000 steps). I looked at the final grid, the training samples, the audit images and the metrics. In the final grid, row 2 is Photo to Monet and row 4 is Monet to Photo.

## Summary

| Case | Direction | Problem | Main cause | Fix |
| --- | --- | --- | --- | --- |
| 1 | Photo to Monet | Same small pattern repeats in the sky | Discriminator only checks small patches | Identity loss, upsample + conv in decoder |
| 2 | Monet to Photo | Dark, blurry photos | Only 300 Monet paintings to learn from | More augmentation, LSGAN loss |
| 3 | Both | White smears around edges | U-Net skips copy bright edges from the input | ResNet generator |
| 4 | Both | Longer training scored worse | Discriminator became too strong | Pick checkpoint by FID, slow down D |
| 5 | Photo to Monet | Looks like a tinted photo in busy scenes | Cycle loss keeps too much detail | Lower cycle weight |

## Case 1: Repeated pattern in the sky

Where: row 2, columns 2, 4, 5 and 6.

A small flower-like pattern repeats across flat sky areas, even in very different images. The discriminator only looks at 34 x 34 patches, so it can't see that the same patch is copied all over the image. This matches the low precision of 0.27 for Photo to Monet.

Fix: turn on the identity loss and replace transposed convolutions with upsampling + convolution.

## Case 2: Dark, blurry photos

Where: row 4, columns 3 and 4.

Some outputs are much darker than the painting and lose their detail. Turning a painting into a photo means inventing sharp detail, and the model only had 300 paintings to learn from. This direction has the worse FID (143 against 123) and low recall (0.26).

Fix: more augmentation on the Monet side and LSGAN loss instead of BCE.

## Case 3: White smears

Where: row 4, column 2 and row 2, column 3.

Bright white streaks appear along edges and around small objects. The U-Net skip connections pass the original bright edges straight to the output while the rest of the colours change.

Fix: a ResNet generator, the usual CycleGAN choice, which has no long skip connections.

## Case 4: Training longer made it worse

Where: the 80,000-step run in outputs/long_run_80k.

The score went from 66.90 to 68.30, FID got worse in both directions, and the images got grainy. The discriminator loss dropped from 0.69 to about 0.3, so the discriminators were winning and the generators learned texture tricks instead of better paintings.

Fix: choose the checkpoint by FID and slow the discriminator down with a lower learning rate or label smoothing.

## Case 5: Weak style in busy scenes

Where: row 2, columns 1 and 7.

The colours change to Monet's palette, but the picture still looks like a photo with a filter. With cycle weight 8 the model has to keep a lot of detail, which leaves less room for brushstrokes.

Fix: try a lower cycle weight, such as 5, and compare FID.

## What I would change first

Most problems are about texture: repeated patterns, grain and white edges. My first step would be to turn on the identity loss and use upsampling + convolution in the decoder, then retrain for 20,000 steps and compare with this run. Training longer on its own does not help.
