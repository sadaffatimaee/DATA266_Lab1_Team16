# Task 3: shared Monet and photo images

The competition data lives here as two flat folders of JPEG images, which are not committed:

```
task3_gan/data/monet_jpg/   real Monet paintings, 256 x 256
task3_gan/data/photo_jpg/   real photos, 256 x 256
```

Source: the class Kaggle competition [data-266-fall-2026-gan-image-style-transfer](https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer). With the Kaggle CLI configured (`kaggle.json` in your home `.kaggle` folder, never in the repo), download and unpack with:

```
python task3_gan/Poushali_Purkayastha/src/download_data.py --source kaggle
```

For local testing without Kaggle credentials the public monet2photo release from the CycleGAN authors has the same photo set (7,038 images) and a larger Monet set (1,193 images):

```
python task3_gan/Poushali_Purkayastha/src/download_data.py --source berkeley
```

The Kaggle submission must be generated from a model trained on the competition's own monet_jpg and photo_jpg, so the lab run uses the Kaggle download.
