# HuBMAP
Hacking the Human Vasculature (Kaggle Competition)

## Submissions

### Release 1
The notebook should be runnable from start to finish on Kaggle's servers as long as you add the correct datasets to the environment.

Here are the specifications:
- UNet architecture
- Supervised learning, only
- 200 epochs of training
- Adam optimizer
- Batch normalization
- No weight decay (not AdamW)
- `3e-4` fixed learning rate
- 90-10 training-validation split
- Predicting 3 separate classes:
    - 0: background
    - 1: glomerulus
    - 2: blood vessel (target structure)
 
Results:
- Training loss decreased to near `0.0` but validation loss did not (overfitting!).
- Training mAP was hovering near `0.9` after 200 epochs (using my custom implementation).
- Validation mAP was `0.1815` after 200 epochs (using my custom implementation).
- Kaggle leaderboard mAP was `0.183` which got me 870th out of 1002 with 6 days left in the competition.

## Ideas
- UNet architecture
- Predict multiple masks (one for glomerulus and one for blood vessels)
    - Overlaps are possible (see image with ID `0870e4f9d580`) so we cannot exclude the glomerulus region
- When doing data transformations make sure to also apply them to the masks
    - If there is any randomness, we need to be able to control it so we can apply the same exact transform multiple times
- Use an ensemble to boost the final prediction
    - Maybe one UNet and one transformer?
    - Maybe just use dropout (doesn't work well with batch norm though)
- Scale the loss on unsure labels by setting the labeled regions to `0.5`
- A lot more unlabeled images than labeled ones.
    - Self-supervised or unsupervised pre-training followed by supervised fine-tuning
