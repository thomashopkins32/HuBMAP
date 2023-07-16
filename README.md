# HuBMAP
Hacking the Human Vasculature (Kaggle Competition)

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
