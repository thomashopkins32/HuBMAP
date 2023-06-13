# HuBMAP
Hacking the Human Vasculature (Kaggle Competition)

## Ideas
- UNet architecture
- Predict multiple masks (one for glomerulus and one for blood vessels)
    - This way, overlaps are possible (see image with ID `0870e4f9d580`)
- Fill in gaps in the predicted mask using a post-processing step (not sure if this will be needed)
- When doing data transformations make sure to also apply them to the masks
    - If there is any randomness, we need to be able to control it so we can apply the same exact transform multiple times
- Use an ensemble to boost the final prediction
    - Maybe one conv net and one transformer model?
    - Maybe just use dropout
- Scale the loss on unsure labels by setting the labeled regions to `0.5`
- A lot more unlabeled images than labeled ones. Right now, I don't have any ideas for utilizing these.