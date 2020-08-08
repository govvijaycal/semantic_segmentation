# Semantic Segmentation with Keras

* This is a repository for Keras-based semantic segmentation networks, tuned for CARLA.
* In theory, should work with other datasets but requires tuning of dataloader implementation.
  - e.g. Input sizes must be a factor of 32 to work with pretrained models.
* References:
  - https://github.com/divamgupta/image-segmentation-keras for UNet and PSPNet implementations.
  - Panoptic FPN for FPN-based implementation: https://arxiv.org/pdf/1901.02446.pdf
  - Dice Loss and mIOU: https://www.jeremyjordan.me/semantic-segmentation/#loss

