# Third-party dependencies and attribution

No third-party source tree, patient dataset, or checkpoint is vendored in this repository. Install and use these dependencies under their own terms.

## EchoJEPA

The training and inference scripts import the ViT-L architecture from [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA). The expected initialization checkpoint is vitl-vmix22m-pt220-c55.pt; it is a large pretrained model asset and is not stored here. The upstream EchoJEPA repository is licensed under Apache-2.0 and requests citation of:

> Munim A, Fallahpour A, Szasz T, et al. *EchoJEPA: A Latent Predictive Foundation Model for Echocardiography*. arXiv:2602.02603, 2026. https://arxiv.org/abs/2602.02603

The supplied study scripts do not record the exact EchoJEPA Git commit used at study time. Pin and record the commit that matches the checkpoint/API used for a reproduction; do not assume the current upstream default branch is identical to the original run.

## EchoNet-Dynamic LV segmentation

The scripts construct a torchvision DeepLabV3-ResNet50 model and load the expected deeplabv3_resnet50_random.pt LV-segmentation checkpoint associated with [echonet/dynamic](https://github.com/echonet/dynamic). The EchoNet-Dynamic code repository is MIT-licensed. The segmentation checkpoint is not included here. Attribute the method and model to:

> Ouyang D, He B, Ghorbani A, et al. *Video-based AI for beat-to-beat assessment of cardiac function*. Nature. 2020. doi: [10.1038/s41586-020-2145-8](https://doi.org/10.1038/s41586-020-2145-8).

EchoNet-Dynamic dataset terms are separate from its source-code license. This project does not distribute the EchoNet-Dynamic dataset.

## Clinical datasets

The manuscript's external cohort references MIMIC-IV-ECHO and MIMIC-IV. Those data are not included. Access and use them only under the current [PhysioNet project terms](https://physionet.org/content/mimic-iv-echo/) and the data-use requirements applicable to the development cohort.

## Runtime packages

Direct imports used by the supplied scripts include PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm, and OpenCV. EchoJEPA supplies additional upstream dependencies; install them from the pinned EchoJEPA checkout. requirements.txt lists the project-level direct dependencies without inventing exact versions that were not captured from the study environment.

