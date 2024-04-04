# AdamW optimizer for bfloat16 in PyTorch

This is a version of the AdamW optimizer for use in torch that achieves the same results in ViT training tests as training with the weights in float32 with operations in float32 or bfloat16 (autocast). By keeping your weights in bfloat16, you can save approximately half the weights they would normally take up in memory. It uses [stochastic rounding and a correction term](https://arxiv.org/pdf/2010.06192.pdf) to achieve this.

There is a small (~10-20%) performance hit depending on your hardware.

To use:

```py
from adamw_bf16 import AdamWBF16

model = model.to(dtype=torch.bfloat16)
optimizer = AdamWBF16(model.parameters(), ...)

# Train your model
```

This repository was created using code from the following two projects. It was found that insights from both could be combined to match the performance with the model weights stored in float32.

- [adamw_bfloat16](https://github.com/arogozhnikov/adamw_bfloat16)
- [OneTrainer](https://github.com/Nerogar/OneTrainer)
