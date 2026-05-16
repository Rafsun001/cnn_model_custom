# EfficientNetV2-S Flow Architecture

```text
EfficientNetV2SModel
    |
    +-- self.model = EfficientNetV2SFinalBlock
```

```text
##############################################################
######### Full architecture diagram with shape flow ##########
##############################################################

Input = [B, 3, 64, 64]
    |
    v
EfficientNetV2SModel
    |
    +-- EfficientNetV2SFinalBlock
            |
            +-- self.input_block      = InputBlock
            +-- self.stem_block       = StemBlock
            +-- self.stage1           = Stage1
            +-- self.stage2           = Stage2
            +-- self.stage3           = Stage3
            +-- self.stage4           = Stage4
            +-- self.stage5           = Stage5
            +-- self.stage6           = Stage6
            +-- self.head_block       = HeadBlock
            +-- self.pooling_block    = GlobalAveragePoolingBlock
            +-- self.classifier_block = ClassifierBlock
```

## 1. Main Forward Flow

```text
EfficientNetV2SFinalBlock.forward(x)
    |
    +-- InputBlock
    |       |
    |       +-- input  = [B, 3, 64, 64]
    |       +-- output = [B, 3, 64, 64]
    |
    +-- StemBlock
    |       |
    |       +-- Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
    |       +-- BatchNorm2d(24, eps=1e-3)
    |       +-- SiLU
    |       |
    |       +-- output = [B, 24, 32, 32]
    |
    +-- Stage1
    |       |
    |       +-- FusedMBConvBlock1
    |       +-- FusedMBConvBlock2
    |       |
    |       +-- output = [B, 24, 32, 32]
    |
    +-- Stage2
    |       |
    |       +-- FusedMBConvBlock3   stride 2
    |       +-- FusedMBConvBlock4
    |       +-- FusedMBConvBlock5
    |       +-- FusedMBConvBlock6
    |       |
    |       +-- output = [B, 48, 16, 16]
    |
    +-- Stage3
    |       |
    |       +-- FusedMBConvBlock7   stride 2
    |       +-- FusedMBConvBlock8
    |       +-- FusedMBConvBlock9
    |       +-- FusedMBConvBlock10
    |       |
    |       +-- output = [B, 64, 8, 8]
    |
    +-- Stage4
    |       |
    |       +-- MBConvSEBlock11     stride 2
    |       +-- MBConvSEBlock12
    |       +-- MBConvSEBlock13
    |       +-- MBConvSEBlock14
    |       +-- MBConvSEBlock15
    |       +-- MBConvSEBlock16
    |       |
    |       +-- output = [B, 128, 4, 4]
    |
    +-- Stage5
    |       |
    |       +-- MBConvSEBlock17
    |       +-- MBConvSEBlock18
    |       +-- MBConvSEBlock19
    |       +-- MBConvSEBlock20
    |       +-- MBConvSEBlock21
    |       +-- MBConvSEBlock22
    |       +-- MBConvSEBlock23
    |       +-- MBConvSEBlock24
    |       +-- MBConvSEBlock25
    |       |
    |       +-- output = [B, 160, 4, 4]
    |
    +-- Stage6
    |       |
    |       +-- MBConvSEBlock26     stride 2
    |       +-- MBConvSEBlock27
    |       +-- MBConvSEBlock28
    |       +-- MBConvSEBlock29
    |       +-- MBConvSEBlock30
    |       +-- MBConvSEBlock31
    |       +-- MBConvSEBlock32
    |       +-- MBConvSEBlock33
    |       +-- MBConvSEBlock34
    |       +-- MBConvSEBlock35
    |       +-- MBConvSEBlock36
    |       +-- MBConvSEBlock37
    |       +-- MBConvSEBlock38
    |       +-- MBConvSEBlock39
    |       +-- MBConvSEBlock40
    |       |
    |       +-- output = [B, 256, 2, 2]
    |
    +-- HeadBlock
    |       |
    |       +-- Conv2d(256, 1280, kernel_size=1, stride=1)
    |       +-- BatchNorm2d(1280, eps=1e-3)
    |       +-- SiLU
    |       |
    |       +-- output = [B, 1280, 2, 2]
    |
    +-- GlobalAveragePoolingBlock
    |       |
    |       +-- AdaptiveAvgPool2d(1)
    |       |
    |       +-- output = [B, 1280, 1, 1]
    |
    +-- ClassifierBlock
            |
            +-- Flatten
            +-- Dropout
            +-- Linear(1280, num_classes)
            |
            +-- output logits = [B, num_classes]
```

## 2. Stage Summary

```text
Stage layout:
    Stem   = Conv-BN-SiLU, stride 2, output 24
    Stage1 = FusedMBConv x2,  output 24
    Stage2 = FusedMBConv x4,  output 48,  first block stride 2
    Stage3 = FusedMBConv x4,  output 64,  first block stride 2
    Stage4 = MBConv + SE x6,  output 128, first block stride 2
    Stage5 = MBConv + SE x9,  output 160
    Stage6 = MBConv + SE x15, output 256, first block stride 2
    Head   = 1x1 Conv-BN-SiLU, output 1280
```

```text
Tensor shape flow for Tiny ImageNet input:
    input                  = [B, 3, 64, 64]
    input_block            = [B, 3, 64, 64]
    stem_block             = [B, 24, 32, 32]
    stage1                 = [B, 24, 32, 32]
    stage2                 = [B, 48, 16, 16]
    stage3                 = [B, 64, 8, 8]
    stage4                 = [B, 128, 4, 4]
    stage5                 = [B, 160, 4, 4]
    stage6                 = [B, 256, 2, 2]
    head_block             = [B, 1280, 2, 2]
    pooling_block          = [B, 1280, 1, 1]
    classifier_block       = [B, num_classes]
```

## 3. FusedMBConv Flow

```text
FusedMBConv without expansion, used in Stage1:
    input = [B, 24, H, W]
        |
        +-- Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        +-- BatchNorm2d(24, eps=1e-3)
        +-- SiLU
        +-- stochastic_depth
        +-- add identity
        |
        +-- output = [B, 24, H, W]
```

```text
FusedMBConv with expansion/projection, used in Stage2 and Stage3:
    input = [B, C_in, H, W]
        |
        +-- expand_conv:
        |       Conv2d(C_in, expanded_channels, kernel_size=3, stride=s, padding=1)
        |       BatchNorm2d(expanded_channels, eps=1e-3)
        |       SiLU
        |
        +-- project_conv:
        |       Conv2d(expanded_channels, C_out, kernel_size=1, stride=1)
        |       BatchNorm2d(C_out, eps=1e-3)
        |
        +-- if stride == 1 and C_in == C_out:
                stochastic_depth
                add identity

Stage2 first block:
    [B, 24, 32, 32] -> [B, 48, 16, 16]

Stage3 first block:
    [B, 48, 16, 16] -> [B, 64, 8, 8]
```

## 4. MBConv + SE Flow

```text
MBConvSEBlock.forward(x)
    |
    +-- identity = x
    |
    +-- expand:
    |       Conv2d(C_in, expanded_channels, kernel_size=1)
    |       BatchNorm2d(expanded_channels, eps=1e-3)
    |       SiLU
    |
    +-- depthwise:
    |       Conv2d(
    |           expanded_channels,
    |           expanded_channels,
    |           kernel_size=3,
    |           stride=s,
    |           padding=1,
    |           groups=expanded_channels,
    |       )
    |       BatchNorm2d(expanded_channels, eps=1e-3)
    |       SiLU
    |
    +-- squeeze excitation:
    |       AdaptiveAvgPool2d(1)
    |       Conv2d(expanded_channels, se_channels, kernel_size=1)
    |       SiLU
    |       Conv2d(se_channels, expanded_channels, kernel_size=1)
    |       Sigmoid
    |       out = out * scale
    |
    +-- project:
    |       Conv2d(expanded_channels, C_out, kernel_size=1)
    |       BatchNorm2d(C_out, eps=1e-3)
    |
    +-- if stride == 1 and C_in == C_out:
            stochastic_depth
            add identity
```

```text
Stage4 first block:
    [B, 64, 8, 8] -> [B, 128, 4, 4]

Stage5 first block:
    [B, 128, 4, 4] -> [B, 160, 4, 4]

Stage6 first block:
    [B, 160, 4, 4] -> [B, 256, 2, 2]
```

## 5. Stochastic Depth Flow

```text
EfficientNetV2SFinalBlock
    |
    +-- make_drop_path_rates(total_blocks=40, final_drop_path_rate=drop_path_rate)
    |
    +-- Stage1 receives rates[0:2]
    +-- Stage2 receives rates[2:6]
    +-- Stage3 receives rates[6:10]
    +-- Stage4 receives rates[10:16]
    +-- Stage5 receives rates[16:25]
    +-- Stage6 receives rates[25:40]
```

```text
StochasticDepth.forward(x)
    |
    +-- if drop_path_rate == 0.0 or model is eval:
    |       return x
    |
    +-- keep_prob = 1.0 - drop_path_rate
    +-- random_tensor shape = [B, 1, 1, 1]
    +-- return x * random_tensor / keep_prob
```

Stochastic depth is only applied when the block has a valid residual skip
connection. Downsampling blocks and channel-changing blocks do not add the
identity path.

## 6. Training Connection

```text
run_training.py
    |
    +-- ImageProcessingConfig
    |       |
    |       +-- image_size = 64
    |       +-- optional strong augmentation
    |       +-- optional random erasing
    |
    +-- TinyImageNetDataModule
    |       |
    |       +-- train_loader
    |       +-- val_loader
    |
    +-- EfficientNetV2SModel(
    |       num_classes=200,
    |       dropout_rate=training_config.model_dropout_rate,
    |       drop_path_rate=training_config.model_drop_path_rate,
    |   )
    |
    +-- EfficientNetTrainer
            |
            +-- CrossEntropyLoss
            +-- AdamW
            +-- optional AMP
            +-- optional cosine learning-rate schedule
            +-- optional mixup
            +-- optional EMA
            +-- checkpoint saving
            +-- metrics CSV writing
```

## 7. Complete Forward Summary

```text
images
    |
    v
InputBlock
    |
    v
StemBlock
    |
    v
Stage1: FusedMBConv x2
    |
    v
Stage2: FusedMBConv x4
    |
    v
Stage3: FusedMBConv x4
    |
    v
Stage4: MBConvSE x6
    |
    v
Stage5: MBConvSE x9
    |
    v
Stage6: MBConvSE x15
    |
    v
HeadBlock
    |
    v
GlobalAveragePoolingBlock
    |
    v
ClassifierBlock
    |
    v
class logits
```

## 8. Verification Command

```text
python .\EfficientNetV2-S\verify_architecture.py
```

Expected result:

```text
Output shape: [B, 200]
Parameter count with num_classes=200: 20,433,688
Parameter count with num_classes=1000: 21,458,488
```
