# V3版本
### 第一层也可以做粒度比较小的attention,使用leaky attention
### 递增形式的feature map channel就可以了诶,如(64,128,512,1024
### 组卷积的组数和attention的头数也需要改一改:1,2,4,8
### channel shuffle
### RandomMix和cutMix
### Relu替换成Gelu(只替换convblock的最后一个relu
### checkpoint 保存+恢复