import torch
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    if padding is not None:
        return padding

    if isinstance(kernel_size, int):
        effective_kernel = dilation * (kernel_size - 1) + 1
        return effective_kernel // 2

    effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    return [k // 2 for k in effective_kernel]


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Attention(nn.Module):
    """
    PSA attention.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, channels, num_heads=4, attn_ratio=0.5):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.head_dim = channels // num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim ** -0.5

        qkv_channels = num_heads * (
            self.key_dim + self.key_dim + self.head_dim
        )

        self.qkv = Conv(
            c1=channels,
            c2=qkv_channels,
            k=1,
            s=1,
            act=False,
        )

        self.proj = Conv(
            c1=channels,
            c2=channels,
            k=1,
            s=1,
            act=False,
        )

        self.pe = Conv(
            c1=channels,
            c2=channels,
            k=3,
            s=1,
            g=channels,
            act=False,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        num_positions = height * width

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            self.num_heads,
            self.key_dim + self.key_dim + self.head_dim,
            num_positions,
        )

        q, k, v = qkv.split(
            [self.key_dim, self.key_dim, self.head_dim],
            dim=2,
        )

        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(batch_size, channels, height, width)

        v_image = v.reshape(batch_size, channels, height, width)
        out = out + self.pe(v_image)

        out = self.proj(out)

        return out


class PSABlock(nn.Module):
    """
    PSA block:
        x = x + Attention(x)
        x = x + FFN(x)
    """

    def __init__(self, channels, shortcut=True):
        super().__init__()

        self.attn = Attention(
            channels=channels,
            num_heads=max(channels // 64, 1),
            attn_ratio=0.5,
        )

        self.ffn = nn.Sequential(
            Conv(
                c1=channels,
                c2=channels * 2,
                k=1,
                s=1,
                act=True,
            ),
            Conv(
                c1=channels * 2,
                c2=channels,
                k=1,
                s=1,
                act=False,
            ),
        )

        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            x = x + self.attn(x)
            x = x + self.ffn(x)
        else:
            x = self.attn(x)
            x = self.ffn(x)

        return x


class BackboneC2PSA1(nn.Module):
    """
    Official-style C2PSA block

    Configuration:
        C2PSA(c1=256, c2=256, n=1, e=0.5)

    Input : [B, 256, H, W]
    Output: [B, 256, H, W]

    Flow:
        cv1: 256 -> 256
        split: 128 + 128
        second branch through PSABlock
        concat: 128 + 128 = 256
        cv2: 256 -> 256
    """

    def __init__(self):
        super().__init__()

        self.hidden_channels = 128

        self.cv1 = Conv(
            c1=256,
            c2=256,
            k=1,
            s=1,
        )

        self.m = nn.Sequential(
            PSABlock(
                channels=128,
                shortcut=True,
            )
        )

        self.cv2 = Conv(
            c1=256,
            c2=256,
            k=1,
            s=1,
        )

    def forward(self, x):
        a, b = self.cv1(x).split(
            [self.hidden_channels, self.hidden_channels],
            dim=1,
        )

        b = self.m(b)

        out = torch.cat([a, b], dim=1)
        out = self.cv2(out)

        return out
""" #=====> Attention  <=====#
#---> self.channels = channels
Image Input shape = [B, C, H, W]
Here,
Batch(B) = 1
Channel(C) = 128 
Height(H) = 20
Width(W) = 20               

#---> self.num_heads = num_heads
Attention is split into multiple smaller attention parts called heads.
For example, if we have 128 channels and we set num_heads to 2, then each head will attend to 128/2 = 64 channels.
So the shape of each head will be [B, 64, H, W].
Each head learns attention in a slightly different way.
Simple idea:
Head 1 may learn one type of relationship.
Head 2 may learn another type of relationship.

#---> self.head_dim = channels // num_heads
This calculates how many channels each head gets.
For example, if channels = 128 and num_heads = 2, then head_dim = 128 // 2 = 64.
So each head will have 64 channels to work with.

#--->attn_ratio
This is a hyperparameter that determines how much of the head's channels are used for keys and queries versus values.
For example, if head_dim = 64 and attn_ratio = 0.5, then key_dim = int(64 * 0.5) = 32.
This means that for each head, 32 channels will be used for keys and queries, and the remaining 32 channels will be used for values.
Why not make Q and K also 64 channels?
Because we want to reduce the computational cost of the attention mechanism. 
Q and K are mainly used to calculate attention scores:
Attention score = Qᵀ × K
Using smaller Q/K dimensions makes attention cheaper and faster to compute, while still allowing the model to learn useful relationships between features.

So attn_ratio = 0.5 means:
Use half of the head dimension for Q and K and the other half for V.

#---> self.key_dim = int(self.head_dim * attn_ratio)
This calculates the dimension of the keys and queries based on the head dimension and the attention ratio.
For example, if head_dim = 64 and attn_ratio = 0.5, then key_dim = int(64 * 0.5) = 32.
This means that for each head, the keys and queries will have 32 channels, while the values will have the remaining 32 channels.

#---> self.scale = self.key_dim ** -0.5
This is a scaling factor used in the attention mechanism to prevent the dot product of Q and K from becoming too large, which can lead to very small gradients during training.
The formula is derived from the original Transformer paper, where they found that scaling the dot product by the square root of the key dimension helps stabilize training.
For example, if key_dim = 32, then scale = 32 ** -0.5 = 1 / sqrt(32) ≈ 0.176.
This means that when we compute the attention scores, we will multiply the dot product of Q and K by this scale factor to keep the values in a reasonable range.

#---> qkv_channels = num_heads * (self.key_dim + self.key_dim + self.head_dim)
This calculates the total number of channels needed to compute the queries, keys, and values for all heads in one go.
For example, if num_heads = 2, key_dim = 32, and head_dim = 64, then:
qkv_channels = 2 * (32 + 32 + 64) = 2 * 128 = 256.
This means that the convolution layer that computes Q, K, and V will output 256 channels, which can be reshaped to separate the heads and the Q/K/V components.

#---> self.qkv = Conv(channels, qkv_channels, k=1, s=1, p=0, act=False)
This is a 1x1 convolution layer that takes the input features and produces the combined Q, K, and V representations for all heads.
For example, if channels = 128 and qkv_channels = 256, then this convolution will take the input of shape [B, 128, H, W] and output a tensor of shape [B, 256, H, W].

#---> self.proj = Conv(channels, channels, k=1, s=1, p=0, act=False)
This is a 1x1 convolution layer that projects the output of the attention mechanism back to the original number of channels.
For example, if channels = 128, then this convolution will take the output of shape [B, 128, H, W] from the attention mechanism and produce an output of shape [B, 128, H, W].

#---> self.pe = Conv(channels, channels, k=3, s=1, p=1, g=channels, act=False)
This is a depthwise convolution layer that is used to add positional information to the features.
For example, if channels = 128, then this convolution will take the input of shape [B, 128, H, W] and output a tensor of shape [B, 128, H, W].
The kernel size of 3 and padding of 1 means that it will look at a 3x3 neighborhood around each pixel, which helps the model understand spatial relationships between features.

#---> b, c, h, w = x.shape
x shape = [B, C, H, W]

Here, 
Batch(B) = 1
Channel(C) = 128
Channel Height(C) = 20
Channel Width(W) = 20

x shape = [1, 128, 20, 20]

#---> n = h * w
here, 
height(h) = 20
width(w) = 20

n = 20 * 20 = 400

So we can also write,
x shape as [B, C, n]
x shape = [1, 128, 400]

#---> qkv = self.qkv(x)
This applies the qkv convolution to the input x, which produces a tensor of shape [B, qkv_channels, H, W].
For example, if B=1, qkv_channels=256, H=20, W=20, then qkv shape will be [1, 256, 20, 20]. 

Let's x = [1, 128, 20, 20]
After qkv convolution, we get qkv = [1, 256, 20, 20]

#---> qkv = qkv.view(b, self.num_heads, self.key_dim + self.key_dim + self.head_dim, n)
This reshapes the qkv tensor to separate the heads and the Q/K/V components.
For example, if b=1, num_heads=2, key_dim=32, head_dim=64, and n=400, then:
qkv shape will be [1, 2, 32 + 32 + 64, 400] = [1, 2, 128, 400]

here, 
1= Batch size
2= Number of heads
128 = QKV features per head
400 = number of spatial positions

We pass image style like tensor([B, C, H, W]) through qkv convolution to get qkv tensor of shape [B, qkv_channels, H, W], then we reshape it to [B, num_heads, QKV features per head, number of spatial positions] to separate the heads and the Q/K/V components for attention calculation.

#---> q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
This splits the qkv tensor into separate tensors for queries (q), keys (k), and values (v) along the channel dimension.
For example, if qkv shape is [1, 2, 128, 400] after reshaeping, key_dim=32, and head_dim=64, then:
q shape will be [1, 2, 32, 400]. 

So,
key will be [1, 2, 32, 400]
query will be [1, 2, 32, 400]
value will be [1, 2, 64, 400]

Here 32, 32 is for key and query because we did self.key_dim = int(self.head_dim * attn_ratio).

#---> attn = torch.matmul(q.transpose(-2, -1), k)
This computes the attention scores by performing a matrix multiplication between the transposed queries and the keys.

#---> attn = attn * self.scale
This scales the attention scores by the factor calculated earlier to prevent them from becoming too large.

#---> attn = attn.softmax(dim=-1)
This applies the softmax function to the attention scores to convert them into probabilities.

#---> out = torch.matmul(v, attn.transpose(-2, -1))
This computes the output of the attention mechanism by performing a matrix multiplication between the values and the transposed attention scores.   

#---> out = out.view(b, c, h, w)
This reshapes the output back to the original spatial dimensions. The shpae is [B, C, H, W] after this step.

#---> out = out + self.pe(v.view(b, c, h, w))
This adds the positional encoding to the output of the attention mechanism. The positional encoding helps the model understand the spatial relationships between features.

#---> out = self.proj(out)
This projects the output of the attention mechanism back to the original number of channels.

"""

""" #=====> Attention  <=====#
#---> self.channels = channels
Image/Input feature shape = [B, C, H, W]

Here,
Batch(B)  = 1
Channel(C) = 128 
Height(H) = 20
Width(W)  = 20               

So input shape is:
x = [1, 128, 20, 20]

Here, channels = 128.
That means the Attention block receives 128 feature channels.


#---> self.num_heads = num_heads
Attention is split into multiple smaller attention parts called heads.

For example, if we have 128 channels and we set num_heads to 2, then each head will work with:

128 / 2 = 64 channels

So each head has 64 feature channels.

Each head learns attention in a slightly different way.

Simple idea:
Head 1 may learn one type of relationship.
Head 2 may learn another type of relationship.


#---> self.head_dim = channels // num_heads
This calculates how many channels each head gets.

For example:
channels = 128
num_heads = 2

head_dim = 128 // 2 = 64

So each head will have 64 channels to work with.

This 64 is also the value dimension for each head.


#---> attn_ratio
attn_ratio is a hyperparameter that controls the size of Query(Q) and Key(K).

For example:
head_dim = 64
attn_ratio = 0.5

Then:
key_dim = int(64 * 0.5) = 32

So for each head:
Query(Q) dimension = 32
Key(K) dimension   = 32
Value(V) dimension = 64

Important:
Value(V) does NOT use the remaining 32 channels.
Value(V) uses the full head_dim, which is 64.

Why make Q and K smaller than V?

Because Q and K are mainly used to calculate attention scores:

Attention score = Qᵀ × K

Using smaller Q/K dimensions reduces computation cost.
But V keeps the full head_dim because V carries the actual information that will be mixed by attention.

So attn_ratio = 0.5 means:
Q and K use half of the head dimension.
V still uses the full head dimension.


#---> self.key_dim = int(self.head_dim * attn_ratio)
This calculates the dimension of Query(Q) and Key(K).

For example:
head_dim = 64
attn_ratio = 0.5

key_dim = int(64 * 0.5) = 32

This means for each head:
Query(Q) has 32 features.
Key(K) has 32 features.
Value(V) has 64 features.


#---> self.scale = self.key_dim ** -0.5
This is a scaling factor used in attention.

The attention score is calculated using:

Qᵀ × K

This dot product can become large when key_dim is large.
Large attention scores can make softmax too sharp/extreme.

So we scale the scores before softmax.

Formula:
scale = 1 / sqrt(key_dim)

In code:
self.scale = self.key_dim ** -0.5

For example:
key_dim = 32

scale = 32 ** -0.5
scale = 1 / sqrt(32)
scale ≈ 0.176

Later we do:

attn = attn * self.scale

This keeps the attention scores in a stable range before applying softmax.


#---> qkv_channels = num_heads * (self.key_dim + self.key_dim + self.head_dim)
This calculates the total number of channels needed to create Q, K, and V for all heads.

For each head:
Q = key_dim
K = key_dim
V = head_dim

So per head:
Q + K + V = key_dim + key_dim + head_dim

For example:
num_heads = 2
key_dim = 32
head_dim = 64

qkv_channels = 2 * (32 + 32 + 64)
qkv_channels = 2 * 128
qkv_channels = 256

So the qkv convolution will output 256 channels.

These 256 channels contain:
Q for all heads
K for all heads
V for all heads


#---> self.qkv = Conv(channels, qkv_channels, k=1, s=1, p=0, act=False)
This is a 1x1 convolution layer that creates Q, K, and V together.

For example:
channels = 128
qkv_channels = 256

Input shape:
[B, 128, H, W]

Output shape:
[B, 256, H, W]

If input is:
[1, 128, 20, 20]

Then after qkv convolution:
[1, 256, 20, 20]

Important:
This 1x1 convolution mixes channel information at each spatial position.
It does not mix different spatial positions yet.

The position-to-position comparison happens later in:

attn = torch.matmul(q.transpose(-2, -1), k)


#---> self.proj = Conv(channels, channels, k=1, s=1, p=0, act=False)
This is a 1x1 convolution layer used after attention.

The attention output has shape:
[B, 128, H, W]

This projection layer keeps the same number of channels:

Input:
[B, 128, H, W]

Output:
[B, 128, H, W]

It mixes the channels after the attention operation.


#---> self.pe = Conv(channels, channels, k=3, s=1, p=1, g=channels, act=False)
This is a depthwise convolution layer used to add positional/spatial information.

For example:
channels = 128

Input:
[B, 128, H, W]

Output:
[B, 128, H, W]

Because:
k = 3
s = 1
p = 1

The height and width stay the same.

Because:
g = channels

This becomes a depthwise convolution.
That means each channel is processed separately.

Why do we need this?

Attention compares positions globally, but it does not naturally understand local 2D position like convolution does.
So this depthwise 3x3 convolution adds local spatial information.


#---> b, c, h, w = x.shape
This extracts the shape of input x.

x shape = [B, C, H, W]

Here, 
Batch(B)  = 1
Channel(C) = 128
Height(H) = 20
Width(W)  = 20

So:
x shape = [1, 128, 20, 20]


#---> n = h * w
This calculates the number of spatial positions.

Here:
height(h) = 20
width(w) = 20

n = 20 * 20 = 400

So the 20x20 feature map has 400 spatial positions.

We can think of the input as:

x shape = [B, C, H, W]
x shape = [1, 128, 20, 20]

After flattening spatial dimensions:

x shape = [B, C, n]
x shape = [1, 128, 400]

Here:
400 = number of spatial positions.


#---> qkv = self.qkv(x)
This applies the qkv convolution to input x.

Input:
x = [1, 128, 20, 20]

qkv convolution:
Conv(128, 256, k=1)

Output:
qkv = [1, 256, 20, 20]

Here:
1   = batch size
256 = QKV channels for all heads
20  = height
20  = width

This output contains Q, K, and V together.
They are not separated yet.


#---> qkv = qkv.view(b, self.num_heads, self.key_dim + self.key_dim + self.head_dim, n)
This reshapes the qkv tensor to separate attention heads and flatten spatial positions.

Before view:
qkv = [1, 256, 20, 20]

Here:
1   = batch size
256 = QKV channels for all heads
20  = height
20  = width

Now:
n = 20 * 20 = 400

Also:
num_heads = 2
key_dim = 32
head_dim = 64

QKV features per head:
32 + 32 + 64 = 128

So after view:

qkv = [1, 2, 128, 400]

Here,
1   = batch size
2   = number of heads
128 = QKV features per head
400 = number of spatial positions

Important:
After this view, it is no longer image-style [B, C, H, W].

Now it is:

[B, num_heads, QKV_features_per_head, spatial_positions]

So:
[1, 2, 128, 400]

means:

Batch 1
2 attention heads
each head has 128 QKV features
each feature has 400 spatial positions


#---> q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
This splits the qkv tensor into Query(Q), Key(K), and Value(V).

Before split:
qkv = [1, 2, 128, 400]

We split dimension 2.

Dimension 2 means:
QKV features per head

We split 128 into:

32 + 32 + 64

Because:
key_dim = 32
key_dim = 32
head_dim = 64

So:

q = [1, 2, 32, 400]
k = [1, 2, 32, 400]
v = [1, 2, 64, 400]

Here:
q = query
k = key
v = value

For q:
1   = batch size
2   = number of heads
32  = query features per head
400 = spatial positions

For k:
1   = batch size
2   = number of heads
32  = key features per head
400 = spatial positions

For v:
1   = batch size
2   = number of heads
64  = value features per head
400 = spatial positions


#---> attn = torch.matmul(q.transpose(-2, -1), k)
This computes the attention scores.

Before transpose:

q = [1, 2, 32, 400]
k = [1, 2, 32, 400]

q.transpose(-2, -1) changes q from:

[1, 2, 32, 400]

to:

[1, 2, 400, 32]

Now matrix multiplication:

q.transpose(-2, -1) @ k

Shape:

[1, 2, 400, 32] @ [1, 2, 32, 400]

Output:

attn = [1, 2, 400, 400]

Here:
1   = batch size
2   = number of heads
400 = query positions
400 = key positions

Meaning:
Each spatial position compares with every other spatial position.

So for each head, we get a 400 x 400 attention score matrix.


#---> attn = attn * self.scale
This scales the attention scores.

If key_dim = 32:

scale = 1 / sqrt(32)
scale ≈ 0.176

So every attention score is multiplied by 0.176.

Why?

Because Qᵀ × K can produce large values.
Large values can make softmax too extreme.

Scaling keeps attention scores more stable.


#---> attn = attn.softmax(dim=-1)
This applies softmax to the last dimension.

Before softmax:
attn = [1, 2, 400, 400]

After softmax:
attn = [1, 2, 400, 400]

Shape does not change.

But values become probabilities.

For each query position, softmax is applied over all key positions.

Meaning:
For each position, the model decides how much attention to give to every other position.

Each row of the 400 x 400 attention matrix sums to 1.


#---> out = torch.matmul(v, attn.transpose(-2, -1))
This uses the attention weights to mix the Value(V) features.

Before:

v = [1, 2, 64, 400]
attn = [1, 2, 400, 400]

attn.transpose(-2, -1) shape:

[1, 2, 400, 400]

The shape looks the same because the last two dimensions are both 400,
but mathematically it swaps rows and columns.

Now matrix multiplication:

v @ attn.transpose(-2, -1)

Shape:

[1, 2, 64, 400] @ [1, 2, 400, 400]

Output:

out = [1, 2, 64, 400]

Here:
1   = batch size
2   = number of heads
64  = value features per head
400 = spatial positions

Meaning:
Each output position becomes a weighted mixture of all value positions.


#---> out = out.view(b, c, h, w)
This reshapes the attention output back to image-style format.

Before:
out = [1, 2, 64, 400]

Since:
2 heads * 64 value features per head = 128 channels
400 positions = 20 * 20

After view:

out = [1, 128, 20, 20]

So the output is back to:

[B, C, H, W]


#---> out = out + self.pe(v.view(b, c, h, w))
This adds positional encoding to the attention output.

First, v is reshaped back:

v = [1, 2, 64, 400]

v.view(b, c, h, w) gives:

v = [1, 128, 20, 20]

Then:

self.pe(v)

applies depthwise 3x3 convolution.

Output of self.pe(v):

[1, 128, 20, 20]

Then we add it to out:

out = out + self.pe(v.view(b, c, h, w))

Shape:

out = [1, 128, 20, 20]

Why add positional encoding?

Attention mixes information across positions,
but it does not naturally know local spatial structure like convolution.

The depthwise 3x3 positional encoding helps add local position/spatial information.


#---> out = self.proj(out)
This applies the final 1x1 projection convolution.

Input:
out = [1, 128, 20, 20]

Projection:
Conv(128, 128, k=1)

Output:
out = [1, 128, 20, 20]

This mixes the channels after attention and positional encoding.


#---> return out
This returns the final output of the Attention block.

Final output shape:

[1, 128, 20, 20]

So the Attention block receives:

[1, 128, 20, 20]

and returns:

[1, 128, 20, 20]

The shape stays the same,
but the feature values become richer because each spatial position has gathered information from other spatial positions.

"""

""" #=====> Attention Calculation  <=====#
image:
Channel 1 =
[
  [1, 0],
  [1, 0]
]

Channel 2 =
[
  [0, 1],
  [0, 1]
]

Channel 3 =
[
  [1, 1],
  [0, 0]
]

Channel 4 =
[
  [0, 0],
  [1, 1]
]
image_shape = [1, 4, 2, 2]
num_heads = 1
attn_ratio = 0.5
channels = 4

#---> head_dim = channels // num_heads = 4 // 1 = 4

#---> self.key_dim = int(head_dim * attn_ratio) = int(4 * 0.5) = 2

#---> qkv_channels = num_heads * (self.key_dim + self.key_dim + head_dim) = 1 * (2 + 2 + 4) = 8

#---> self.qkv = Conv(channels, qkv_channels, k=1, s=1, p=0, act=False) = Conv(4, 8, k=1, s=1, p=0, act=False)

#---> self.proj = Conv(channels, channels, k=1, s=1, p=0, act=False) = Conv(4, 4, k=1, s=1, p=0, act=False)

#---> self.pe = Conv(channels, channels, k=3, s=1, p=1, g=channels, act=False) = Conv(4, 4, k=3, s=1, p=1, g=4, act=False)

#---> b, c, h, w = x.shape = [1, 4, 2, 2] 

#---> n = h * w = 2 * 2 = 4

#---> qkv = self.qkv(x) = Conv(4, 8, k=1, s=1, p=0, act=False)([1, 4, 2, 2]) = [1, 8, 2, 2]
so the output:
qkv Channel 1 =
[
  [1, 0],
  [1, 0]
]

qkv Channel 2 =
[
  [0, 1],
  [0, 1]
]

qkv Channel 3 =
[
  [1, 1],
  [0, 0]
]

qkv Channel 4 =
[
  [0, 0],
  [1, 1]
]

qkv Channel 5 =
[
  [1, 0],
  [1, 0]
]

qkv Channel 6 =
[
  [0, 1],
  [0, 1]
]

qkv Channel 7 =
[
  [1, 1],
  [0, 0]
]

qkv Channel 8 =
[
  [0, 0],
  [1, 1]
]

#---> qkv = qkv.view(b, self.num_heads, self.key_dim + self.key_dim + self.head_dim,n))
Here, 
b = 1
num_heads = 1
key_dim = 2
head_dim = 4
n = h * w = 2 * 2 = 4

This is the step where spatial flattening happens
Now each 2×2 channel becomes one row of 4 positions:
qkv =
[
  [1, 0, 1, 0],   # qkv channel 1
  [0, 1, 0, 1],   # qkv channel 2

  [1, 1, 0, 0],   # qkv channel 3
  [0, 0, 1, 1],   # qkv channel 4

  [1, 0, 1, 0],   # qkv channel 5
  [0, 1, 0, 1],   # qkv channel 6
  [1, 1, 0, 0],   # qkv channel 7
  [0, 0, 1, 1]    # qkv channel 8
]

So the shape of qkv after view will be: [1, 8, 2, 2] → [1, 1, 8, 4]

#---> q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2) = qkv.split([2, 2, 4], dim=2)
So,
Q = first 2 rows
K = next 2 rows
V = last 4 rows
Because 2 + 2 + 4 = 8
So,
Q =
[
  [1, 0, 1, 0],
  [0, 1, 0, 1]
]
shape = [1, 1, 2, 4]
K =
[
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]
shape = [1, 1, 2, 4]
V =
[
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]
Shape = [1, 1, 4, 4]

#---> attn = torch.matmul(q.transpose(-2, -1), k) = torch.matmul([1, 1, 4, 2], [1, 1, 2, 4]) = [1, 1, 4, 4]
First transpose Q.
Q =
[
  [1, 0, 1, 0],
  [0, 1, 0, 1]
]
After transpose:
Qᵀ =
[
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1]
]
K =
[
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]
Now multiply: attn = Qᵀ × K
[
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1]
]
×
[
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]
output =
[
  [1, 1, 0, 0],
  [0, 0, 1, 1],
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]
Shape = [1, 1, 4, 4]

#---> attn = attn * self.scale = attn * (self.key_dim ** -0.5) = attn * (2 ** -0.5) = attn * 0.707
Input:
[
  [1, 1, 0, 0],
  [0, 0, 1, 1],
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]
Output =
[
  [0.707, 0.707, 0.000, 0.000],
  [0.000, 0.000, 0.707, 0.707],
  [0.707, 0.707, 0.000, 0.000],
  [0.000, 0.000, 0.707, 0.707]
]

#---> attn = attn.softmax(dim=-1) = attn.softmax(dim=-1) = [1, 1, 4, 4]
output =
[
  [0.335, 0.335, 0.165, 0.165],
  [0.165, 0.165, 0.335, 0.335],
  [0.335, 0.335, 0.165, 0.165],
  [0.165, 0.165, 0.335, 0.335]
]
shape = [1, 1, 4, 4]

#---> out = torch.matmul(v, attn.transpose(-2, -1)) = torch.matmul([1, 1, 4, 4], [1, 1, 4, 4]) = [1, 1, 4, 4]
Input V =
[
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 1, 0, 0],
  [0, 0, 1, 1]
]

softmatx output  =
[
  [0.335, 0.335, 0.165, 0.165],
  [0.165, 0.165, 0.335, 0.335],
  [0.335, 0.335, 0.165, 0.165],
  [0.165, 0.165, 0.335, 0.335]
]
Softmax_outputᵀ =
[
  [0.335, 0.165, 0.335, 0.165],
  [0.335, 0.165, 0.335, 0.165],
  [0.165, 0.335, 0.165, 0.335],
  [0.165, 0.335, 0.165, 0.335]
]
out = V × Softmax_outputᵀᵀ
out =
[
  [0.500, 0.500, 0.500, 0.500],
  [0.500, 0.500, 0.500, 0.500],
  [0.670, 0.330, 0.670, 0.330],
  [0.330, 0.670, 0.330, 0.670]
]
#---> out = out.view(b, c, h, w) = out.view(1, 4, 2, 2) = [1, 4, 2, 2]
previous_out =
[
  [0.500, 0.500, 0.500, 0.500],
  [0.500, 0.500, 0.500, 0.500],
  [0.670, 0.330, 0.670, 0.330],
  [0.330, 0.670, 0.330, 0.670]
]
After reshaping:
out Channel 1 =
[
  [0.500, 0.500],
  [0.500, 0.500]
]

out Channel 2 =
[
  [0.500, 0.500],
  [0.500, 0.500]
]

out Channel 3 =
[
  [0.670, 0.330],
  [0.670, 0.330]
]

out Channel 4 =
[
  [0.330, 0.670],
  [0.330, 0.670]
]
#---> out = out + self.pe(v.view(b, c, h, w)) = out + Conv(4, 4, k=3, s=1, p=1, g=4, act=False)([1, 4, 2, 2]) = [1, 4, 2, 2]
V Channel 1 =
[
  [1, 0],
  [1, 0]
]

V Channel 2 =
[
  [0, 1],
  [0, 1]
]

V Channel 3 =
[
  [1, 1],
  [0, 0]
]

V Channel 4 =
[
  [0, 0],
  [1, 1]
]
Dummy PE: self.pe(v) = 0.1 × v
PE Channel 1 =
[
  [0.100, 0.000],
  [0.100, 0.000]
]

PE Channel 2 =
[
  [0.000, 0.100],
  [0.000, 0.100]
]

PE Channel 3 =
[
  [0.100, 0.100],
  [0.000, 0.000]
]

PE Channel 4 =
[
  [0.000, 0.000],
  [0.100, 0.100]
]
Now add to current out.

Current:

out Channel 1 =
[
  [0.500, 0.500],
  [0.500, 0.500]
]

Add PE channel 1:

[
  [0.500, 0.500],
  [0.500, 0.500]
]
+
[
  [0.100, 0.000],
  [0.100, 0.000]
]
=
[
  [0.600, 0.500],
  [0.600, 0.500]
]

Final after PE:

out Channel 1 =
[
  [0.600, 0.500],
  [0.600, 0.500]
]

out Channel 2 =
[
  [0.500, 0.600],
  [0.500, 0.600]
]

out Channel 3 =
[
  [0.770, 0.430],
  [0.670, 0.330]
]

out Channel 4 =
[
  [0.330, 0.670],
  [0.430, 0.770]
]

Shape:
out = [1, 4, 2, 2]
#---> out = self.proj(out) = Conv(4, 4, k=1, s=1, p=0, act=False)([1, 4, 2, 2]) = [1, 4, 2, 2]
Final Channel 1 =
[
  [0.600, 0.500],
  [0.600, 0.500]
]

Final Channel 2 =
[
  [0.500, 0.600],
  [0.500, 0.600]
]

Final Channel 3 =
[
  [0.770, 0.430],
  [0.670, 0.330]
]

Final Channel 4 =
[
  [0.330, 0.670],
  [0.430, 0.770]
]
"""

""" #=====> PSABlock  <=====#
PSABlock means:
Attention block + Feed Forward Network block

It is similar to a small Transformer-style block inside a CNN.

It has two main parts:

1. Attention part:
   self.attn(x)

2. Feed Forward Network part:
   self.ffn(x)

If shortcut=True, then it uses residual connections:

x = x + self.attn(x)
x = x + self.ffn(x)

This means:
First, add attention information to x.
Then, add feed-forward information to x.

So the block does not completely replace x.
It improves x by adding new learned information.


#---> def __init__(self, channels, shortcut=True):

This initializes the PSABlock.

Here:
channels = number of input channels
shortcut = whether to use residual connection or not

For example:
channels = 128
shortcut = True

Input shape:
x = [B, 128, H, W]

Example:
x = [1, 128, 20, 20]


#---> super().__init__()

This initializes the parent PyTorch nn.Module class.

We write this whenever we create a custom PyTorch module.


#---> self.attn = Attention(channels=channels, num_heads=max(channels // 64, 1), attn_ratio=0.5)

This creates the Attention block.

For example:
channels = 128

num_heads = max(channels // 64, 1)

num_heads = max(128 // 64, 1)
num_heads = max(2, 1)
num_heads = 2

So Attention receives:

channels = 128
num_heads = 2
attn_ratio = 0.5

That means:
The Attention block input shape is:

[1, 128, 20, 20]

and output shape is also:

[1, 128, 20, 20]

The shape stays the same,
but the feature values become richer because each spatial position gathers information from other positions.


#---> Why use max(channels // 64, 1)?

This chooses the number of attention heads.

Example 1:
channels = 128

channels // 64 = 2

So:
num_heads = 2

Example 2:
channels = 256

channels // 64 = 4

So:
num_heads = 4

Example 3:
channels = 32

channels // 64 = 0

But we cannot have 0 attention heads.
So max(0, 1) gives 1.

That is why we use:

max(channels // 64, 1)

Simple meaning:
Use about 64 channels per attention head,
but always use at least 1 head.


#---> self.ffn = nn.Sequential(...)

This creates the Feed Forward Network.

FFN means:
Feed Forward Network

It is made of two 1x1 convolution layers.

The FFN structure is:

Conv(channels, channels * 2, k=1)
Conv(channels * 2, channels, k=1, act=False)

For example:
channels = 128

First Conv:
128 -> 256

Second Conv:
256 -> 128

So FFN does:

[1, 128, 20, 20]
        ↓
[1, 256, 20, 20]
        ↓
[1, 128, 20, 20]

The first Conv expands channels.
The second Conv reduces channels back.


#---> Conv(channels, channels * 2, k=1, s=1, p=0)

This is the first FFN Conv layer.

For example:
channels = 128

So:

Conv(128, 256, k=1, s=1, p=0)

Input:
[1, 128, 20, 20]

Output:
[1, 256, 20, 20]

This layer expands the channel dimension.

Why expand?

Because the model gets more temporary feature space to learn richer transformations.

Simple idea:
128 channels are expanded to 256 channels,
so the model can learn more feature combinations.


#---> Conv(channels * 2, channels, k=1, s=1, p=0, act=False)

This is the second FFN Conv layer.

For example:
channels = 128

So:

Conv(256, 128, k=1, s=1, p=0, act=False)

Input:
[1, 256, 20, 20]

Output:
[1, 128, 20, 20]

This layer reduces the channels back to the original number.

Why act=False here?

Because this output will be added back to x using shortcut:

x = x + self.ffn(x)

So the last FFN layer is kept as a linear projection.
It produces a clean residual update.


#---> self.shortcut = shortcut

This stores whether shortcut connection is used.

If shortcut=True:

x = x + self.attn(x)
x = x + self.ffn(x)

If shortcut=False:

x = self.attn(x)
x = self.ffn(x)

In YOLO-style PSA blocks, shortcut=True is commonly used.

Shortcut helps keep old information while adding new learned information.


#=====> Forward Pass <=====#

#---> def forward(self, x):

This defines how input x passes through the PSABlock.

Input shape example:

x = [1, 128, 20, 20]


#---> if self.shortcut:

This checks if shortcut connection is enabled.

If shortcut=True, then this part runs:

x = x + self.attn(x)
x = x + self.ffn(x)

If shortcut=False, then this part runs:

x = self.attn(x)
x = self.ffn(x)


#---> x = x + self.attn(x)

This is the first residual connection.

Input:
x = [1, 128, 20, 20]

Attention output:
self.attn(x) = [1, 128, 20, 20]

Because both shapes are the same, we can add them:

x + self.attn(x)

Output shape:
[1, 128, 20, 20]

Simple meaning:
Original feature + attention feature

So:
x becomes richer because attention information is added to it.


#---> x = x + self.ffn(x)

This is the second residual connection.

Input:
x = [1, 128, 20, 20]

FFN output:
self.ffn(x) = [1, 128, 20, 20]

Because both shapes are the same, we can add them:

x + self.ffn(x)

Output shape:
[1, 128, 20, 20]

Simple meaning:
Current feature + feed-forward transformed feature

So:
x becomes richer again after FFN.


#---> else:

If shortcut=False, then no residual addition is used.

The block becomes:

x = self.attn(x)
x = self.ffn(x)

This means the input is replaced by the Attention output,
then replaced by the FFN output.

But with shortcut=True, the block preserves old information better.


#---> return x

This returns the final output of the PSABlock.

Input shape:
[1, 128, 20, 20]

Output shape:
[1, 128, 20, 20]

The shape stays the same.

But the values become richer because:

1. Attention adds spatial relationship information.
2. FFN adds channel transformation information.
3. Shortcut keeps the original information.

"""

""" #=====> PSABlock Calculation <=====#

We will use a tiny dummy input.

Input image:
Channel 1 =
[
  [1, 0],
  [1, 0]
]

Channel 2 =
[
  [0, 1],
  [0, 1]
]

Channel 3 =
[
  [1, 1],
  [0, 0]
]

Channel 4 =
[
  [0, 0],
  [1, 1]
]

image_shape = [1, 4, 2, 2]

channels = 4
shortcut = True


#---> self.attn = Attention(channels=channels, num_heads=max(channels // 64, 1), attn_ratio=0.5)

Here:
channels = 4

num_heads = max(channels // 64, 1)
num_heads = max(4 // 64, 1)
num_heads = max(0, 1)
num_heads = 1

So:

self.attn = Attention(
    channels=4,
    num_heads=1,
    attn_ratio=0.5
)

Input shape:
[1, 4, 2, 2]

Attention output shape:
[1, 4, 2, 2]


#---> self.ffn = nn.Sequential(...)

For channels = 4:

self.ffn = nn.Sequential(
    Conv(4, 8, k=1, s=1, p=0),
    Conv(8, 4, k=1, s=1, p=0, act=False)
)

So FFN does:

[1, 4, 2, 2]
        ↓
[1, 8, 2, 2]
        ↓
[1, 4, 2, 2]


#---> self.shortcut = shortcut

shortcut = True

So forward pass will use:

x = x + self.attn(x)
x = x + self.ffn(x)


#=====> Forward Calculation <=====#

Original input x:

x Channel 1 =
[
  [1, 0],
  [1, 0]
]

x Channel 2 =
[
  [0, 1],
  [0, 1]
]

x Channel 3 =
[
  [1, 1],
  [0, 0]
]

x Channel 4 =
[
  [0, 0],
  [1, 1]
]

Shape:
x = [1, 4, 2, 2]


#---> x = x + self.attn(x)

From our Attention calculation, suppose:

self.attn(x) Channel 1 =
[
  [0.600, 0.500],
  [0.600, 0.500]
]

self.attn(x) Channel 2 =
[
  [0.500, 0.600],
  [0.500, 0.600]
]

self.attn(x) Channel 3 =
[
  [0.770, 0.430],
  [0.670, 0.330]
]

self.attn(x) Channel 4 =
[
  [0.330, 0.670],
  [0.430, 0.770]
]

Now add original x + attention output.


Channel 1:

x Channel 1 =
[
  [1, 0],
  [1, 0]
]

Attention Channel 1 =
[
  [0.600, 0.500],
  [0.600, 0.500]
]

x + attention =
[
  [1 + 0.600, 0 + 0.500],
  [1 + 0.600, 0 + 0.500]
]

Result Channel 1 =
[
  [1.600, 0.500],
  [1.600, 0.500]
]


Channel 2:

x Channel 2 =
[
  [0, 1],
  [0, 1]
]

Attention Channel 2 =
[
  [0.500, 0.600],
  [0.500, 0.600]
]

x + attention =
[
  [0 + 0.500, 1 + 0.600],
  [0 + 0.500, 1 + 0.600]
]

Result Channel 2 =
[
  [0.500, 1.600],
  [0.500, 1.600]
]


Channel 3:

x Channel 3 =
[
  [1, 1],
  [0, 0]
]

Attention Channel 3 =
[
  [0.770, 0.430],
  [0.670, 0.330]
]

x + attention =
[
  [1 + 0.770, 1 + 0.430],
  [0 + 0.670, 0 + 0.330]
]

Result Channel 3 =
[
  [1.770, 1.430],
  [0.670, 0.330]
]


Channel 4:

x Channel 4 =
[
  [0, 0],
  [1, 1]
]

Attention Channel 4 =
[
  [0.330, 0.670],
  [0.430, 0.770]
]

x + attention =
[
  [0 + 0.330, 0 + 0.670],
  [1 + 0.430, 1 + 0.770]
]

Result Channel 4 =
[
  [0.330, 0.670],
  [1.430, 1.770]
]


So after:

x = x + self.attn(x)

New x is:

x Channel 1 =
[
  [1.600, 0.500],
  [1.600, 0.500]
]

x Channel 2 =
[
  [0.500, 1.600],
  [0.500, 1.600]
]

x Channel 3 =
[
  [1.770, 1.430],
  [0.670, 0.330]
]

x Channel 4 =
[
  [0.330, 0.670],
  [1.430, 1.770]
]

Shape:
x = [1, 4, 2, 2]


#---> x = x + self.ffn(x)

Now this updated x goes into FFN.

FFN structure:

Conv(4, 8, k=1)
Conv(8, 4, k=1, act=False)

For this dummy calculation, to keep the numbers simple, assume:

self.ffn(x) = 0.1 × x

In a real model:
self.ffn(x) is learned by Conv layers.
But here we use 0.1 × x only to show the residual calculation clearly.

So:

self.ffn(x) Channel 1 =
0.1 ×
[
  [1.600, 0.500],
  [1.600, 0.500]
]

=
[
  [0.160, 0.050],
  [0.160, 0.050]
]


self.ffn(x) Channel 2 =
0.1 ×
[
  [0.500, 1.600],
  [0.500, 1.600]
]

=
[
  [0.050, 0.160],
  [0.050, 0.160]
]


self.ffn(x) Channel 3 =
0.1 ×
[
  [1.770, 1.430],
  [0.670, 0.330]
]

=
[
  [0.177, 0.143],
  [0.067, 0.033]
]


self.ffn(x) Channel 4 =
0.1 ×
[
  [0.330, 0.670],
  [1.430, 1.770]
]

=
[
  [0.033, 0.067],
  [0.143, 0.177]
]


Now add:

x = x + self.ffn(x)


Channel 1:

x Channel 1 =
[
  [1.600, 0.500],
  [1.600, 0.500]
]

FFN Channel 1 =
[
  [0.160, 0.050],
  [0.160, 0.050]
]

Result Channel 1 =
[
  [1.600 + 0.160, 0.500 + 0.050],
  [1.600 + 0.160, 0.500 + 0.050]
]

=
[
  [1.760, 0.550],
  [1.760, 0.550]
]


Channel 2:

x Channel 2 =
[
  [0.500, 1.600],
  [0.500, 1.600]
]

FFN Channel 2 =
[
  [0.050, 0.160],
  [0.050, 0.160]
]

Result Channel 2 =
[
  [0.500 + 0.050, 1.600 + 0.160],
  [0.500 + 0.050, 1.600 + 0.160]
]

=
[
  [0.550, 1.760],
  [0.550, 1.760]
]


Channel 3:

x Channel 3 =
[
  [1.770, 1.430],
  [0.670, 0.330]
]

FFN Channel 3 =
[
  [0.177, 0.143],
  [0.067, 0.033]
]

Result Channel 3 =
[
  [1.770 + 0.177, 1.430 + 0.143],
  [0.670 + 0.067, 0.330 + 0.033]
]

=
[
  [1.947, 1.573],
  [0.737, 0.363]
]


Channel 4:

x Channel 4 =
[
  [0.330, 0.670],
  [1.430, 1.770]
]

FFN Channel 4 =
[
  [0.033, 0.067],
  [0.143, 0.177]
]

Result Channel 4 =
[
  [0.330 + 0.033, 0.670 + 0.067],
  [1.430 + 0.143, 1.770 + 0.177]
]

=
[
  [0.363, 0.737],
  [1.573, 1.947]
]


#---> return x

Final PSABlock output:

Output Channel 1 =
[
  [1.760, 0.550],
  [1.760, 0.550]
]

Output Channel 2 =
[
  [0.550, 1.760],
  [0.550, 1.760]
]

Output Channel 3 =
[
  [1.947, 1.573],
  [0.737, 0.363]
]

Output Channel 4 =
[
  [0.363, 0.737],
  [1.573, 1.947]
]

Final shape:
[1, 4, 2, 2]


#=====> PSABlock Summary <=====#

Input:
[1, 4, 2, 2]

After attention residual:
x = x + self.attn(x)
shape = [1, 4, 2, 2]

After FFN residual:
x = x + self.ffn(x)
shape = [1, 4, 2, 2]

Output:
[1, 4, 2, 2]

The shape stays the same.

But the feature values change because:

1. Attention adds spatial relationship information.
2. FFN transforms channel information.
3. Shortcut keeps original information and makes training easier.
"""

""" #=====> BackboneC2PSA1 / C2PSA Block  <=====#

C2PSA means:
C2-style split block + PSA attention block

The main idea is:

1. Take input feature.
2. Apply a 1x1 convolution.
3. Split channels into two parts.
4. Keep one part simple.
5. Send the other part through PSABlock.
6. Concatenate both parts again.
7. Apply final 1x1 convolution.

Simple flow:

Input
  ↓
cv1
  ↓
split into a and b
  ↓
a branch stays simple
b branch goes through PSABlock
  ↓
concat a and b
  ↓
cv2
  ↓
output


#=====> Input Shape <=====#

Suppose input x is:

x = [B, C, H, W]

For this block:

B = 1
C = 256
H = 20
W = 20

So:

x = [1, 256, 20, 20]


#---> self.hidden_channels = 128

This means the hidden channel size is 128.

Why 128?

Because the block will split 256 channels into two equal parts:

256 = 128 + 128

So:

a branch = 128 channels
b branch = 128 channels

Here:

self.hidden_channels = 128


#---> self.cv1 = Conv(256, 256, k=1, s=1, p=0)

This is the first 1x1 convolution.

Input:
[1, 256, 20, 20]

Output:
[1, 256, 20, 20]

It keeps the number of channels the same:

256 -> 256

It also keeps height and width the same:

20 -> 20
20 -> 20

Because:
kernel size = 1
stride = 1
padding = 0

This layer mixes channel information before splitting.

Since Conv uses default act=True, this layer is:

Conv2d -> BatchNorm2d -> SiLU


#---> self.psa = PSABlock(128, shortcut=True)

This creates a PSABlock that works on 128 channels.

Why 128?

Because after cv1, the output will be split into two parts:

a = 128 channels
b = 128 channels

Only b goes through PSABlock.

Input to PSABlock:
[1, 128, 20, 20]

Output from PSABlock:
[1, 128, 20, 20]

The shape stays the same,
but the feature values become richer because PSABlock uses:

1. Attention
2. Feed Forward Network
3. Shortcut connections


#---> self.cv2 = Conv(256, 256, k=1, s=1, p=0)

This is the final 1x1 convolution.

Before cv2, we concatenate:

a = [1, 128, 20, 20]
b = [1, 128, 20, 20]

So:

128 + 128 = 256 channels

Input to cv2:
[1, 256, 20, 20]

Output from cv2:
[1, 256, 20, 20]

This final convolution mixes the simple branch and PSA branch together.

Since Conv uses default act=True, this layer is:

Conv2d -> BatchNorm2d -> SiLU


#=====> Forward Pass <=====#

#---> def forward(self, x):

This defines how input x moves through the C2PSA block.

Input:

x = [1, 256, 20, 20]


#---> out = self.cv1(x)

This applies the first 1x1 convolution.

Input:

x = [1, 256, 20, 20]

After cv1:

out = [1, 256, 20, 20]

The channel count stays 256,
but the values are changed because cv1 mixes channel information.


#---> a, b = out.split([self.hidden_channels, self.hidden_channels], dim=1)

This splits out into two parts.

Before split:

out = [1, 256, 20, 20]

Here:

self.hidden_channels = 128

So:

out.split([128, 128], dim=1)

dim=1 means split along the channel dimension.

After split:

a = [1, 128, 20, 20]
b = [1, 128, 20, 20]

Meaning:

a gets the first 128 channels.
b gets the next 128 channels.


#---> What is branch a?

Branch a is the simple branch.

a = [1, 128, 20, 20]

It does not go through PSABlock.

It is kept as a lighter path.

This helps preserve simple information and saves computation.


#---> What is branch b?

Branch b is the PSA branch.

b = [1, 128, 20, 20]

This branch goes through:

b = self.psa(b)

So branch b gets attention and feed-forward processing.


#---> b = self.psa(b)

This sends b into PSABlock.

Input:

b = [1, 128, 20, 20]

Inside PSABlock:

b = b + Attention(b)
b = b + FFN(b)

Output:

b = [1, 128, 20, 20]

The shape stays the same,
but b now contains richer attention-based information.


#---> out = torch.cat([a, b], dim=1)

This concatenates a and b along the channel dimension.

Before concat:

a = [1, 128, 20, 20]
b = [1, 128, 20, 20]

Concatenate along dim=1:

out = torch.cat([a, b], dim=1)

Channel calculation:

128 + 128 = 256

So after concat:

out = [1, 256, 20, 20]

This combines:

a = simple preserved features
b = PSA-enhanced features


#---> out = self.cv2(out)

This applies the final 1x1 convolution.

Input:

out = [1, 256, 20, 20]

After cv2:

out = [1, 256, 20, 20]

This final convolution mixes the two branches together.

Simple meaning:

The model combines the simple path and the attention path into one final feature map.


#---> return out

This returns the final output.

Input shape:

[1, 256, 20, 20]

Output shape:

[1, 256, 20, 20]

The shape stays the same.

But the feature values are changed because:

1. cv1 mixes channels.
2. The feature is split into two branches.
3. One branch stays simple.
4. One branch gets PSA attention.
5. Both branches are concatenated.
6. cv2 mixes everything again.


#=====> Full Shape Flow <=====#

Input:

x = [1, 256, 20, 20]

After cv1:

out = [1, 256, 20, 20]

After split:

a = [1, 128, 20, 20]
b = [1, 128, 20, 20]

After PSABlock on b:

b = [1, 128, 20, 20]

After concat:

out = [1, 256, 20, 20]

After cv2:

out = [1, 256, 20, 20]

Final output:

out = [1, 256, 20, 20]


#=====> Very Simple Summary <=====#

C2PSA does this:

256-channel input
        ↓
1x1 Conv
        ↓
split into 128 + 128
        ↓
left branch: keep simple
right branch: PSABlock
        ↓
concat back to 256
        ↓
1x1 Conv
        ↓
256-channel output

Simple meaning:

C2PSA only applies expensive PSA attention to half of the channels.
The other half stays simple.
Then both are joined together.

This makes the block efficient and powerful.
"""

""" #=====> C2PSA Calculation With Small Dummy Shape <=====#

The real block uses:

Input:
[1, 256, 20, 20]

But for easy calculation, imagine a tiny version:

Input:
[1, 4, 2, 2]

hidden_channels = 2

So instead of:

256 = 128 + 128

we use:

4 = 2 + 2


#=====> Dummy Input <=====#

x Channel 1 =
[
  [1, 0],
  [1, 0]
]

x Channel 2 =
[
  [0, 1],
  [0, 1]
]

x Channel 3 =
[
  [1, 1],
  [0, 0]
]

x Channel 4 =
[
  [0, 0],
  [1, 1]
]

Shape:

x = [1, 4, 2, 2]


#---> out = self.cv1(x)

For this dummy calculation, assume cv1 is identity.

So:

out = x

out Channel 1 =
[
  [1, 0],
  [1, 0]
]

out Channel 2 =
[
  [0, 1],
  [0, 1]
]

out Channel 3 =
[
  [1, 1],
  [0, 0]
]

out Channel 4 =
[
  [0, 0],
  [1, 1]
]

Shape:

out = [1, 4, 2, 2]


#---> a, b = out.split([2, 2], dim=1)

We split 4 channels into:

2 channels + 2 channels

So:

a = first 2 channels
b = last 2 channels


a Channel 1 =
[
  [1, 0],
  [1, 0]
]

a Channel 2 =
[
  [0, 1],
  [0, 1]
]

Shape:

a = [1, 2, 2, 2]


b Channel 1 =
[
  [1, 1],
  [0, 0]
]

b Channel 2 =
[
  [0, 0],
  [1, 1]
]

Shape:

b = [1, 2, 2, 2]


#---> b = self.psa(b)

Now b goes through PSABlock.

For dummy calculation, assume PSABlock changes b into:

b Channel 1 =
[
  [1.500, 1.400],
  [0.300, 0.200]
]

b Channel 2 =
[
  [0.200, 0.300],
  [1.400, 1.500]
]

Shape:

b = [1, 2, 2, 2]

Important:
a did not go through PSABlock.
Only b went through PSABlock.


#---> out = torch.cat([a, b], dim=1)

Now concatenate a and b along channel dimension.

a has 2 channels.
b has 2 channels.

2 + 2 = 4 channels

So:

out = [1, 4, 2, 2]


After concat:

out Channel 1 = a Channel 1 =
[
  [1, 0],
  [1, 0]
]

out Channel 2 = a Channel 2 =
[
  [0, 1],
  [0, 1]
]

out Channel 3 = b Channel 1 =
[
  [1.500, 1.400],
  [0.300, 0.200]
]

out Channel 4 = b Channel 2 =
[
  [0.200, 0.300],
  [1.400, 1.500]
]


#---> out = self.cv2(out)

For this dummy calculation, assume cv2 is identity.

So output stays the same:

Final out Channel 1 =
[
  [1, 0],
  [1, 0]
]

Final out Channel 2 =
[
  [0, 1],
  [0, 1]
]

Final out Channel 3 =
[
  [1.500, 1.400],
  [0.300, 0.200]
]

Final out Channel 4 =
[
  [0.200, 0.300],
  [1.400, 1.500]
]

Final shape:

out = [1, 4, 2, 2]


#=====> Real Block Shape Version <=====#

For your actual code:

Input:
x = [1, 256, 20, 20]

cv1:
[1, 256, 20, 20] -> [1, 256, 20, 20]

split:
a = [1, 128, 20, 20]
b = [1, 128, 20, 20]

PSABlock:
b = [1, 128, 20, 20]

concat:
[1, 128, 20, 20] + [1, 128, 20, 20]
= [1, 256, 20, 20]

cv2:
[1, 256, 20, 20] -> [1, 256, 20, 20]

return:
[1, 256, 20, 20]
"""