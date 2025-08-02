Some layers need to preserve length/size of temporal dimension, as pytorch has no ready to use casual same padding we will implement our own padding. Using the formula:

$$L_{out}=\lfloor \frac{L_{in} + 2*padding - dilation*(kernel\_size-1)-1}{stride} +1\rfloor$$
From https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

We will derive amount of zeros that need to be added on the left side of the input to preserve temporal dimension.

We will only pad left side, so our formula is:


$$L_{out}=\lfloor \frac{L_{in} + padding\_left - dilation*(kernel\_size-1)-1}{stride} +1\rfloor$$

We want $L_{out}$ to be equal to $L_{in}$:
$$L_{in}=\lfloor \frac{L_{in} + padding\_left - dilation*(kernel\_size-1)-1}{stride} +1\rfloor$$
We will skip the floor as formula for padding won't involve any division operation:

$$L_{in}=\frac{L_{in} + padding\_left - dilation*(kernel\_size-1)-1}{stride} +1$$

$$L_{in}*stride=L_{in} + padding\_left - dilation*(kernel\_size-1)-1 +stride$$

$$(L_{in}-1)*stride - L_{in} + dilation*(kernel\_size-1) + 1=padding\_left$$

$$padding\_left= (L_{in}-1)*stride - L_{in} + dilation*(kernel\_size-1) + 1$$