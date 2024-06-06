# Receptive field Computation Analysis

For any model using CNN layers, the receptive field can be analysed using the
formula:

$$n_{out} = \frac{n_{in}+2p-k}{s} + 1$$
$$r_{out} = r_{in} + (k - 1) \cdot j_{in}$$
$$j_{out} = j_{in} \cdot s$$
