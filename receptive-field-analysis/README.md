# Receptive field Computation Analysis

For any model using CNN layers, the receptive field can be analysed using the
below formulae:

$$n_{out} = \frac{n_{in}+2p-k}{s} + 1$$

$$r_{out} = r_{in} + (k - 1) \cdot j_{in}$$

$$j_{out} = j_{in} \cdot s$$

## Model Architecture

| Layers        | Description   | Params #  |
| :-----------: | :-----------: | :-------: |
| Input         | [1, 10]       | --        |
| Conv2d        | [32, 28, 28]  | 320       |
| Conv2d        | [64, 28, 28]  | 18,496    |
| MaxPool2d     | [64, 14, 14]  | --        |
| Conv2d        | [128, 14, 14] | 73,856    |
| Conv2d        | [256, 14, 14] | 295,168   |
| MaxPool2d     | [256, 7, 7]   | --        |
| Conv2d        | [512, 5, 5]   | 1,180,160 |
| Conv2d        | [1024, 3, 3]  | 4,719,616 |
| Conv2d        | [10, 1, 1]    | 92,170    |

## Receptive Field Computation

Computation of the receptive field at each layer using the above listed formulae is shown :

| Layers        | $n_{in}$ | $r_{in}$ | $j_{in}$ | $s$ | $p$ | $k$ | $n_{out}$ | $r_{out}$ | $j_{out}$ |
| :-----------: | :------: | :------: | :------: | :-: | :-: | :-: | :-------: | :-------: | :-------: |
| Input         | --       | --       | --       | --  | --  | --  | --        | --        | --        |
| Conv2d        | 28       | 1        | 1        | 1   | 1   | 3   | 28        | 3         | 1         |
| Conv2d        | 28       | 3        | 1        | 1   | 1   | 3   | 28        | 5         | 1         |
| MaxPool2d     | 28       | 5        | 1        | 2   | 0   | 2   | 14        | 6         | 2         |
| Conv2d        | 14       | 6        | 2        | 1   | 1   | 3   | 14        | 10        | 2         |
| Conv2d        | 14       | 10       | 2        | 1   | 1   | 3   | 14        | 14        | 2         |
| MaxPool2d     | 14       | 14       | 2        | 2   | 0   | 2   | 7         | 16        | 4         |
| Conv2d        | 7        | 16       | 4        | 1   | 0   | 3   | 5         | 24        | 4         |
| Conv2d        | 5        | 24       | 4        | 1   | 0   | 3   | 3         | 32        | 4         |
| Conv2d        | 3        | 32       | 4        | 1   | 0   | 3   | 1         | 40        | 4         |
