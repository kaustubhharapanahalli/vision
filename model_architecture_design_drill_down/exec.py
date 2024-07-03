import matplotlib.pyplot as plt
import torch
from architectures import Architecture1, Architecture2
from torch import nn, optim
from torchinfo import summary
from torchvision import datasets, transforms
from utils import get_device, plot_images, test, train

SEED = 42
torch.manual_seed(SEED)

DEVICE = get_device()

loss_function = nn.CrossEntropyLoss()

###############################################################################
############################### ARCHITECTURE:01 ###############################
###############################################################################

# train_transforms_arch_1 = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
#     ]
# )

# test_transforms_arch_1 = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
#     ]
# )

# train_data_1 = datasets.MNIST(
#     "../data", train=True, download=True, transform=train_transforms_arch_1
# )
# test_data_1 = datasets.MNIST(
#     "../data", train=False, download=True, transform=test_transforms_arch_1
# )


# batches = 64

# kwargs = {"batch_size": batches, "shuffle": False}
# if torch.cuda.is_available():
#     kwargs = {
#         "batch_size": batches,
#         "shuffle": False,
#         "num_workers": 0,
#         "pin_memory": True,
#     }


# train_loader_1 = torch.utils.data.DataLoader(train_data_1, **kwargs)
# test_loader_1 = torch.utils.data.DataLoader(test_data_1, **kwargs)

# # plot_images(train_loader)

# model_1 = Architecture1().to(DEVICE)
# model_1 = torch.compile(model_1)

# summary(model_1, input_size=(1, 1, 28, 28))

# optimizer = optim.SGD(model_1.parameters(), lr=0.05, momentum=0.95)

# epochs = 15

# for epoch in range(1, epochs + 1):
#     print(f"Epoch {epoch}")
#     train(
#         model=model_1,
#         device=DEVICE,
#         train_loader=train_loader_1,
#         optimizer=optimizer,
#         loss_function=loss_function,
#     )
#     test(
#         model=model_1,
#         device=DEVICE,
#         test_loader=test_loader_1,
#         loss_function=loss_function,
#     )
#     optimizer.step()


###############################################################################
############################### ARCHITECTURE:02 ###############################
###############################################################################

# train_transforms_arch_2 = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
#     ]
# )

# test_transforms_arch_2 = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
#     ]
# )

# train_data_2 = datasets.MNIST(
#     "../data", train=True, download=True, transform=train_transforms_arch_2
# )
# test_data_2 = datasets.MNIST(
#     "../data", train=False, download=True, transform=test_transforms_arch_2
# )


# batches = 64

# kwargs = {"batch_size": batches, "shuffle": False}
# if torch.cuda.is_available():
#     kwargs = {
#         "batch_size": batches,
#         "shuffle": False,
#         "num_workers": 0,
#         "pin_memory": True,
#     }


# train_loader_2 = torch.utils.data.DataLoader(train_data_2, **kwargs)
# test_loader_2 = torch.utils.data.DataLoader(test_data_2, **kwargs)

# # plot_images(train_loader)

# model_2 = Architecture2().to(DEVICE)
# model_2 = torch.compile(model_2)

# summary(model_2, input_size=(1, 1, 28, 28))

# optimizer = optim.SGD(model_2.parameters(), lr=0.05, momentum=0.95)

# epochs = 15

# for epoch in range(1, epochs + 1):
#     print(f"Epoch {epoch}")
#     train(
#         model=model_2,
#         device=DEVICE,
#         train_loader=train_loader_2,
#         optimizer=optimizer,
#         loss_function=loss_function,
#     )
#     test(
#         model=model_2,
#         device=DEVICE,
#         test_loader=test_loader_2,
#         loss_function=loss_function,
#     )
#     optimizer.step()

###############################################################################
############################### ARCHITECTURE:03 ###############################
###############################################################################

train_transforms_arch_3 = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15.0, 15.0), fill=0),
        transforms.RandomCrop(4, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

test_transforms_arch_3 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_data_3 = datasets.MNIST(
    "../data", train=True, download=True, transform=train_transforms_arch_3
)
test_data_3 = datasets.MNIST(
    "../data", train=False, download=True, transform=test_transforms_arch_3
)


batches = 64

kwargs = {"batch_size": batches, "shuffle": False}
if torch.cuda.is_available():
    kwargs = {
        "batch_size": batches,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
    }


train_loader_3 = torch.utils.data.DataLoader(train_data_3, **kwargs)
test_loader_3 = torch.utils.data.DataLoader(test_data_3, **kwargs)

# plot_images(train_loader)

model_3 = Architecture2().to(DEVICE)
model_3 = torch.compile(model_3)

summary(model_3, input_size=(1, 1, 28, 28))

optimizer = optim.SGD(model_3.parameters(), lr=0.05, momentum=0.95)

epochs = 15

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    train(
        model=model_3,
        device=DEVICE,
        train_loader=train_loader_3,
        optimizer=optimizer,
        loss_function=loss_function,
    )
    test(
        model=model_3,
        device=DEVICE,
        test_loader=test_loader_3,
        loss_function=loss_function,
    )
    optimizer.step()
