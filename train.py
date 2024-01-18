import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchinfo import summary

from pruning import *
from trace_and_hook import trace_and_hook_model

from torchvision.datasets.cifar import CIFAR10

from vgg import VGG16
from lenet300100 import LeNet_300_100

MODEL_LEARNING_RATE = 0.001
MASK_LEARNING_RATE = 0.001
MASK_UPDATE_COUNTER = 40
STARTING_SALIENCE_THRESHOLD = 0.00
MODEL_LOSS_FACTOR = 1
RESOURCE_LOSS_FACTOR = 2

NUM_EPOCHS = 150
BATCH_SIZE = 64
DEVICE = torch.device("cuda")
SINGLE_INPUT_SIZE = (1, 3, 32, 32)
MODEL_EXPORT_ONNX_PATH = "modded_vgg16_pruned_6_5pct.onnx"

DESIRED_PARAMETER_PERCENT_FINAL = 5/100


def training_loop(model, classes_count, train_loader, test_loader, model_parameters, mask_parameters, global_counter, wrapped_modules=[]):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_parameters, lr=MODEL_LEARNING_RATE)
    mask_optimizer = optim.AdamW(mask_parameters, lr=MASK_LEARNING_RATE)
    mask_criterion = PruningLoss(
        wrapped_modules, MODEL_LOSS_FACTOR, RESOURCE_LOSS_FACTOR, DESIRED_PARAMETER_PERCENT_FINAL)

    writer = SummaryWriter()

    mask_update_counter = 1
    last_mask_loss = 0
    last_resource_loss = 1
    should_continue_pruning = True
    current_parameter_count = int(
        mask_criterion.current_active_parameter_count())
    print("Starting prunable parameter count is {}, desired prunable parameter count is {}".format(current_parameter_count, int(current_parameter_count * DESIRED_PARAMETER_PERCENT_FINAL)))
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader)
        for i, data in enumerate(loop):
            if last_resource_loss <= 0:
                should_continue_pruning = False
                mask_optimizer.zero_grad()
                toggle_training(model, False)
                current_parameter_count = int(mask_criterion.current_active_parameter_count())
                mask_optimizer.zero_grad()

            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            targets = F.one_hot(labels, classes_count).float()

            optimizer.zero_grad()
            mask_optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            global_batch_number = epoch * len(train_loader) + i
            writer.add_scalar("train/Loss", loss, global_batch_number)

            outputs = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            total = predicted.size(0)
            correct = (predicted == labels).sum().item()

            loss.backward(retain_graph=True)
            optimizer.step()

            loop.set_description(
                f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix({
                "loss": f"{float(loss):.3f}",
                "mask_loss": f"{float(last_mask_loss):.3f}",
                "acc": f"{float(correct / total)*100:.2f}%",
                "remain_prunable_param_count": f"{current_parameter_count}",
                "rl": mask_criterion.resource_loss_1.item(),
                "rl2": mask_criterion.resource_loss_2.item()
            })

            writer.add_scalar("train/Accuracy",
                              float(correct / total)*100, global_batch_number)

            mask_update_counter += 1
            global_counter[0] += 1

            if mask_update_counter >= MASK_UPDATE_COUNTER and should_continue_pruning:
                optimizer.zero_grad()
                mask_optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                mask_loss = mask_criterion(loss)
                last_mask_loss = mask_loss
                last_resource_loss = mask_criterion.resource_loss_2

                mask_loss.backward(retain_graph=True)

                mask_optimizer.step()
                current_parameter_count = int(
                    mask_criterion.current_active_parameter_count())

                writer.add_scalar("train/Mask Loss",
                                  last_mask_loss, global_batch_number)
                writer.add_scalar("train/Active Parameter Count",
                                  current_parameter_count, global_batch_number)

                mask_update_counter = 0

        test_overall_model_accuracy(model, test_loader, writer, epoch)

    print("Finished training")
    return model


def test_overall_model_accuracy(model, val_loader, writer=None, epoch=None):
    correct = 0
    total = 0

    model.eval()
    toggle_training(model, False)

    with torch.no_grad():
        loop = tqdm(val_loader)
        for i, data in enumerate(loop):
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            outputs = F.softmax(outputs, dim=1)

            predicted = torch.argmax(outputs, dim=1)
            total += predicted.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_description("Testing...")

    print("Accuracy of the network on the {} test images: {}%, guessed {}".format(
        total, 100 * (correct / total), correct))
    if writer:
        writer.add_scalar("val/Accuracy", float(correct / total)*100, epoch)

    toggle_training(model, True)
    model.train()


def main():
    classes_count = 10
    x = torch.randn(SINGLE_INPUT_SIZE, device=DEVICE)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_set = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    train_set = CIFAR10("./data", train=True, download=True,
                        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True)

    # test_set = torchvision.datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
    test_set = CIFAR10("./data", train=False, download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False)

    net = VGG16(num_classes=10).to(DEVICE)
    net.train()
    # net = resnet18(num_classes=10).to(DEVICE)
    net, global_counter = trace_and_hook_model(
        net, 1, STARTING_SALIENCE_THRESHOLD, DEVICE)

    model_parameters, mask_parameters, wrapped_modules = get_model_and_mask_parameters(
        net)

    net.eval()
    dump_parameter_count(net)
    summary(net, input_size=SINGLE_INPUT_SIZE)
    net.train()

    with torch.enable_grad():
        net = training_loop(net, classes_count, train_loader, test_loader, model_parameters,
                            mask_parameters, global_counter, wrapped_modules)

    print("Total parameter count is {}".format(int(get_parameter_count(net))))
    freeze_model_masks(net)

    net.eval()
    summary(net, input_size=SINGLE_INPUT_SIZE)

    test_overall_model_accuracy(net, test_loader)

    # Export the model
    with torch.no_grad():
        torch.onnx.export(net, x, MODEL_EXPORT_ONNX_PATH, export_params=True, opset_version=10,
                          do_constant_folding=True, input_names=['input'], output_names=['output'])


main()
