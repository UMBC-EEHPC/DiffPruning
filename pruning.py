import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
import math


# Straight-Through Estimator for binary mask created from differentiable mask.
class CreateBinaryMask(Function):
    @staticmethod
    def forward(differentiable_mask, salience_threshold):
        binary_mask = torch.where(
            differentiable_mask > salience_threshold[0], 1., 0.)
        assert torch.any(binary_mask < 0) == False
        return binary_mask

    @staticmethod
    def setup_context(ctx, inputs, output):
        differentiable_mask, _ = inputs
        ctx.save_for_backward(differentiable_mask)

    @staticmethod
    def backward(ctx, grad_binary_mask):
        grad_differentiable_mask = F.hardtanh(grad_binary_mask)
        return grad_differentiable_mask, None


class FCStructuredPruningWrapper(nn.Module):
    def __init__(self, numeric_id, my_names, shared_pruning_weights, global_counter, network_metadata, wrapped_fc, device):
        super(FCStructuredPruningWrapper, self).__init__()

        self.numeric_id = numeric_id
        self.my_names = my_names
        self.shared_pruning_weights = shared_pruning_weights
        self.global_counter = global_counter
        self.network_metadata = network_metadata

        self.prior_prunables = None
        self.prior_diff_prunables = None

        self.is_frozen = False
        self.should_prune_outputs = True
        self.training = True

        self.device = device

        self.fc = wrapped_fc

        self.mask_length = self.fc.weight.shape[0]

        self.last_output_activation_size = None

        # Mask used for actually thinning channels
        self.shared_pruning_weights[self.numeric_id]["channel_masks"] = torch.ones(
            (self.mask_length), requires_grad=True, device=device)

        self.output_size = torch.tensor(
            self.fc.weight.shape[0], dtype=torch.float32, requires_grad=False, device=device)
        self.input_size = torch.tensor(
            self.fc.weight.shape[1], dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, x):
        y = self.fc(x)

        if self.training and self.should_prune_outputs:
            shared_layer_info = self.shared_pruning_weights[self.numeric_id]

            if self.global_counter[0] != shared_layer_info["last_iteration"]:
                combin_norm_weights = torch.zeros_like(
                    self.fc.weight.clone().detach())

                # We need to figure out the output salience to figure out what to prune
                for i in range(len(shared_layer_info["wrapper_list"])):
                    combin_norm_weights += shared_layer_info["wrapper_list"][i].fc.weight.clone(
                    ).detach()
                combin_norm_weights /= len(shared_layer_info["wrapper_list"])
                salience = torch.linalg.vector_norm(combin_norm_weights, dim=1)

                # The paper defines this as the differentiable mask formula
                differentiable_masks = 1 / \
                    (1 + (shared_layer_info["salience_threshold"] / salience))

                # The paper specifies that the number of channels greater than the
                # salience threshold should be roughly equivalent to the layerwise
                # width multiplier
                shared_layer_info["channel_masks"] = CreateBinaryMask.apply(
                    differentiable_masks, shared_layer_info["salience_threshold"].clone().detach())
                assert torch.any(
                    shared_layer_info["channel_masks"] < 0) == False

                shared_layer_info["last_iteration"] += 1

        if self.should_prune_outputs and not self.is_frozen:
            y = shared_layer_info["channel_masks"].unsqueeze(
                0).clone().detach() * y

        self.last_output_activation_size = y.shape

        return y

    def num_active_elements(self):
        active_in_channels = self.input_size
        active_out_channels = self.output_size
        shared_layer_info = self.shared_pruning_weights[self.numeric_id]

        # Calculate number of live elements in output mask.
        if self.should_prune_outputs:
            active_out_channels = torch.sum(
                F.relu(shared_layer_info["channel_masks"]))

        # Calculate percentage of live elements in output mask
        active_channels = active_out_channels / \
            shared_layer_info["channel_masks"].shape[0]

        if len(self.prior_prunables):
            # Calculate number of live elements in input mask.
            active_in_channels = torch.sum(
                F.relu(self.shared_pruning_weights[self.prior_prunables[0]]["channel_masks"]))

            # Calculate percentage of live elements in input mask and update
            # percentage of over all active channels.
            active_channels *= active_in_channels / \
                self.shared_pruning_weights[self.prior_prunables[0]
                                            ]["channel_masks"].shape[0]
        elif len(self.prior_diff_prunables):
            # ID of the prior wrapper.
            id_of_prior_diff_prunable = self.prior_diff_prunables[0]["id"]
            # Any layers that modify the output activation size from the prior wrappers.
            prior_activation_size_modifiers = self.prior_diff_prunables[
                0]["modifies_activation_size"]
            # Prior ConvStructuredPruningWrapper.
            prior_diff_prunable = self.shared_pruning_weights[
                id_of_prior_diff_prunable]["wrapper_list"][0]

            # If there is a prior activation size.
            if prior_diff_prunable.last_output_activation_size:
                prior_input_activation_size = prior_diff_prunable.last_output_activation_size
                # Check if there's any prior activation size modifiers, and if
                # there is, check how they would end up modifying what will end
                # up being the input activation size to this current wrapper.
                if len(prior_activation_size_modifiers):
                    for size_modifier in prior_activation_size_modifiers:
                        prior_input_activation_size = new_activation_size(
                            size_modifier, prior_input_activation_size)

                _, _, H, W = prior_input_activation_size
                active_in_channels = torch.sum(
                    F.relu(self.shared_pruning_weights[id_of_prior_diff_prunable]["channel_masks"]))
                active_in_channels *= (H * W)

        assert active_in_channels >= 0
        if active_out_channels < 0:
            print("active_out_channels: {}".format(active_out_channels))
            print(shared_layer_info["channel_masks"])
            assert active_out_channels >= 0
        # Calculate number of active elements post-mask application.
        # Left-hand is linear weights, right-hand is linear bias.
        return (active_in_channels * active_out_channels) + (1 * active_out_channels)

    def apply_mask(self):
        prior_mask = torch.ones((self.fc.weight.shape[1])).bool()
        own_mask = torch.ones((self.fc.weight.shape[0])).bool()

        print("FC {}: {} active parameters".format(
            self.my_names[0], int(self.num_active_elements())))

        # Grab mask from prior prunable layer.
        if len(self.prior_prunables):
            prior_mask = self.shared_pruning_weights[self.prior_prunables[0]]["channel_masks"].bool(
            )
        elif len(self.prior_diff_prunables):
            # ID of the prior wrapper.
            id_of_prior_diff_prunable = self.prior_diff_prunables[0]["id"]
            # Any layers that modify the output activation size from the prior wrappers.
            prior_activation_size_modifiers = self.prior_diff_prunables[
                0]["modifies_activation_size"]
            # Prior ConvStructuredPruningWrapper.
            prior_diff_prunable = self.shared_pruning_weights[
                id_of_prior_diff_prunable]["wrapper_list"][0]

            # If there is a prior activation size.
            if prior_diff_prunable.last_output_activation_size:
                prior_input_activation_size = prior_diff_prunable.last_output_activation_size
                # Check if there's any prior activation size modifiers, and if
                # there is, check how they would end up modifying what will end
                # up being the input activation size to this current wrapper.
                if len(prior_activation_size_modifiers):
                    for size_modifier in prior_activation_size_modifiers:
                        prior_input_activation_size = new_activation_size(
                            size_modifier, prior_input_activation_size)

                _, C, H, W = prior_input_activation_size
                expanded_mask = self.shared_pruning_weights[id_of_prior_diff_prunable]["channel_masks"].view(
                    C, 1, 1)
                expanded_mask = expanded_mask.expand(C, H, W)
                prior_mask = expanded_mask.flatten().bool()

        # Grab our own mask if we are going to prune our output channels.
        if self.should_prune_outputs:
            own_mask = self.shared_pruning_weights[self.numeric_id]["channel_masks"].bool(
            )

        new_fc = nn.Linear(in_features=int(torch.sum(prior_mask.int()).item()), out_features=int(torch.sum(
            own_mask.int()).item()), bias=(self.fc.bias != None), device=self.device)

        print("{} old weight shape: {} new weight shape: {}".format(
            self.my_names[0], self.fc.weight.shape, new_fc.weight.shape))

        new_weights = self.fc.weight.data.clone().detach()

        new_weights = new_weights[own_mask]
        new_weights = new_weights[:, prior_mask]
        print("{} new out channels is {} new in_channels is {} \n".format(self.my_names[0], int(
            torch.sum(own_mask.int()).item()), int(torch.sum(prior_mask.int()).item())))

        new_fc.weight = nn.Parameter(new_weights)

        if self.fc.bias != None:
            new_bias = self.fc.bias.data.clone().detach()

            new_fc.bias = nn.Parameter(new_bias[own_mask])

        self.fc = new_fc
        self.is_frozen = True


class ConvStructuredPruningWrapper(nn.Module):
    def __init__(self, numeric_id, my_names, shared_pruning_weights, global_counter, network_metadata, wrapped_conv2d, wrapped_bn, device):
        super(ConvStructuredPruningWrapper, self).__init__()

        self.numeric_id = numeric_id
        self.my_names = my_names
        self.shared_pruning_weights = shared_pruning_weights
        self.global_counter = global_counter
        self.network_metadata = network_metadata

        self.prior_prunables = None
        self.prior_diff_prunables = None

        self.is_frozen = False
        self.should_prune_outputs = True
        self.training = True

        self.device = device

        self.conv = wrapped_conv2d
        self.bn = wrapped_bn

        self.mask_length = self.conv.weight.shape[0]

        self.last_output_activation_size = None

        # Mask used for actually thinning channels.
        self.shared_pruning_weights[self.numeric_id]["channel_masks"] = torch.ones(
            (self.mask_length), requires_grad=True, device=device)

        self.output_channels = torch.tensor(
            self.conv.weight.shape[0], dtype=torch.float32, requires_grad=False, device=device)
        self.in_channels = torch.tensor(
            self.conv.weight.shape[1], dtype=torch.float32, requires_grad=False, device=device)
        self.elems = torch.tensor(
            self.conv.weight.shape[2] * self.conv.weight.shape[3], dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)

        if self.training and self.should_prune_outputs:
            shared_layer_info = self.shared_pruning_weights[self.numeric_id]

            if self.global_counter[0] != shared_layer_info["last_iteration"]:
                wrapper_list = shared_layer_info["wrapper_list"]

                combin_abs_weights = torch.zeros_like(
                    self.bn.weight.clone().detach())
                combin_norm_weights = torch.zeros(
                    self.conv.weight.shape[0]).to(self.device)

                # We need to figure out the output salience to figure out what
                # to prune, however, in order to do that, we need to average
                # out the initial saliency measurements across all the layers
                # that share the same pruning mask.
                for i in range(len(wrapper_list)):
                    combin_abs_weights += wrapper_list[i].bn.weight.clone().detach()
                    combin_norm_weights += torch.linalg.norm(torch.flatten(
                        wrapper_list[i].conv.weight.clone().detach(), start_dim=1), dim=1)
                combin_abs_weights /= len(wrapper_list)
                combin_norm_weights /= len(wrapper_list)
                combin_norm_weights /= torch.sum(combin_norm_weights)
                combin_norm_weights = torch.abs(combin_norm_weights)
                salience = torch.abs(combin_abs_weights) * combin_norm_weights

                # The paper defines this as the differentiable mask formula.
                differentiable_masks = 1 / \
                    (1 + (shared_layer_info["salience_threshold"] / salience))

                # The paper specifies that the number of channels greater than the
                # salience threshold should be roughly equivalent to the layerwise
                # width multiplier.
                shared_layer_info["channel_masks"] = CreateBinaryMask.apply(
                    differentiable_masks, shared_layer_info["salience_threshold"].clone().detach())
                assert torch.any(
                    shared_layer_info["channel_masks"] < 0) == False

                shared_layer_info["last_iteration"] += 1

        if self.should_prune_outputs and not self.is_frozen:
            y = shared_layer_info["channel_masks"].unsqueeze(
                -1).unsqueeze(-1).unsqueeze(0).clone().detach() * y

        self.last_output_activation_size = y.shape

        return y

    def num_active_elements(self):
        active_in_channels = self.in_channels
        active_out_channels = self.output_channels

        shared_layer_info = self.shared_pruning_weights[self.numeric_id]

        # Calculate number of live elements in output mask.
        if self.should_prune_outputs:
            active_out_channels = torch.sum(F.relu(shared_layer_info["channel_masks"]))

        # Calculate percentage of live elements in output mask.
        active_channels = active_out_channels / shared_layer_info["channel_masks"].shape[0]

        if len(self.prior_prunables):
            # Calculate number of live elements in input mask.
            active_in_channels = torch.sum(
                F.relu(self.shared_pruning_weights[self.prior_prunables[0]]["channel_masks"]))

            # Calculate percentage of live elements in input mask and update
            # percentage of over all active channels.
            active_channels *= active_in_channels / \
                self.shared_pruning_weights[self.prior_prunables[0]
                                            ]["channel_masks"].shape[0]

        # Calculate number of active elements post-mask application.
        # Left-hand is Conv2d weights, right-hand is BatchNorm2d weights and
        # both layer biases.
        # return (self.output_channels * self.in_channels * active_channels * self.elems) + (3 * active_out_channels)
        assert active_in_channels >= 0
        if active_out_channels < 0:
            print("active_out_channels: {}".format(active_out_channels))
            print(shared_layer_info["channel_masks"])
            assert active_out_channels >= 0
        assert self.elems >= 0
        return (active_in_channels * active_out_channels * self.elems) + (3 * active_out_channels)

    def apply_mask(self):
        prior_mask = torch.ones((self.conv.weight.shape[1])).bool()
        own_mask = torch.ones((self.conv.weight.shape[0])).bool()
        kernel_size = (self.conv.weight.shape[2], self.conv.weight.shape[3])

        print("Conv-BN {}-{}: {} active parameters".format(
            self.my_names[0], self.my_names[1], int(self.num_active_elements())))

        # Grab mask from prior prunable layer.
        if len(self.prior_prunables):
            prior_mask = self.shared_pruning_weights[self.prior_prunables[0]
                                                     ]["channel_masks"]
            prior_mask = prior_mask.bool()

        # Grab our own mask if we are going to prune our output channels.
        if self.should_prune_outputs:
            own_mask = self.shared_pruning_weights[self.numeric_id]["channel_masks"].bool(
            )

        new_conv = nn.Conv2d(in_channels=int(torch.sum(prior_mask.int()).item()), out_channels=int(torch.sum(
            own_mask.int()).item()), kernel_size=kernel_size, stride=self.conv.stride, padding=self.conv.padding, device=self.device)
        new_bn = nn.BatchNorm2d(num_features=torch.sum(
            own_mask.int()), device=self.device)

        print("{} old weight shape: {} new weight shape: {}".format(
            self.my_names[0], self.conv.weight.shape, new_conv.weight.shape))
        print("{} old weight shape: {} new weight shape: {}".format(
            self.my_names[1], self.bn.weight.shape, new_bn.weight.shape))

        new_conv_weights = self.conv.weight.data.clone().detach()
        new_conv_bias = self.conv.bias.data.clone().detach()

        new_conv_weights = new_conv_weights[own_mask]
        new_conv_weights = new_conv_weights[:, prior_mask]
        new_conv.weight = nn.Parameter(new_conv_weights)
        new_conv_bias = nn.Parameter(new_conv_bias[own_mask])

        new_bn_weights = self.bn.weight.data.clone().detach()
        new_bn_bias = self.bn.bias.data.clone().detach()
        new_bn_running_mean = self.bn.running_mean.data.clone().detach()
        new_bn_running_var = self.bn.running_var.data.clone().detach()

        new_bn.weight = nn.Parameter(new_bn_weights[own_mask])
        new_bn.bias = nn.Parameter(new_bn_bias[own_mask])
        new_bn.running_mean = nn.Parameter(new_bn_running_mean[own_mask])
        new_bn.running_var = nn.Parameter(new_bn_running_var[own_mask])

        self.conv = new_conv
        self.bn = new_bn
        self.is_frozen = True


class PruningLoss(nn.Module):
    def __init__(self, wrapped_modules, model_loss_factor, resource_loss_factor, desired_parameter_percent_final):
        super(PruningLoss, self).__init__()

        self.model_loss_factor = model_loss_factor
        self.resource_loss_factor = resource_loss_factor

        self.wrapped_modules = wrapped_modules

        self.start_parameter_count = self.current_active_parameter_count()
        self.desired_parameter_percent_final = desired_parameter_percent_final

        self.resource_loss_1 = torch.tensor([0.])
        self.resource_loss_2 = torch.tensor([0.])

    def forward(self, task_loss):
        current_percentage = self.current_active_parameter_count() / \
            self.start_parameter_count
        resource_loss = torch.pow(
            (current_percentage - self.desired_parameter_percent_final), 3)
        self.resource_loss_1 = resource_loss
        self.resource_loss_2 = (current_percentage -
                                self.desired_parameter_percent_final)

        return (self.model_loss_factor * task_loss) + (self.resource_loss_factor * resource_loss)

    def current_active_parameter_count(self):
        parameter_count = 0
        for mod in self.wrapped_modules:
            parameter_count += mod.num_active_elements()
        return parameter_count


def new_activation_size(layer, input_size):
    if isinstance(layer, nn.MaxPool2d):
        assert len(input_size) == 4 or len(input_size) == 3

        layer_kernel_size = layer.kernel_size
        layer_padding = layer.padding
        layer_dilation = layer.dilation
        layer_stride = layer.stride

        if isinstance(layer_kernel_size, int):
            layer_kernel_size = [layer_kernel_size, layer_kernel_size]
        if isinstance(layer_padding, int):
            layer_padding = [layer_padding, layer_padding]
        if isinstance(layer_dilation, int):
            layer_dilation = [layer_dilation, layer_dilation]
        if isinstance(layer_stride, int):
            layer_stride = [layer_stride, layer_stride]

        h_out = math.floor(((input_size[-2] + 2 * layer_padding[0] - layer_dilation[0] * (
            layer_kernel_size[0] - 1) - 1) / layer_stride[0]) + 1)
        w_out = math.floor(((input_size[-1] + 2 * layer_padding[1] - layer_dilation[1] * (
            layer_kernel_size[1] - 1) - 1) / layer_stride[1]) + 1)

        if len(input_size) == 4:
            output_size = [input_size[0],
                           input_size[1], h_out, w_out]
        else:
            output_size = [input_size[0], h_out, w_out]
        return output_size

    assert False


def freeze_model_masks(model, prior_mask=None):
    named_children = list(model.named_children())
    for module_subindex in range(len(named_children)):
        _, module = named_children[module_subindex]
        if isinstance(module, FCStructuredPruningWrapper) or isinstance(module, ConvStructuredPruningWrapper):
            module.apply_mask()
            module.training = False

        freeze_model_masks(module, prior_mask)


def toggle_training(model, new_state):
    named_children = list(model.named_children())
    for module_subindex in range(len(named_children)):
        _, module = named_children[module_subindex]
        if isinstance(module, FCStructuredPruningWrapper) or isinstance(module, ConvStructuredPruningWrapper):
            module.training = new_state

        toggle_training(module, new_state)


def get_parameter_count(model):
    module_parameter_count = 0
    named_children = list(model.named_children())
    for module_subindex in range(len(named_children)):
        _, module = named_children[module_subindex]
        if isinstance(module, nn.Module):
            if isinstance(module, FCStructuredPruningWrapper) or isinstance(module, ConvStructuredPruningWrapper):
                module_parameter_count += module.num_active_elements()

            module_parameter_count += get_parameter_count(module)

    return module_parameter_count


def dump_parameter_count(model):
    module_parameter_count = 0
    named_children = list(model.named_children())
    for module_subindex in range(len(named_children)):
        _, module = named_children[module_subindex]
        if isinstance(module, nn.Module):
            if isinstance(module, FCStructuredPruningWrapper):
                module_parameter_count += module.num_active_elements()
                print("ID: {} FC {} Active Parameters: {} should_prune_outputs: {}".format(
                    module.numeric_id, module.my_names[0], int(module.num_active_elements()), module.should_prune_outputs))
            elif isinstance(module, ConvStructuredPruningWrapper):
                module_parameter_count += module.num_active_elements()
                print("ID: {} Conv-BN {}-{} Active Parameters: {} should_prune_outputs: {}".format(module.numeric_id,
                      module.my_names[0], module.my_names[1], int(module.num_active_elements()), module.should_prune_outputs))

            module_parameter_count += dump_parameter_count(module)

    return module_parameter_count


def get_model_and_mask_parameters(model, model_parameters=[], mask_parameters=set(), wrapped_modules=[]):
    named_children = list(model.named_children())
    for module_subindex in range(len(named_children)):
        _, module = named_children[module_subindex]
        if isinstance(module, nn.Module):
            if isinstance(module, FCStructuredPruningWrapper) or isinstance(module, ConvStructuredPruningWrapper):
                mask_parameters.add(
                    module.shared_pruning_weights[module.numeric_id]["salience_threshold"])
                mask_parameters.add(
                    module.shared_pruning_weights[module.numeric_id]["channel_masks"])
                wrapped_modules.append(module)
            else:
                model_parameters.extend([*module.parameters(recurse=False)])

            # Recurse over the child modules
            get_model_and_mask_parameters(
                module, model_parameters, mask_parameters, wrapped_modules)

    return model_parameters, mask_parameters, wrapped_modules
