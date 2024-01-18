import torch
import torch.nn as nn
from typing import Tuple
from torch.fx import symbolic_trace
from pruning import ConvStructuredPruningWrapper, FCStructuredPruningWrapper


def may_modify_activation_size(node, traced_model):
    return node_is_module_type(node, nn.MaxPool2d, traced_model)


# Not all nodes within a tree are necessarily a module, some might also merely
# be an operation like an add or multiply.
def node_is_module_type(node, module_type, traced_model):
    modules = dict(traced_model.named_modules())
    if not isinstance(node, int):
        if node.op == "call_module":
            if node.target in modules:
                if isinstance(modules[node.target], module_type):
                    return True
    return False


# Came from https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L17
def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    target = target.replace("_", ".")
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def is_conv_and_bn(node, traced_model):
    if node_is_module_type(node, nn.Conv2d, traced_model):
        # Going to have to check this, not sure if the batchnorm is guaranteed
        # to be next up after a Conv within the tree.
        if node_is_module_type(node.next, nn.BatchNorm2d, traced_model):
            return True
    return False


def is_fc(node, traced_model):
    if node_is_module_type(node, nn.Linear, traced_model):
        return True
    return False


def is_prunable(node, traced_model):
    return is_conv_and_bn(node, traced_model) or is_fc(node, traced_model)


def get_id_from_prunable(node, traced_model):
    modules = dict(traced_model.named_modules())
    return modules[node.target].numeric_id


def is_wrapped(node, traced_model):
    if node_is_module_type(node, FCStructuredPruningWrapper, traced_model) or node_is_module_type(node, ConvStructuredPruningWrapper, traced_model):
        return True
    return False


def are_wrapped_module_types_the_same(node, other_node, traced_model):
    modules = dict(traced_model.named_modules())
    module = modules[node.target]
    if isinstance(module, FCStructuredPruningWrapper) and isinstance(module, FCStructuredPruningWrapper):
        return True
    elif isinstance(module, ConvStructuredPruningWrapper) and isinstance(module, ConvStructuredPruningWrapper):
        return True
    return False


def find_termination_node(start_node):
    # Check to see if all argument nodes to module have already been checked.
    def dependencies_met(module, active_at):
        for parent in module.args:
            if not isinstance(parent, int):
                if not parent.name in active_at:
                    return False
        return True

    # Check if all items within list1 are also present within list2.
    def all_in(list1, list2):
        for item in list1:
            if item not in list2:
                return False
        return True

    # Add all items in list1 that aren't already in list2 to list2.
    def add_not_already_present(list1, list2):
        for item in list1:
            if item not in list2:
                list2.append(item)
        return list2

    # If we've already visited a node or not.
    active_at = {}

    # Intermediate residual nodes from each branch.
    active = []

    # Queue since we're doing BFS.
    module_queue = [start_node]

    # Until we've exhausted all possible child nodes starting from the
    # residual block, keep looking for the termination node.
    while len(module_queue):
        module = module_queue.pop(0)

        # If all current module arguments are a subset of active modules, those
        # module arguments are "dead", i.e., their parameters aren't active
        # anymore and we can assume the space they occupy is now occupied by
        # the current module.
        if all_in(module.args, active):
            for item in module.args:
                active.remove(item)

        # We just visited this node since we're currently on it.
        active_at[module.name] = True

        # Add child nodes of this current node to the intermediate node list.
        active = add_not_already_present(module.users.keys(), active)

        # If the current node is in the currently active list, we should remove
        # it since we're moving onto the child nodes.
        if module in active:
            active.remove(module)

        # If there's only one remaining active node, that means we've visited
        # all the intermediate residual nodes and are at the termination node.
        if len(active) == 1:
            return active[0]

        # Begin checking the children of the current node.
        for child in module.users.keys():
            # If we've already visited all the arguments to the child node,
            # we can add it into the queue of modules to check.
            if dependencies_met(child, active_at):
                module_queue.append(child)

    raise Exception(
        "Unable to find suitable termination node for residual block start {}".format(start_node))


def check_backwards_for_final_prunable_nodes_in_residual_block(termination_node, start_node, traced_model, prunable_nodes):
    backwards_prunable_nodes = []

    if termination_node != start_node:

        # Check each possible residual path going into the termination node.
        for res_path_node in termination_node.args:
            if not isinstance(res_path_node, int):
                # If it's a prunable node let's save it.
                if is_fc(res_path_node, traced_model) or is_conv_and_bn(res_path_node, traced_model):
                    backwards_prunable_nodes.append(res_path_node.name)

                # If there's only 1 argument into this module, and we found a
                # prunable node, we've already found the final prunable node on
                # this path.
                if len(termination_node.args) == 1 and len(backwards_prunable_nodes):
                    return backwards_prunable_nodes

                # Check the next path and save any prunable nodes it finds.
                backwards_prunable_nodes = backwards_prunable_nodes + \
                    check_backwards_for_final_prunable_nodes_in_residual_block(
                        res_path_node, start_node, traced_model, prunable_nodes)

    return backwards_prunable_nodes


def check_backwards_for_final_prunable_parent_nodes(child_node, traced_model, modifies_activation_size=[], start_node=None):
    backwards_same_prunable_nodes = []
    backwards_diff_prunable_nodes = []

    if start_node == None:
        start_node = child_node

    # Check each possible residual path going into the termination node.
    for res_path_node in child_node.args:
        if not isinstance(res_path_node, int):
            # If it's a prunable node let's save it.
            if is_wrapped(res_path_node, traced_model):
                if are_wrapped_module_types_the_same(res_path_node, start_node, traced_model):
                    backwards_same_prunable_nodes.append(
                        get_id_from_prunable(res_path_node, traced_model))
                else:
                    backwards_diff_prunable_nodes.append(
                        {
                            "id": get_id_from_prunable(res_path_node, traced_model),
                            "modifies_activation_size": modifies_activation_size
                        }
                    )
            # Keep track of any nodes other than prunable nodes that modify
            # activation size.
            elif may_modify_activation_size(res_path_node, traced_model):
                modules = dict(traced_model.named_modules())
                results = check_backwards_for_final_prunable_parent_nodes(
                    res_path_node, traced_model, modifies_activation_size + [modules[res_path_node.target]], start_node)
                backwards_same_prunable_nodes.extend(results[0])
                backwards_diff_prunable_nodes.extend(results[1])
            # Check the next path and save any prunable nodes it finds.
            else:
                results = check_backwards_for_final_prunable_parent_nodes(
                    res_path_node, traced_model, modifies_activation_size, start_node)
                backwards_same_prunable_nodes.extend(results[0])
                backwards_diff_prunable_nodes.extend(results[1])

    return backwards_same_prunable_nodes, backwards_diff_prunable_nodes


def find_prunable_parents_for_all_nodes(traced_model, num_outputs):
    num_outputs_found = 0
    modules = dict(traced_model.named_modules())
    for node in traced_model.graph.nodes:
        if is_wrapped(node, traced_model):
            prior_prunables, prior_diff_prunables = check_backwards_for_final_prunable_parent_nodes(
                node, traced_model)
            if len(prior_diff_prunables) > 0:
                for diff_prunable in prior_diff_prunables:
                    wrappers = modules[node.target].shared_pruning_weights[diff_prunable["id"]]["wrapper_list"]
                    for wrapper in wrappers:
                        if not (node_is_module_type(node, FCStructuredPruningWrapper, traced_model) and isinstance(wrapper, ConvStructuredPruningWrapper)):
                            wrapper.should_prune_outputs = False
            modules[node.target].prior_prunables = prior_prunables
            modules[node.target].prior_diff_prunables = prior_diff_prunables

            if node.next.op == "output":
                modules[node.target].should_prune_outputs = False
                assert num_outputs >= num_outputs_found


# Insert the appropriate structured pruning wrapper.
def replace_prunable_nodes_with_wrapper(traced_model, prunable_node_shares_with, residual_nodes_info, starting_salience_threshold, accelerator_device):
    id = 1
    global_counter = [0]
    shared_pruning_weights = {}
    traced_model_named_mods = dict(traced_model.named_modules())
    for node in traced_model.graph.nodes:
        prunable_node_id = id
        if node.name in prunable_node_shares_with:
            shares_mask_with = prunable_node_shares_with[node.name]
            # If we haven't yet assigned an ID to the weights that a
            # StructuredPruningWrapper will use or share with other
            # StructuredPruningWrapper's, assign a new id.
            if shares_mask_with["shared_id"] <= 0:
                shares_mask_with["shared_id"] = prunable_node_id
            # If the weights already have an ID associated, grab that.
            else:
                prunable_node_id = shares_mask_with["shared_id"]

        if is_conv_and_bn(node, traced_model):
            # If there isn't any metadata assigned to the prunable node's ID
            # yet, create a metadata dict and assign it.
            if prunable_node_id not in shared_pruning_weights:
                shared_pruning_weights[prunable_node_id] = {
                    "channel_masks": None,
                    "layerwise_width_multiplier": torch.tensor([1.0], requires_grad=True, device=accelerator_device),
                    "salience_threshold": torch.tensor([starting_salience_threshold], requires_grad=True, device=accelerator_device),
                    "last_iteration": -1,
                    "wrapper_list": []
                }

            # Grab the layers we want to wrap and then wrap them in a
            # ConvStructuredPruningWrapper.
            conv_layer = traced_model_named_mods[node.target]
            bn_layer = traced_model_named_mods[node.next.target]
            wrapper = ConvStructuredPruningWrapper(
                prunable_node_id, [node.name, node.next.name], shared_pruning_weights, global_counter, residual_nodes_info, conv_layer, bn_layer, accelerator_device)

            # Save the ConvStructuredPruningWrapper into the dict of all
            # StructuredPruningWrapper's indexed by ID.
            shared_pruning_weights[prunable_node_id]["wrapper_list"].append(
                wrapper)

            # Replace the old Conv layer with the wrapper.
            traced_model_named_mods[node.target] = wrapper
            parent_name, name = _parent_name(node.name)
            setattr(traced_model_named_mods[parent_name], name, wrapper)

            # Replace the old BatchNorm2d layer with an Identity function.
            fake_bn = nn.Identity()
            traced_model_named_mods[node.next.target] = fake_bn
            bn_parent_name, bn_name = _parent_name(node.next.name)
            setattr(traced_model_named_mods[bn_parent_name], bn_name, fake_bn)

            id = id + 1

        elif is_fc(node, traced_model):
            # If there isn't any metadata assigned to the prunable node's ID
            # yet, create a metadata dict and assign it.
            if prunable_node_id not in shared_pruning_weights:
                shared_pruning_weights[prunable_node_id] = {
                    "channel_masks": None,
                    "layerwise_width_multiplier": torch.tensor([1.0], requires_grad=True, device=accelerator_device),
                    "salience_threshold": torch.tensor([starting_salience_threshold], requires_grad=True, device=accelerator_device),
                    "last_iteration": -1,
                    "wrapper_list": []
                }

            # Save the old FCStructuredPruningWrapper into the dict of all
            # StructuredPruningWrapper's indexed by ID.
            fc_layer = traced_model_named_mods[node.target]
            wrapper = FCStructuredPruningWrapper(
                prunable_node_id, [node.name], shared_pruning_weights, global_counter, residual_nodes_info, fc_layer, accelerator_device)
            shared_pruning_weights[prunable_node_id]["wrapper_list"].append(
                wrapper)

            # Replace the old FC layer with the wrapper.
            traced_model_named_mods[node.target] = wrapper
            parent_name, name = _parent_name(node.name)
            setattr(traced_model_named_mods[parent_name], name, wrapper)

            id = id + 1

    return traced_model, global_counter


def find_residual_nodes(traced_model, prunable_nodes):
    residual_nodes = {}
    prunable_node_shares_with = {}
    for node in traced_model.graph.nodes:
        if len(node.users) > 1:
            terminates_at = find_termination_node(node)

            # Find the last prunable nodes prior to the end of the residual
            # block.
            final_prunable_nodes = check_backwards_for_final_prunable_nodes_in_residual_block(
                terminates_at, node, traced_model, prunable_nodes)

            # We'll need to save the list of those last prunable nodes 
            # specifically for use in the wrappers, so that we can maintain 
            # shared pruning masks across wrappers that will be combined at
            # the end of this residual block.
            final_prunable_nodes_metadata = {
                "nodes": final_prunable_nodes,
                "shared_id": 0
            }

            # Clone the prunable node list we just found across every single 
            # one of the nodes in the list within a shared hashmap, as they'll 
            # need the info within the list so they can access the shared 
            # pruning mask and calculate the # of parameters still active using
            # the mask applied to the wrapped node immediately prior.
            for prunable_node in final_prunable_nodes:
                if prunable_node not in prunable_node_shares_with:
                    prunable_node_shares_with[prunable_node] = final_prunable_nodes_metadata

            # Save the metadata for this residual block we've found.
            residual_nodes[node.name] = {
                "self": node,
                "terminates_at": terminates_at,
                "final_prunable_nodes": final_prunable_nodes
            }

    return residual_nodes, prunable_node_shares_with


# Find prunable nodes so they can later on get wrapped.
def find_prunable_nodes(traced_model):
    prunable_nodes = {}
    prior_prunables = []
    for node in traced_model.graph.nodes:
        if is_conv_and_bn(node, traced_model):
            prunable_nodes[node.name] = {
                "type": "conv_bn",
                "nodes": [node, node.next],
                "prior_prunables": prior_prunables.copy()
            }
            prior_prunables.append([node, node.next])
        elif is_fc(node, traced_model):
            prunable_nodes[node.name] = {
                "type": "fc",
                "nodes": [node],
                "prior_prunables": prior_prunables.copy()
            }
            prior_prunables.append([node])
    return prunable_nodes


def trace_and_hook_model(model, num_outputs, starting_salience_threshold, accelerator_device):
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)

    # Find all prunable nodes.
    prunable_nodes = find_prunable_nodes(symbolic_traced)

    # Find any residual blocks.
    residual_start_nodes, prunable_node_shares_with = find_residual_nodes(
        symbolic_traced, prunable_nodes)
    # Wrap any prunable nodes.
    traced_model, global_counter = replace_prunable_nodes_with_wrapper(
        symbolic_traced, prunable_node_shares_with, residual_start_nodes, starting_salience_threshold, accelerator_device)

    # Find the prunable parents of all prunable nodes.
    find_prunable_parents_for_all_nodes(traced_model, num_outputs)

    # Recompile the graph
    traced_model.graph.eliminate_dead_code()
    traced_model.graph.lint()
    traced_model.recompile()
    return symbolic_traced, global_counter
