import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import subprocess
import re
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# For GNNs
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

scale = 1
design = '/s208.v'
netlist = 'mod_netlist' + design

def replace_specific_text_in_file(file_path, replacement_text):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace text that starts with '/s' and ends with '.v' with the replacement text
    updated_content = re.sub(r'/s.*?\.v', replacement_text, content)

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)

    print(f"Text in {file_path} replaced successfully.")

replace_specific_text_in_file('script/fanScripts/fault_atpg.script', design)
replace_specific_text_in_file('script/fanScripts/fault_sim.script', design)
print("Running fault ATPG script...")
subprocess.check_call(['/bin/bash', '-c', './bin/opt/fan -f ./script/fanScripts/fault_atpg.script'])

def copy_patterns(f1, f2):
    with open(f1, 'r') as file1, open(f2, 'w') as file2:
        for line in file1:
            file2.write(line)
            if '_pattern_2 ' in line:
                break
    print(f"Copied patterns from {f1} to {f2}")

copy_patterns('pat/atpg_sim.pat', 'pat/fault_sim.pat')

script = './bin/opt/fan -f ./script/fanScripts/fault_sim.script'
print("Running fault simulation script...")
subprocess.check_call(['/bin/bash', '-c', script])
pat_file, report = 'pat/fault_sim.pat', 'rpt/fault_sim.rpt'

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def flip_bit(vector, id):
    print(f"flip_bit called with vector: {vector}, id: {id}")
    patt = ''
    for i in range(len(vector)):
        if i == id:
            patt += str(1 - int(vector[i]))
        else:
            patt += str(vector[i])
    print(f"Resulting pattern after flip: {patt}")
    return patt

def parse_netlist(netlist_file):
    """
    Parses the netlist file and extracts gates, inputs, outputs, and connections.

    Returns:
        nodes: A list of nodes (gates) with their attributes.
        edges: A list of connections between gates.
        inputs: A list of primary input names.
        outputs: A list of primary output names.
        flip_flops: A list of flip-flop gate names.
        gate_name_to_index: A mapping from gate names to indices.
    """
    print(f"Parsing netlist file: {netlist_file}")
    nodes = []
    edges = []
    inputs = []
    outputs = []
    flip_flops = []
    gate_index = 0
    gate_name_to_index = {}

    with open(netlist_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('//'):
            continue

        # Extract module ports (if needed)
        module_match = re.match(r'module\s+\w+\s*\((.*)\);', line)
        if module_match:
            module_ports = module_match.group(1).replace(' ', '').split(',')
            continue  # Already processing inputs and outputs separately

        # Extract inputs
        if line.startswith('input'):
            input_match = re.findall(r'input\s+([\w, ]+);', line)
            if input_match:
                input_names = input_match[0].replace(' ', '').split(',')
                inputs.extend(input_names)
                # Create nodes for inputs
                for input_name in input_names:
                    if input_name not in gate_name_to_index:
                        gate_name_to_index[input_name] = gate_index
                        nodes.append({'name': input_name, 'type': 'input', 'index': gate_index})
                        gate_index += 1

        # Extract outputs
        elif line.startswith('output'):
            output_match = re.findall(r'output\s+([\w, ]+);', line)
            if output_match:
                output_names = output_match[0].replace(' ', '').split(',')
                outputs.extend(output_names)
                # Create nodes for outputs
                for output_name in output_names:
                    if output_name not in gate_name_to_index:
                        gate_name_to_index[output_name] = gate_index
                        nodes.append({'name': output_name, 'type': 'output', 'index': gate_index})
                        gate_index += 1

        # Extract gates
        else:
            # Match gate instances, e.g., INVX1 U_II6( .A(II104), .Y(II6) );
            gate_match = re.match(r'(\w+)\s+(\w+)\s*\((.*)\);', line)
            if gate_match:
                gate_type = gate_match.group(1)
                gate_name = gate_match.group(2)
                gate_ports = gate_match.group(3)

                # Save gate information
                gate_name_to_index[gate_name] = gate_index
                nodes.append({'name': gate_name, 'type': gate_type, 'index': gate_index})
                gate_index += 1

                # Check if it's a flip-flop
                if gate_type.startswith('SDFF'):
                    flip_flops.append(gate_name)

                # Extract connections
                port_connections = re.findall(r'\.(\w+)\(([\w_]+)\)', gate_ports)
                for port, signal in port_connections:
                    # Create nodes for signals if they don't exist
                    if signal not in gate_name_to_index:
                        gate_name_to_index[signal] = gate_index
                        nodes.append({'name': signal, 'type': 'net', 'index': gate_index})
                        gate_index += 1

                    # Determine edge direction based on port
                    if port in ['A', 'B', 'C', 'D', 'SI', 'SE', 'CK', 'SD', 'R', 'S']:  # Input ports
                        edges.append((gate_name_to_index[signal], gate_name_to_index[gate_name]))
                    elif port in ['Y', 'Q', 'QN', 'Z', 'SO']:  # Output ports
                        edges.append((gate_name_to_index[gate_name], gate_name_to_index[signal]))
                    else:
                        # For other ports, determine direction or ignore
                        pass

            # Handle assign statements
            elif line.startswith('assign'):
                assign_match = re.match(r'assign\s+(\w+)\s*=\s*(\w+);', line)
                if assign_match:
                    left_signal = assign_match.group(1)
                    right_signal = assign_match.group(2)

                    # Ensure nodes exist for the signals
                    for signal in [left_signal, right_signal]:
                        if signal not in gate_name_to_index:
                            gate_name_to_index[signal] = gate_index
                            nodes.append({'name': signal, 'type': 'net', 'index': gate_index})
                            gate_index += 1

                    # Create an edge from right_signal to left_signal
                    edges.append((gate_name_to_index[right_signal], gate_name_to_index[left_signal]))

    print(f"Total gates (nodes): {len(nodes)}")
    print(f"Total connections (edges): {len(edges)}")
    print(f"Primary inputs: {inputs}")
    print(f"Primary outputs: {outputs}")
    print(f"Flip-flops: {flip_flops}")
    return nodes, edges, inputs, outputs, flip_flops, gate_name_to_index

def build_circuit_graph(nodes, edges):
    """
    Builds a PyTorch Geometric Data object representing the circuit graph.

    Parameters:
        nodes: List of nodes with attributes.
        edges: List of connections between nodes.

    Returns:
        data: A PyTorch Geometric Data object.
    """
    print("Building circuit graph...")
    # Create node features
    gate_types = set(node['type'] for node in nodes)
    gate_type_to_id = {gt: idx for idx, gt in enumerate(gate_types)}
    
    # Initialize node fan-in and fan-out counts
    node_fan_in = {node['index']: 0 for node in nodes}
    node_fan_out = {node['index']: 0 for node in nodes}
    
    # Compute fan-in and fan-out counts
    for src, dst in edges:
        node_fan_out[src] += 1
        node_fan_in[dst] += 1
    
    # Compute topological depth
    # For simplicity, we can assign depth based on distance from inputs
    node_depth = {node['index']: -1 for node in nodes}
    
    # First, find input nodes
    input_indices = [node['index'] for node in nodes if node['type'] == 'input']
    
    # Initialize depth of input nodes to 0
    for idx in input_indices:
        node_depth[idx] = 0
    
    # Perform BFS to assign depths
    from collections import deque
    queue = deque(input_indices)
    while queue:
        current = queue.popleft()
        current_depth = node_depth[current]
        # For each neighbor, update depth if not already set
        for src, dst in edges:
            if src == current:
                if node_depth[dst] == -1 or node_depth[dst] > current_depth + 1:
                    node_depth[dst] = current_depth + 1
                    queue.append(dst)
    
    # Find maximum values for normalization
    max_fan_in = max(node_fan_in.values()) if node_fan_in.values() else 1
    max_fan_out = max(node_fan_out.values()) if node_fan_out.values() else 1
    max_depth = max(node_depth.values()) if node_depth.values() else 1

    # Create node features
    x = []
    for node in nodes:
        # One-hot encode gate type
        gate_type_id = gate_type_to_id[node['type']]
        gate_type_one_hot = [0] * len(gate_type_to_id)
        gate_type_one_hot[gate_type_id] = 1
        
        # Add fan-in and fan-out counts
        fan_in = node_fan_in[node['index']]
        fan_out = node_fan_out[node['index']]
        
        # Add topological depth
        depth = node_depth[node['index']]
        
        # Normalize features
        fan_in_norm = fan_in / max_fan_in if max_fan_in > 0 else 0
        fan_out_norm = fan_out / max_fan_out if max_fan_out > 0 else 0
        depth_norm = depth / max_depth if max_depth > 0 else 0
        
        # Combine all features
        features = gate_type_one_hot + [fan_in_norm, fan_out_norm, depth_norm]
        
        x.append(features)
    
    x = torch.tensor(x, dtype=torch.float)
    print(f"Node feature matrix x shape: {x.shape}")
    
    # Create edge index
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    print(f"Edge index shape: {edge_index.shape}")
    
    data = Data(x=x, edge_index=edge_index)
    print("Circuit graph built successfully.")
    return data

def find_ff_inputs(netlist_file):
    """
    Find flip-flop inputs from the netlist.

    Returns:
        dffs: List of flip-flop output names (Q outputs).
    """
    print(f"Finding flip-flop inputs from netlist: {netlist_file}")
    with open(netlist_file, 'r') as f:
        dffs = []
        for line in f:
            line = line.strip()
            if line.startswith('SDFF'):
                # Extract the Q output signal
                q_match = re.search(r'\.Q\(([\w_]+)\)', line)
                if q_match:
                    q_signal = q_match.group(1)
                    dffs.append(q_signal)
    print(f"Flip-flop outputs found: {dffs}")
    return dffs

def extract_values(logic_report):
    print(f"Extracting values from logic report: {logic_report}")
    with open(logic_report) as f:
        values = {}
        flag = 0
        for line in f:
            gate = line.strip().split()
            if len(gate) > 2 and 'frame' in line:
                flag = 1
                key = gate[2]
            if 'good:' in gate and flag == 1:
                flag = 0  # Corrected assignment operator
                value = gate[-1].split('X')[-1][0]
                values[key] = value
    print(f"Values extracted: {values}")
    return values

def pat_num_change(file_name):
    print(f"Updating pattern number in {file_name}")
    with open(file_name, 'r') as file:
        # read a list of lines into data
        data = file.read()
    with open(file_name, 'r') as file:
        for line in file:
            if '_num_of_pattern_' in line:
                k = line.split('_')
                num_pat = int(k[-1]) + 1
                k[-1] = str(num_pat)
                k = '_'.join(k) + '\n'
                break
        data = data.replace(line, k)
    with open(file_name, 'w') as file:
        file.write(data)
    print(f"Pattern number updated in {file_name}")

def Write_new_pattern(file_path, pattern, dff):
    print(f"Write_new_pattern called with file_path: {file_path}, pattern: {pattern}, dff: {dff}")
    with open(file_path, 'r+') as f:
        lines = f.readlines()
        last_line = lines[-1]
        last_line = last_line.strip()
        last_line_parts = last_line.split('_')
        pat = last_line_parts[2].split(' ')

        num_pat = int(pat[0]) + 1
        pat[0] = str(num_pat)
        pat[1] = pattern
        pat[5] = dff
        last_line_parts[2] = ' '.join(pat)
        new_line = '_'.join(last_line_parts)
        f.write('\n' + new_line)
    pat_num_change(file_path)
    print(f"Text appended to {file_path} successfully.")

def find_dff_in(curr_state, dffs):
    print(f"Finding DFF input states. DFFs: {dffs}")
    dff_in = ''
    for dff in dffs:
        if dff in curr_state:
            dff_in += curr_state[dff]
        else:
            dff_in += '0'  # Default value if not found
    print(f"DFF input state: {dff_in}")
    return dff_in

def read_coverage(report):
    print(f"Reading coverage from report: {report}")
    with open(report) as f:
        for line in f:
            if 'fault coverage' in line:
                coverage = float(line.split(' ')[-1][:-2])
                print(f"Fault coverage found: {coverage}")
                return coverage

def check_coverage(netlist_file, pat_file, pattern, report):
    print(f"check_coverage called with netlist: {netlist_file}, pat_file: {pat_file}, pattern: {pattern}, report: {report}")
    ff_in = find_ff_inputs(netlist_file)
    curr_state = extract_values('logic_sim_report.txt')
    dff_state = find_dff_in(curr_state, ff_in)
    Write_new_pattern(pat_file, pattern, dff_state)
    print("Running simulation script...")
    subprocess.check_call(['/bin/bash', '-c', script])
    coverage = read_coverage(report)
    return coverage, curr_state

from gym import Env, spaces

class CircuitEnv(Env):
    def __init__(self):
        print("Initializing CircuitEnv")
        # Parse netlist and build graph
        netlist_file = netlist
        nodes, edges, self.inputs, self.outputs, self.flip_flops, self.gate_name_to_index = parse_netlist(netlist_file)
        self.nodes = nodes  # Save nodes for later reference
        self.graph = build_circuit_graph(nodes, edges)
        self.num_nodes = self.graph.num_nodes
        self.coverage = 0
        self.cumulative_reward = 0
        self.max_steps = 20  # Maximum steps per episode
        self.current_step = 0  # Step counter

        # Define action space (number of primary inputs)
        self.action_space = spaces.Discrete(len(self.inputs))
        print(f"Action space: {self.action_space}")

        # Map input names to node indices
        self.input_name_to_index = {name: self.gate_name_to_index[name] for name in self.inputs}
        self.input_indices = list(self.input_name_to_index.values())

        # Initialize input node features (append a feature for input value)
        # Update node features to accommodate the additional input value feature
        num_additional_features = 1  # For the input value
        self.graph.x = torch.cat([self.graph.x, torch.zeros((self.num_nodes, num_additional_features), dtype=torch.float)], dim=1)
        for idx in self.input_indices:
            self.graph.x[idx, -1] = 0  # Initialize input values to 0

        self.state = self.graph

    def step(self, action):
        print(f"Step called with action: {action}")
        info = {}
        self.current_step += 1

        # Modify the input pattern based on the action
        input_idx = action % len(self.inputs)
        input_node_index = self.input_indices[input_idx]

        # Toggle the input node's feature (assuming the last position represents the input value)
        self.graph.x[input_node_index, -1] = 1 - self.graph.x[input_node_index, -1]
        print(f"Toggled input node {self.inputs[input_idx]} at index {input_node_index}")

        # Run simulation with the updated graph/state
        pattern = self.get_input_pattern_from_graph()
        coverage, curr_state = check_coverage(netlist, pat_file, pattern, report)

        # Compute reward
        if coverage > self.coverage:
            reward = (coverage - self.coverage) * scale
            self.cumulative_reward += reward
            print(f'Coverage increased. Reward: {reward}, Cumulative Reward: {self.cumulative_reward}')
        else:
            reward = -1 * scale
            self.cumulative_reward += reward
            print(f'No coverage increase. Reward: {reward}, Cumulative Reward: {self.cumulative_reward}')

        self.coverage = coverage
        self.state = self.graph

        # Check if episode is done
        done = False
        if self.coverage >= 100.0:
            done = True
            print("Maximum coverage reached. Episode terminated.")
        elif self.current_step >= self.max_steps:
            done = True
            print("Maximum steps reached. Episode terminated.")

        # Include coverage in the info dictionary
        info['coverage'] = self.coverage

        return self.state, reward, done, info

    def reset(self):
        print("Resetting environment")
        self.cumulative_reward = 0
        self.coverage = 0
        self.current_step = 0

        # Re-initialize the graph
        netlist_file = netlist
        nodes, edges, self.inputs, self.outputs, self.flip_flops, self.gate_name_to_index = parse_netlist(netlist_file)
        self.nodes = nodes
        self.graph = build_circuit_graph(nodes, edges)
        self.num_nodes = self.graph.num_nodes

        # Map input names to node indices
        self.input_name_to_index = {name: self.gate_name_to_index[name] for name in self.inputs}
        self.input_indices = list(self.input_name_to_index.values())

        # Initialize input node features (append a feature for input value)
        num_additional_features = 1  # For the input value
        self.graph.x = torch.cat([self.graph.x, torch.zeros((self.num_nodes, num_additional_features), dtype=torch.float)], dim=1)
        for idx in self.input_indices:
            self.graph.x[idx, -1] = 0  # Initialize input values to 0

        self.state = self.graph

        return self.state

    def get_input_pattern_from_graph(self):
        # Extract the input pattern from the graph's node features
        input_pattern = ''
        for idx in self.input_indices:
            node_feature = self.graph.x[idx]
            # Assuming the last feature represents the input value
            value = int(node_feature[-1].item())
            input_pattern += str(value)
        print(f"Input pattern from graph: {input_pattern}")
        return input_pattern

class DQN(nn.Module):
    def __init__(self, num_node_features, num_actions):
        super(DQN, self).__init__()
        print(f"Initializing DQN with {num_node_features} node features and {num_actions} actions")
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.lin = nn.Linear(128, num_actions)

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        output = self.lin(x)
        return output

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        print(f"ReplayMemory initialized with capacity {capacity}")

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        # print(f"Transition saved. Memory size: {len(self.memory)}")

    def sample(self, batch_size):
        # print(f"Sampling {batch_size} transitions")
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

optimizer = None
memory = None
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    print(f"Selecting action at step {steps_done} with eps_threshold {eps_threshold}")
    if sample > eps_threshold:
        with torch.no_grad():
            # Create a batch with a single graph
            batch_state = Batch.from_data_list([state]).to(device)
            q_values = policy_net(batch_state)
            action = q_values.max(1)[1].view(1, 1)
            print(f"Action selected by policy: {action.item()}")
            return action
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        print(f"Random action selected: {action.item()}")
        return action

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        print(f"Not enough transitions to optimize. Current memory size: {len(memory)}")
        return
    print("Optimizing model")
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare the batch data
    batch_states = Batch.from_data_list([data for data in batch.state]).to(device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    if non_final_next_states_list:
        batch_next_states = Batch.from_data_list(non_final_next_states_list).to(device)
    else:
        batch_next_states = None

    batch_actions = torch.cat(batch.action).to(device)
    batch_rewards = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a)
    state_action_values = policy_net(batch_states).gather(1, batch_actions)

    # Compute V(s_{t+1})
    next_state_values = torch.zeros(len(batch_rewards), device=device)
    if batch_next_states is not None:
        with torch.no_grad():
            next_state_values_batch = target_net(batch_next_states)
            next_state_values[non_final_mask] = next_state_values_batch.max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + batch_rewards

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)
    print(f"Loss computed: {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    env = CircuitEnv()
    state = env.reset()
    num_node_features = state.num_node_features
    n_actions = env.action_space.n

    policy_net = DQN(num_node_features, n_actions).to(device)
    target_net = DQN(num_node_features, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 1
    else:
        num_episodes = 5

    for i_episode in range(num_episodes):
        print(f"Starting episode {i_episode}")
        state = env.reset()
        for t in range(env.max_steps):
            print(f"Time step {t}")
            action = select_action(state)
            next_state, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, next_state if not done else None, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()