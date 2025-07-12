import torch
import torch.nn as nn
import numpy as np
 
class EarShotModel(nn.Module):
    def __init__(self):
        super(EarShotModel, self).__init__()

        self.activation = nn.Sigmoid()
        self.loss_fn = nn.BCELoss(reduction='none')
        self.input_layer = nn.LSTM(256, 512, batch_first=True, num_layers=1)
        self.output_layer = nn.Linear(512, 300)

    def forward(self, x, hidden=None):
        x, hidden = self.input_layer(x, hidden)
        x = self.activation(self.output_layer(x))
        return x 

# Sigmoid and tanh activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Manual LSTM cell computation for one time step
def lstm_cell(x_t, h_prev, c_prev, kernel, recurrent_kernel, bias):
    """
    Args:
        x_t: Input at current time step, shape (batch_size, input_size)
        h_prev: Previous hidden state, shape (batch_size, hidden_size)
        c_prev: Previous cell state, shape (batch_size, hidden_size)
        kernel: Input-to-hidden weights, shape (input_size, 4*hidden_size)
        recurrent_kernel: Hidden-to-hidden weights, shape (hidden_size, 4*hidden_size)
        bias: Bias, shape (4*hidden_size)
    
    Returns:
        h_t: New hidden state, shape (batch_size, hidden_size)
        c_t: New cell state, shape (batch_size, hidden_size)
    """
    # Compute gate activations
    z = x_t @ kernel  # [1000, 256] @ [256, 2048] = [1000, 2048]
    z += h_prev @ recurrent_kernel  # [1000, 512] @ [512, 2048] = [1000, 2048]
    z += bias
    
    # Split into gates (input, forget, cell, output)
    hidden_size = h_prev.shape[-1]
    i_t = sigmoid(z[:, :hidden_size])  # Input gate
    g_t = tanh(z[:, hidden_size:2*hidden_size])  # Cell gate
    f_t = sigmoid(z[:, 2*hidden_size:3*hidden_size] + 1) # Forget gate
    o_t = sigmoid(z[:, 3*hidden_size:])  # Output gate
    
    # Update cell and hidden states
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * tanh(c_t)
    
    return h_t, c_t

# Manual LSTM sequence computation
def lstm_sequence(inputs, kernel, recurrent_kernel, bias, h_init=None, c_init=None):
    """
    Args:
        inputs: Input sequence, shape (batch_size, seq_len, input_size)
        kernel: Input-to-hidden weights, shape (input_size, 4*hidden_size)
        recurrent_kernel: Hidden-to-hidden weights, shape (hidden_size, 4*hidden_size)
        bias: Bias, shape (4*hidden_size)
        h_init: Initial hidden state, shape (batch_size, hidden_size) (optional)
        c_init: Initial cell state, shape (batch_size, hidden_size) (optional)
    
    Returns:
        outputs: Hidden states for all time steps, shape (batch_size, seq_len, hidden_size)
    """
    batch_size, seq_len, input_size = inputs.shape
    hidden_size = recurrent_kernel.shape[0]
    
    # Initialize hidden and cell states
    h_t = h_init if h_init is not None else np.zeros((batch_size, hidden_size))
    c_t = c_init if c_init is not None else np.zeros((batch_size, hidden_size))
    
    outputs = []
    
    # Process each time step
    for t in range(seq_len):
        x_t = inputs[:, t, :]
        h_t, c_t = lstm_cell(x_t, h_t, c_t, kernel, recurrent_kernel, bias)
        outputs.append(h_t)
    
    # Stack outputs to shape (batch_size, seq_len, hidden_size)
    return np.stack(outputs, axis=1)


def swap_weights(weight, fgate=False):
    with torch.no_grad():
        if len(weight.shape) == 2:
            slice1 = weight[:, 512:1024].clone()
            slice2 = weight[:, 1024:1536].clone()
            weight[:, 512:1024].copy_(slice2)
            weight[:, 1024:1536].copy_(slice1)
        else:
            slice1 = weight[512:1024].clone()
            slice2 = weight[1024:1536].clone()
            weight[512:1024].copy_(slice2)
            weight[1024:1536].copy_(slice1)
            if fgate:
                weight[512:1024] += 1.0
    return weight

if __name__ == "__main__":
    model = EarShotModel()

    # Load TensorFlow weights
    tf_weight = np.load("/home/fie24002/EARSHOT_ZERO/tf_weights.npz", allow_pickle=False)
    lstm_weight_ih = tf_weight['rnn/lstm_cell/kernel'].astype(np.float32) # (256, 2048)
    lstm_weight_hh = tf_weight['rnn/lstm_cell/recurrent_kernel'].astype(np.float32) # (512, 2048)
    lstm_bias_ih = tf_weight['rnn/lstm_cell/bias'].astype(np.float32)
    linear_weight = tf_weight['semantic_logits/kernel'].astype(np.float32)
    linear_bias = tf_weight['semantic_logits/bias'].astype(np.float32)
    
    # Load input and output data abtained from TensorFlow code
    inp = np.load("/home/fie24002/EARSHOT_ZERO/demoinp.npz", allow_pickle=False)
    out = np.load("/home/fie24002/EARSHOT_ZERO/demoout.npz", allow_pickle=False)
    print("Tensorflow LSTM output:")
    print(out["activation"][0,0,:10])

    # Run LSTM inference
    lstm_outputs = lstm_sequence(inp["acoustic"], lstm_weight_ih, lstm_weight_hh, lstm_bias_ih)
    
    # Apply linear layer and sigmoid activation (mimicking EarShotModel's output_layer)
    # Reshape lstm_outputs to (batch_size*seq_len, hidden_size) for linear layer
    lstm_outputs_flat = lstm_outputs.reshape(-1, 512)
    linear_output = lstm_outputs_flat @ linear_weight + linear_bias
    final_output = sigmoid(linear_output)
    
    # Reshape back to (batch_size, seq_len, outputther_size)
    final_output = final_output.reshape(1000, -1, 300)
    print("Hand written LSTM output:")
    print(final_output[0,0,:10])

    # Load weights into PyTorch model
    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}")

    with torch.no_grad(): 
        hidden_size = lstm_weight_ih.shape[1] // 4  # 2048 / 4 = 512
        # igfo -> ifco
        # ifco =  input, forget, cell, output gates
        model.input_layer.weight_ih_l0.copy_(swap_weights(torch.from_numpy(lstm_weight_ih)).T)
        model.input_layer.weight_hh_l0.copy_(swap_weights(torch.from_numpy(lstm_weight_hh)).T)
        model.input_layer.bias_hh_l0.copy_(swap_weights(torch.from_numpy(lstm_bias_ih), fgate=True))
        model.input_layer.bias_ih_l0.zero_()
        model.output_layer.weight.copy_(torch.from_numpy(linear_weight).T)
        model.output_layer.bias.copy_(torch.from_numpy(linear_bias))

    # Compute Torch LSTM output
    torch_input = torch.from_numpy(inp["acoustic"]).float()
    torch_output = model(torch_input)
    torch_output_np = torch_output.detach().numpy()
    print("Torch LSTM output:")
    print(torch_output_np[0,0,:10])
 
    