def calculate_sparse_coding_inference_flops(M, N, num_samples, learn_D=False):
    """
    Calculate the number of FLOPs for inference in SparseCoding.
    
    Args:
    M (int): Number of measurements
    N (int): Number of sparse sources
    num_samples (int): Number of samples to perform inference on
    learn_D (bool): Whether D is being learned or not
    
    Returns:
    int: Total number of FLOPs for inference
    """
    flops = 0
    
    if learn_D:
        # Normalization of D_
        flops += M * N * 2  # Norm calculation
        flops += M * N     # Division
    
    # Exponential of log_S_
    flops += N * num_samples
    
    # Matrix multiplication (S_ @ D_)
    flops += 2 * M * N * num_samples
    
    return flops

def calculate_sparse_coding_training_flops(M, N, num_samples, num_steps, learn_D=False, batch_size=None):
    """
    Calculate the number of FLOPs for training SparseCoding.
    
    Args:
    M (int): Number of measurements
    N (int): Number of sparse sources
    num_samples (int): Total number of training samples
    num_steps (int): Number of training steps
    learn_D (bool): Whether D is being learned or not
    batch_size (int, optional): Batch size for training. If None, full batch is assumed.
    
    Returns:
    int: Total number of FLOPs for training
    """
    if batch_size is None:
        batch_size = num_samples
    
    # Forward pass FLOPs (same as inference)
    forward_flops = calculate_sparse_coding_inference_flops(M, N, 1, learn_D)
    
    # Loss calculation
    loss_flops = M * batch_size * 2  # MSE
    loss_flops += N * batch_size     # L1
    
    # Backward pass FLOPs (approximation, actual might vary based on autograd)
    backward_flops = forward_flops * 2  # Roughly twice the forward pass
    
    # Parameter updates
    update_flops = N * batch_size
    if learn_D:
        update_flops += M * N
    
    # Total FLOPs for one training iteration
    flops_per_iteration = forward_flops + loss_flops + backward_flops + update_flops
    
    # Calculate effective number of iterations
    effective_iterations = num_steps * (batch_size / num_samples)
    
    # Total FLOPs for all training
    total_flops = int(flops_per_iteration * num_samples * effective_iterations)
    
    return total_flops

# Example usage
M, N = 8, 16
num_samples = 1024
train_flops = calculate_sparse_coding_training_flops(M, N, num_samples, 20000, learn_D=True)
print(f"Total FLOPs for training: {train_flops}")
inference_flops = calculate_sparse_coding_inference_flops(M, N, num_samples, learn_D=True)
print(f"Total FLOPs for inference: {inference_flops}")

def calculate_sae_training_flops(M, N, num_samples, num_steps, learn_D=False, batch_size=None):
    """
    Calculate the number of FLOPs for training an SAE.
    
    Args:
    M (int): Number of input features
    N (int): Number of output features (neurons in the hidden layer)
    num_samples (int): Total number of training samples
    num_steps (int): Number of training steps
    learn_D (bool): Whether the decoder weights are being learned
    batch_size (int, optional): Batch size for training. If None, full batch is assumed.
    
    Returns:
    int: Total number of FLOPs for training
    """
    if batch_size is None:
        batch_size = num_samples
    
    # Forward pass FLOPs (including decoder)
    forward_flops = 2 * M * N + N + 2 * N * M  # Encoder + ReLU + Decoder
    if learn_D:
        forward_flops += M * N  # Normalization of D
    
    # Backward pass FLOPs
    relu_grad_flops = N  # ReLU gradient
    linear_grad_flops = (2 * N * M) + N  # Linear layer gradient
    decoder_grad_flops = 2 * N * M  # Decoder gradient
    
    # Weight update FLOPs
    weight_update_flops = 2 * (M * N + N)  # Encoder weight update
    if learn_D:
        weight_update_flops += 2 * N * M  # Decoder weight update
    
    backward_flops = relu_grad_flops + linear_grad_flops + decoder_grad_flops + weight_update_flops
    
    # Total FLOPs for one training iteration (forward + backward)
    flops_per_iteration = forward_flops + backward_flops
    
    # Calculate effective number of iterations
    effective_iterations = num_steps * (batch_size / num_samples)
    
    # Total FLOPs for all training
    total_flops = int(flops_per_iteration * num_samples * effective_iterations)
    
    return total_flops

def calculate_mlp_training_flops(M, h, N, num_samples, num_steps, learn_D=False, batch_size=None):
    """
    Calculate the number of FLOPs for training an MLP.
    
    Args:
    M (int): Number of input features
    h (int): Number of neurons in the hidden layer
    N (int): Number of output features
    num_samples (int): Total number of training samples
    num_steps (int): Number of training steps
    learn_D (bool): Whether the decoder weights are being learned
    batch_size (int, optional): Batch size for training. If None, full batch is assumed.
    
    Returns:
    int: Total number of FLOPs for training
    """
    if batch_size is None:
        batch_size = num_samples
    
    # Forward pass FLOPs (including decoder)
    forward_flops = 2 * M * h + h + 2 * h * N + N + 2 * N * M  # First layer + ReLU + Second layer + ReLU + Decoder
    if learn_D:
        forward_flops += M * N  # Normalization of D
    
    # Backward pass FLOPs
    relu2_grad_flops = N
    linear2_grad_flops = (2 * N * h) + N
    relu1_grad_flops = h
    linear1_grad_flops = (2 * M * h) + h
    decoder_grad_flops = 2 * N * M
    
    # Weight update FLOPs
    weight_update_flops = 2 * (M * h + h + h * N + N)  # Encoder weight update
    if learn_D:
        weight_update_flops += 2 * N * M  # Decoder weight update
    
    # Total backward pass FLOPs
    backward_flops = relu2_grad_flops + linear2_grad_flops + relu1_grad_flops + linear1_grad_flops + decoder_grad_flops + weight_update_flops
    
    # Total FLOPs for one training iteration (forward + backward)
    flops_per_iteration = forward_flops + backward_flops
    
    # Calculate effective number of iterations
    effective_iterations = num_steps * (batch_size / num_samples)
    
    # Total FLOPs for all training
    total_training_flops = int(flops_per_iteration * num_samples * effective_iterations)
    
    # Total FLOPs for testing (inference on test set)
    #total_testing_flops = calculate_mlp_inference_flops(M, h, N, num_samples)
    
    # Grand total FLOPs
    grand_total_flops = total_training_flops #+ total_testing_flops
    
    return grand_total_flops

def calculate_mlp_inference_flops(M, h, N, num_samples):
    """
    Calculate the number of FLOPs for inference in an MLP.
    
    Args:
    M (int): Number of input features
    h (int): Number of neurons in the hidden layer
    N (int): Number of output features
    num_samples (int): Number of samples to perform inference on
    
    Returns:
    int: Total number of FLOPs for inference
    """
    # First Linear layer FLOPs
    linear1_flops = 2 * M * h
    
    # First ReLU activation FLOPs
    relu1_flops = h
    
    # Second Linear layer FLOPs
    linear2_flops = 2 * h * N
    
    # Second ReLU activation FLOPs
    relu2_flops = N
    
    # Decoder (output) layer FLOPs
    decoder_flops = 2 * N * M
    
    # Total FLOPs for one sample
    flops_per_sample = linear1_flops + relu1_flops + linear2_flops + relu2_flops + decoder_flops
    
    # Total FLOPs for all samples
    total_flops = flops_per_sample * num_samples
    
    return total_flops

def calculate_sae_inference_flops(M, N, num_samples):
    """
    Calculate the number of FLOPs for inference in an SAE.
    
    Args:
    M (int): Number of input features
    N (int): Number of output features (neurons in the hidden layer)
    num_samples (int): Number of samples to perform inference on
    
    Returns:
    int: Total number of FLOPs for inference
    """
    # Encoder Linear layer FLOPs
    encoder_flops = 2 * M * N  # includes matrix multiplication and bias addition
    
    # ReLU activation FLOPs
    relu_flops = N  # one comparison per output neuron
    
    # Decoder Linear layer FLOPs
    decoder_flops = 2 * N * M  # matrix multiplication for reconstruction
    
    # Total FLOPs for one sample
    flops_per_sample = encoder_flops + relu_flops + decoder_flops
    
    # Total FLOPs for all samples
    total_flops = flops_per_sample * num_samples
    
    return total_flops


def calculate_optimize_codes_flops(M, N, num_data, num_iterations):
    # Initialization FLOPs
    init_flops = M * N + N

    # FLOPs per iteration
    flops_per_iteration = 4 * M * N + 2 * M + 11 * N

    # Total FLOPs
    total_flops = init_flops + (flops_per_iteration * num_iterations)

    # Multiply by num_data as this is done for each data point
    return total_flops * num_data