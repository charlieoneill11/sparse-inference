import math

def calculate_generalsae_flops(projections_up, M):
    inference_flops = (2 * M * projections_up[0] + projections_up[0])
    for i in range(1, len(projections_up)):
        inference_flops += (2 * projections_up[i-1] * projections_up[i] + projections_up[i])
    inference_flops += (2 * projections_up[-1] * M)
    
    training_flops = 3 * inference_flops
    
    # Parameter updates
    param_count = sum(projections_up[i-1] * projections_up[i] + projections_up[i] for i in range(len(projections_up)))
    param_count += M * projections_up[0] + M  # first layer and bias
    
    training_flops += 4 * param_count
    
    return inference_flops, training_flops

def calculate_inference_flops(model_type, N, M, K, num_iterations=100, projections_up=None):
    if model_type == "SparseCoding":
        flops_per_iteration = 3*N*M + 6*M + 6*N - 3 + 4*N
        return num_iterations * flops_per_iteration * 4
    elif model_type == "SparseAutoEncoder":
        return 2*N*M + 2*N
    elif model_type == "GatedSAE":
        return 2*N*M + 10*N + 2*M
    elif model_type == "TopKSAE":
        return 2*N*M + 2*N + 2*M + N*K + K
    elif model_type == "GeneralSAE":
        if projections_up is None:
            raise ValueError("projections_up must be provided for GeneralSAE")
        return calculate_generalsae_flops(projections_up, M)[0]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def calculate_training_flops(model_type, N, M, K, batch_size, num_step, projections_up=None):
    if model_type == "SparseCoding":
        flops_per_step = batch_size * (3*N*M + 2*M + N) + 4 * (N*M + N)
    elif model_type == "SparseAutoEncoder":
        flops_per_step = batch_size * (4*N*M + 2*M + 3*N) + 4 * (N*M + N + M)
    elif model_type == "GatedSAE":
        flops_per_step = batch_size * (4*N*M + 8*M + 9*N) + 4 * (N*M + 6*N + 2*M)
    elif model_type == "TopKSAE":
        flops_per_step = batch_size * (4*N*M + 4*M + 2*N + N*K + K) + 4 * (N*M + N + M)
    elif model_type == "GeneralSAE":
        if projections_up is None:
            raise ValueError("projections_up must be provided for GeneralSAE")
        flops_per_step = batch_size * calculate_generalsae_flops(projections_up, M)[1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return flops_per_step * num_step

# Test the functions
if __name__ == "__main__":
    model_types = ["SparseCoding", "SparseAutoEncoder", "GatedSAE", "TopKSAE", "GeneralSAE"]
    N, M, hidden_size, K = 16, 8, 10, 3
    num_iterations = 10000
    batch_size = 32
    num_step = 20000
    
    for model_type in model_types:
        if model_type == "GeneralSAE":
            projections_up = [12, 16]  # Example projections
            inference_flops = calculate_inference_flops(model_type, N, M, K, num_iterations, projections_up)
            training_flops = calculate_training_flops(model_type, N, M, K, batch_size, num_step, projections_up)
        else:
            inference_flops = calculate_inference_flops(model_type, hidden_size, M, K, num_iterations)
            training_flops = calculate_training_flops(model_type, hidden_size, M, K, batch_size, num_step)
        
        print(f"{model_type}:")
        print(f"  Inference FLOPs: {inference_flops:,}")
        print(f"  Training FLOPs per epoch: {training_flops:,}")
        print(f"  Total FLOPs: {training_flops + inference_flops:,}")