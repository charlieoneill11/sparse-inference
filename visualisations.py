# import json
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Load the JSON data
# with open('results/experiment_results.json', 'r') as f:
#     data = json.load(f)

# # Extract results
# results = data['results']

# # Create a dictionary to store data for each model
# model_data = {}

# # Process the results
# for result in results:
#     model = result['model']
#     if model not in model_data:
#         model_data[model] = {'inference_flops': [], 'training_flops': [], 'total_flops': [], 'mcc': [], 'hidden_dim': []}
    
#     model_data[model]['inference_flops'].append(result['inference_flops'])
#     model_data[model]['training_flops'].append(result['training_flops'])
#     model_data[model]['total_flops'].append(result['total_flops'])
#     model_data[model]['mcc'].append(result['mcc'])
#     model_data[model]['hidden_dim'].append(result['hidden_dim'])

# # Colors for each model
# colors = {'SparseCoding': '#1f77b4', 'SparseAutoEncoder': '#ff7f0e', 
#           'GatedSAE': '#2ca02c', 'TopKSAE': '#d62728'}

# def create_plot(flop_type):
#     fig = go.Figure()

#     for model, data in model_data.items():
#         fig.add_trace(
#             go.Scatter(
#                 x=data[f'{flop_type}_flops'],
#                 y=data['mcc'],
#                 mode='lines+markers',
#                 name=model,
#                 line=dict(color=colors[model]),
#                 marker=dict(size=10, color=colors[model], symbol='circle'),
#                 text=[f'Hidden Dim: {dim}' for dim in data['hidden_dim']],
#                 hoverinfo='text+x+y'
#             )
#         )

#     # Update layout
#     fig.update_layout(
#         title=f'Model Performance: MCC vs {flop_type.capitalize()} FLOPs',
#         xaxis_title=f'{flop_type.capitalize()} FLOPs',
#         yaxis_title='Mean Correlation Coefficient (MCC)',
#         font=dict(family="Arial", size=14),
#         legend=dict(
#             x=0.02,
#             y=0.98,
#             bgcolor='rgba(255, 255, 255, 0.8)',
#             bordercolor='rgba(0, 0, 0, 0.3)',
#             borderwidth=1
#         ),
#         plot_bgcolor='rgba(255, 255, 255, 1)',
#         xaxis=dict(
#             type='log',
#             showgrid=True,
#             gridcolor='rgba(0, 0, 0, 0.1)',
#             zeroline=False,
#             showline=True,
#             linewidth=1,
#             linecolor='black',
#             mirror=True
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(0, 0, 0, 0.1)',
#             zeroline=False,
#             showline=True,
#             linewidth=1,
#             linecolor='black',
#             mirror=True,
#             range=[0, 1]
#         ),
#         width=800,
#         height=600
#     )

#     fig.show()

#     # Save the plot
#     #fig.write_image(f"results/{flop_type}_flops_pareto_curves.pdf")
    
#     print(f"Plot saved as '{flop_type}_flops_pareto_curves.pdf'")

# # Create all three plots
# import time
# import plotly.express as px

# fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
# fig.write_image("results/random.pdf")
# time.sleep(5)

# create_plot('inference')
# create_plot('training')
# create_plot('total')

# # Delete random.pdf
# import os

# os.remove("results/random.pdf")

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Control variables
LINE_WIDTH = 4
BASE_FONT_SIZE = 32
MARKER_SIZE = 18

# Model name mapping
MODEL_NAME_MAP = {
    'SparseCoding': 'Sparse Coding',
    'SparseAutoEncoder': 'Sparse Autoencoder',
    'GatedSAE': 'Gated SAE',
    'TopKSAE': 'Top-k SAE'
}

# Load the JSON data
with open('results/experiment_results.json', 'r') as f:
    data = json.load(f)

# Extract results
results = data['results']

# Create a dictionary to store data for each model
model_data = {}

# Process the results
for result in results:
    model = result['model']
    if model not in model_data:
        model_data[model] = {'inference_flops': [], 'training_flops': [], 'total_flops': [], 'mcc': [], 'hidden_dim': []}
    
    model_data[model]['inference_flops'].append(result['inference_flops'])
    model_data[model]['training_flops'].append(result['training_flops'])
    model_data[model]['total_flops'].append(result['total_flops'])
    model_data[model]['mcc'].append(result['mcc'])
    model_data[model]['hidden_dim'].append(result['hidden_dim'])

# Colors for each model using plasma color scale
num_models = len(model_data)
colors = px.colors.sequential.Plasma_r[2:]  # Skip the first two colors
len_colors = len(colors)
# Uniformly go along colours and assign every nth colour to a model
colors = [colors[i * len_colors // num_models] for i in range(num_models)]
color_dict = dict(zip(model_data.keys(), colors))

def create_plot(flop_type):
    fig = go.Figure()
    
    for model, data in model_data.items():
        fig.add_trace(
            go.Scatter(
                x=data[f'{flop_type}_flops'],
                y=data['mcc'],
                mode='lines+markers',
                name=MODEL_NAME_MAP[model],  # Use mapped name for legend
                line=dict(color=color_dict[model], width=LINE_WIDTH),
                marker=dict(size=MARKER_SIZE, color=color_dict[model], symbol='circle'),
                text=[f'Hidden Dim: {dim}' for dim in data['hidden_dim']],
                hoverinfo='text+x+y'
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title=f'{flop_type.capitalize()} FLOPs',
        yaxis_title='Mean Correlation Coefficient (MCC)',
        font=dict(family="Palatino", size=BASE_FONT_SIZE),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0)',
            font=dict(size=BASE_FONT_SIZE-2)
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        xaxis=dict(
            type='log',
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.1)',
            zeroline=False,
            showline=False,
            linewidth=1,
            linecolor='black',
            mirror=False,
            tickfont=dict(size=BASE_FONT_SIZE-8)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.1)',
            zeroline=False,
            showline=False,
            linewidth=1,
            linecolor='black',
            mirror=False,
            range=[0, 1],
            tickfont=dict(size=BASE_FONT_SIZE-2)
        ),
        width=1200,
        height=800
    )
    
    fig.show()
    
    # Save the plot
    #fig.write_image(f"results/{flop_type}_flops_pareto_curves.pdf")
    
    print(f"Plot saved as '{flop_type}_flops_pareto_curves.pdf'")

# Create all three plots
create_plot('inference')
create_plot('training')
create_plot('total')