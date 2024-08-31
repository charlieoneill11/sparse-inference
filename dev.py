# import plotly.graph_objects as go
# import json
# import numpy as np

# with open('results/latent_regression_results.json', 'r') as f:
#     data = json.load(f)

# fig = go.Figure()

# colors = {
#     "MLP": "blue",
#     "TopKSAE": "green",
#     "SparseAutoEncoder": "red"
# }

# for result in data['results']:
#     model_name = result['model']
#     eval_interval = data['parameters']['eval_interval']
#     steps = np.arange(eval_interval, data['parameters']['num_step'] + 1, eval_interval)
    
#     train_mccs_all_runs = []
#     test_mccs_all_runs = []
    
#     for run in result['runs']:
#         run_steps = [log['step'] for log in run['performance_log']]
#         run_train_mccs = [log['train_mcc'] for log in run['performance_log']]
#         run_test_mccs = [log['test_mcc'] for log in run['performance_log']]
        
#         train_mccs_all_runs.append(run_train_mccs)
#         test_mccs_all_runs.append(run_test_mccs)
    
#     # Convert lists to numpy arrays for easier manipulation
#     train_mccs_all_runs = np.array(train_mccs_all_runs)
#     test_mccs_all_runs = np.array(test_mccs_all_runs)
    
#     mean_train_mccs = np.mean(train_mccs_all_runs, axis=0)
#     mean_test_mccs = np.mean(test_mccs_all_runs, axis=0)
    
#     fig.add_trace(go.Scatter(
#         x=steps,
#         y=mean_train_mccs,
#         mode='lines+markers',
#         name=f'{model_name} Train MCC',
#         line=dict(color=colors[model_name], dash='solid'),
#         marker=dict(symbol='circle')
#     ))
    
#     fig.add_trace(go.Scatter(
#         x=steps,
#         y=mean_test_mccs,
#         mode='lines+markers',
#         name=f'{model_name} Test MCC',
#         line=dict(color=colors[model_name], dash='dash'),
#         marker=dict(symbol='cross')
#     ))

# fig.update_layout(
#     title='Average Training and Test MCCs for MLP, TopKSAE, and SparseAutoEncoder Models',
#     xaxis=dict(title='Training Steps'),
#     yaxis=dict(title='MCC'),
#     legend=dict(
#         title='Legend',
#         x=0.98,
#         y=0.02,
#         bordercolor='Black',
#         borderwidth=1
#     ),
#     font=dict(family="Arial", size=14),
#     plot_bgcolor='white',
#     margin=dict(l=0, r=0, t=0, b=0),
#     autosize=False,
#     width=1400,
#     height=600
# )

# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

# fig.show()
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the JSON data
with open('results/fixed_D_results.json', 'r') as f:
    data = json.load(f)

# Create subplots
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Colors for each model
colors = {'SparseAutoEncoder': 'rgb(31, 119, 180)', 'MLP': 'rgb(255, 127, 14)'}

for result in data['results']:
    print(result)
    model_name = result['model']
    total_flops = [log['total_flops'] for log in result['performance_log']]
    test_mcc = [log['test_mcc'] for log in result['performance_log']]
    final_test_mcc = result['final_test_mcc']
    
    # Add trace for each model
    fig.add_trace(
        go.Scatter(
            x=total_flops,
            y=test_mcc,
            mode='lines',
            name=f"{model_name} (Final MCC: {final_test_mcc:.4f})",
            line=dict(color=colors[model_name], width=3),
        )
    )

# Update layout
fig.update_layout(
    # title='Model Performance Comparison',
    xaxis_title='Total FLOPs',
    yaxis_title='Test MCC',
    font=dict(family="Palatino", size=18),
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='rgba(0, 0, 0, 0.1)',
        borderwidth=1,
        font=dict(size=16)
    ),
    plot_bgcolor='white',
    hovermode='x unified',
    width=800,  # Narrower plot
    height=500,
    margin=dict(l=80, r=50, t=100, b=80)
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(0, 0, 0, 0.1)',
    zeroline=False,
    showline=True,
    linewidth=2,
    linecolor='black',
    type='log',  # Use log scale for FLOPs
    title_standoff=20
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(0, 0, 0, 0.1)',
    zeroline=False,
    showline=True,
    linewidth=2,
    linecolor='black',
    range=[0.3, 1],  # Adjust this range as needed
    title_standoff=20
)

# Save the plot as a PDF
fig.write_image("results/figures/fixed_D.pdf")

# Optionally, display the plot
fig.show()