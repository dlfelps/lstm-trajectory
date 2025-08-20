import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context
import numpy as np
import pickle
import pandas as pd

def load_saved_data():
    """Load the saved data from results folder"""
    print("Loading saved data from results folder...")
    
    # Load trajectories
    with open('results/traj.pkl', 'rb') as f:
        traj = pickle.load(f)
    
    # Load labels
    with open('results/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    
    # Load PCA 2D projections
    pca_2d = np.load('results/pca_2d.npy')
    
    print(f"Loaded {len(traj)} trajectories, {len(labels)} labels, PCA shape: {pca_2d.shape}")
    return traj, labels, pca_2d

def create_interactive_app():
    """Create the Dash interactive app"""
    
    # Load data
    traj, labels, pca_2d = load_saved_data()
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'x': pca_2d[:, 0],
        'y': pca_2d[:, 1],
        'user_id': labels,
        'point_id': range(len(labels))
    })
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Define the layout
    app.layout = html.Div([
        html.H1("Interactive Trajectory Embedding Visualization", 
                style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        html.Div([
            dcc.Graph(
                id='pca-scatter',
                config={'displayModeBar': True},
                style={'height': '600px'}
            )
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.H3("Selected Points Information"),
            html.Div(id='selection-info', 
                    style={'border': '1px solid #ddd', 
                           'padding': '20px', 
                           'borderRadius': '5px',
                           'backgroundColor': '#f9f9f9',
                           'minHeight': '400px',
                           'fontFamily': 'monospace'})
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ])
    
    # Callback for updating the plot
    @app.callback(
        Output('pca-scatter', 'figure'),
        Input('pca-scatter', 'id')  # Dummy input to trigger initial render
    )
    def update_scatter_plot(_):
        fig = px.scatter(
            df, 
            x='x', 
            y='y',
            color='user_id',
            hover_data=['point_id', 'user_id'],
            title='Trajectory Embeddings - First Two Principal Components (Interactive)',
            color_continuous_scale='viridis',
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            height=600,
            showlegend=True,
            hovermode='closest'
        )
        
        fig.update_traces(
            marker=dict(size=8, line=dict(width=0.5, color='white')),
            selector=dict(mode='markers')
        )
        
        return fig
    
    # Callback for handling point selection
    @app.callback(
        Output('selection-info', 'children'),
        Input('pca-scatter', 'selectedData')
    )
    def display_selected_points(selectedData):
        if selectedData is None or len(selectedData['points']) == 0:
            return html.Div([
                html.P("No points selected.", style={'color': '#666'}),
                html.P("Instructions:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                html.Ul([
                    html.Li("Use the box select or lasso select tools in the plot toolbar"),
                    html.Li("Click and drag to select multiple points"),
                    html.Li("Selected point information will appear here")
                ])
            ])
        
        # Extract information about selected points
        selected_points = selectedData['points']
        
        # Create summary - use pointIndex to get data from df
        point_indices = [point['pointIndex'] for point in selected_points]
        user_ids = [df.iloc[idx]['user_id'] for idx in point_indices]
        point_ids = [df.iloc[idx]['point_id'] for idx in point_indices]
        unique_users = list(set(user_ids))
        
        # Create detailed info
        info_elements = [
            html.H4(f"Selected {len(selected_points)} points"),
            html.P(f"From {len(unique_users)} unique users"),
            html.Hr(),
            html.H5("User ID Summary:"),
        ]
        
        # Add user ID counts
        user_counts = {}
        for uid in user_ids:
            user_counts[uid] = user_counts.get(uid, 0) + 1
        
        for uid, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
            info_elements.append(
                html.P(f"User {uid}: {count} points")
            )
        
        info_elements.extend([
            html.Hr(),
            html.H5("Point Details:"),
            html.Div([
                html.P(f"Point {pid}: User {uid} | Coords: ({selected_points[i]['x']:.3f}, {selected_points[i]['y']:.3f})")
                for i, (pid, uid) in enumerate(zip(point_ids[:20], user_ids[:20]))  # Limit to first 20 for display
            ]),
        ])
        
        if len(selected_points) > 20:
            info_elements.append(
                html.P(f"... and {len(selected_points) - 20} more points", style={'fontStyle': 'italic', 'color': '#666'})
            )
        
        return info_elements
    
    return app

if __name__ == '__main__':
    print("Creating interactive trajectory embedding visualization...")
    print("Make sure you have run inference.py first to generate the data files.")
    
    try:
        app = create_interactive_app()
        print("\nStarting Dash app...")
        print("Open your browser and go to: http://127.0.0.1:8050")
        app.run(debug=True, host='127.0.0.1', port=8050)
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files.")
        print(f"Please run inference.py first to generate the results folder.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"Error creating app: {e}")