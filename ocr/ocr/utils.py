
def plot_evaluation_heatmap_1(evaluation_summary_path, model_ids):

    import json
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    # Define metric direction: 1 means higher is better, -1 means lower is better
    metric_directions = {
        'avg_acc': 1,  # Higher accuracy is better
        'avg_cer': -1,  # Lower Character Error Rate (CER) is better
        'avg_wer': -1,  # Lower Word Error Rate (WER) is better
        'avg_order_agnostic_acc': 1,  # Higher order-agnostic accuracy is better
        'avg_processing_time': -1  # Lower processing time is better
    }

    # Read all JSON files into a list of DataFrames
    dfs = {}
    for model_id in model_ids:
        with open(os.path.join(evaluation_summary_path, f"summary_data_{model_id}_vid.json"), 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df = df[['model', 'vid', 'avg_acc', 'avg_cer', 'avg_wer', 'avg_order_agnostic_acc', 'avg_processing_time']]
            dfs[model_id] = df

    # Merge DataFrames on 'vid'
    merged_df = pd.merge(dfs[model_ids[0]], dfs[model_ids[1]], on='vid', suffixes=('_gpt4o', '_mistral'))

    # Calculate transformed differences
    metrics = list(metric_directions.keys())
    for metric in metrics:
        direction = metric_directions[metric]

        # Transform difference to always have positive values when GPT-4o is better
        if direction == 1:  # Higher is better
            merged_df[f'{metric}_diff'] = merged_df[f'{metric}_gpt4o'] - merged_df[f'{metric}_mistral']
        else:  # Lower is better
            merged_df[f'{metric}_diff'] = merged_df[f'{metric}_mistral'] - merged_df[f'{metric}_gpt4o']

    # Normalize each metric and store formatted values
    normalized_df = merged_df.copy()
    metrics_diff = [f"{metric}_diff" for metric in metrics]

    for metric in metrics_diff:
        min_val = merged_df[metric].min()
        max_val = merged_df[metric].max()

        if max_val != min_val:  # Avoid division by zero
            normalized_values = (merged_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_values = merged_df[metric]  # Keep original values if there's no range

        # Store formatted values with original numbers
        normalized_df[metric] = [
            f"{norm_val:.2f} ({orig_val:.4f})" for norm_val, orig_val in zip(normalized_values, merged_df[metric])
        ]

    # Custom colormap (red = GPT-4o better, blue = Mistral better)
    cmap_red_blue = mcolors.LinearSegmentedColormap.from_list("red_blue", ["blue", "white", "red"])

    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_diff), figsize=(20, 8), sharey=True)

    # Convert original values to numeric for heatmap (since we formatted normalized_df with strings)
    numeric_df = merged_df[metrics_diff]

    # Plot each metric
    for ax, metric in zip(axes, metrics_diff):
        sns.heatmap(
            numeric_df[[metric]],
            ax=ax,
            cmap=cmap_red_blue,
            center=0,  # Ensures white at zero
            cbar=True,
            annot=normalized_df[[metric]],
            fmt='',  # Use empty format since values are already formatted
            linewidths=0.5,
            linecolor='black'
        )
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Set common labels
    fig.suptitle('Performance Differences Between Models (Red = GPT-4o, Blue = Mistral)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show();

def plot_evaluation_heatmap_2(evaluation_summary_path, model_ids):

    import json
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    # Define metric direction: 1 means higher is better, -1 means lower is better
    metric_directions = {
        'avg_acc': 1,  # Higher accuracy is better
        'avg_cer': -1,  # Lower Character Error Rate (CER) is better
        'avg_wer': -1,  # Lower Word Error Rate (WER) is better
        'avg_order_agnostic_acc': 1,  # Higher order-agnostic accuracy is better
        'avg_processing_time': -1  # Lower processing time is better
    }

    # Read all JSON files into a dictionary of DataFrames
    dfs = {}
    for model_id in model_ids:
        with open(os.path.join(evaluation_summary_path, f"summary_data_{model_id}_vid.json"), 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df = df[['model', 'vid'] + list(metric_directions.keys())]
            dfs[model_id] = df

    # Merge DataFrames on 'vid'
    merged_df = pd.merge(dfs[model_ids[0]], dfs[model_ids[1]], on='vid', suffixes=(f"_{model_ids[0]}", f"_{model_ids[1]}"))

    # Calculate differences based on metric direction
    for metric, direction in metric_directions.items():
        if direction == 1:  # Higher is better
            merged_df[f'{metric}_diff'] = merged_df[f'{metric}_{model_ids[0]}'] - merged_df[f'{metric}_{model_ids[1]}']
        else:  # Lower is better
            merged_df[f'{metric}_diff'] = merged_df[f'{metric}_{model_ids[1]}'] - merged_df[f'{metric}_{model_ids[0]}']

    # Apply Max Absolute Scaling
    scaled_df = merged_df[[f'{metric}_diff' for metric in metric_directions]].apply(lambda x: x / np.max(np.abs(x)))

    # Create a DataFrame for annotations with original values
    annotations_df = merged_df[[f'{metric}_diff' for metric in metric_directions]].map(lambda x: f'{x:.4f}')

    # Custom colormap (red = GPT-4o better, blue = Mistral better)
    cmap_red_blue = mcolors.LinearSegmentedColormap.from_list("red_blue", ["blue", "white", "red"])

    # Set up the matplotlib figure with increased size
    plt.figure(figsize=(14, len(merged_df) * 1))

    # Plot the heatmap
    sns.heatmap(
        scaled_df,
        cmap=cmap_red_blue,
        center=0,
        cbar=True,
        annot=annotations_df,
        fmt='',
        linewidths=0.5,
        linecolor='black',
        xticklabels=[metric for metric in metric_directions],
        yticklabels=merged_df['vid']
    )

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=90, ha='right')

    # Set labels
    plt.xlabel('Metrics')
    plt.ylabel('Video ID')
    plt.title(f"Performance Differences Between Models (Red = {model_ids[0]}, Blue = {model_ids[1]})")

    # Adjust layout with specific padding
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.show();

import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plot_evaluation_boxplot(evaluation_summary_path, model_ids):
    # Define the metrics and their preferences
    metrics = {
        'avg_acc': 'higher',  # Higher accuracy is better
        'avg_cer': 'lower',   # Lower Character Error Rate (CER) is better
        'avg_wer': 'lower',   # Lower Word Error Rate (WER) is better
        'avg_order_agnostic_acc': 'higher',  # Higher order-agnostic accuracy is better
        'avg_processing_time': 'lower'  # Lower processing time is better
    }
    
    # Read all JSON files into a list of DataFrames
    dfs = []
    for model_id in model_ids:
        file_path = os.path.join(evaluation_summary_path, f"summary_data_{model_id}_vid.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['model'] = model_id  # Track model in DataFrame
            df = df[['model', 'vid'] + list(metrics.keys())]
            dfs.append(df)
    
    # Concatenate DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Reshape data for boxplot format
    melted_df = combined_df.melt(id_vars=['model', 'vid'], value_vars=metrics.keys(), 
                                 var_name='metric', value_name='value')
    
    # Define custom colors for the models
    if len(model_ids) >= 2:
        model_palette = {model_ids[0]: 'red', model_ids[1]: 'blue'}
    else:
        raise ValueError("Please provide at least two model IDs.")
    
    # Create boxplots for each metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 2 * num_metrics), sharex=True)
    
    # Ensure axes is iterable
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    for ax, (metric, preference) in zip(axes, metrics.items()):
        sns.boxplot(x='model', y='value', hue='model', data=melted_df[melted_df['metric'] == metric], 
                    ax=ax, palette=model_palette, dodge=False)
        
        # Determine the appropriate emoji
        if preference == 'higher':
            emoji = "\u2191"
        else:
            emoji = "\u2193"
        
        # Set the title with the emoji
        ax.set_title(f'Comparison of {metric} {emoji}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
       
        # Increase the frequency of y-axis ticks
        y_min, y_max = ax.get_ylim()
        y_ticks = np.linspace(y_min, y_max, num=11)  # Adjust 'num' for desired tick frequency
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
        
        # Set y-axis tick labels to one decimal place
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    plt.tight_layout()
    plt.show()
