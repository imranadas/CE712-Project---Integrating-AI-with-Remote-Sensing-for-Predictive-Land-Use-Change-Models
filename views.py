import io
import os
import base64
import geemap
import numpy as np
import pandas as pd
from PIL import Image
from ipywidgets import widgets
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from IPython.display import display
from ipyleaflet import ImageOverlay
from ipyleaflet import WidgetControl
from utils import process_landsat_data
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

def create_map_with_slider(processed_data_list, bounds, roi_gdf):
    map_center = [26.907436, 75.794157]
    zoom_level = 10
    m = geemap.Map(center=map_center, zoom=zoom_level)
    
    # Add ROI
    m.add_gdf(roi_gdf, "Region of Interest")
    
    # Create custom colormap for classification
    classification_colors = {
        0: '#FFD700',  # Barren - Yellow
        1: '#006400',  # Dense Vegetation - Dark Green
        2: '#90EE90',  # Moderate Vegetation - Light Green
        3: '#FF0000',  # Urban - Red
        4: '#00008B'   # Water - Blue
    }
    
    # Create time slider
    slider = widgets.SelectionSlider(
        options=[(data['date_str'], i) for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    # Keep track of current layers
    current_layers = []
    
    # Custom colormap function
    def custom_colormap(classification):
        colored = np.zeros((*classification.shape, 3), dtype=np.uint8)
        for class_val, color in classification_colors.items():
            mask = (classification == class_val)
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            for i, c in enumerate(rgb):
                colored[mask, i] = c
        return colored
    
    # Function to update visible layers
    def update_layers(change):
        nonlocal current_layers
        
        # Get the index of the selected date
        new_value = change['new']
        if isinstance(new_value, (list, tuple)) and len(new_value) > 1:
            idx = new_value[1]
        else:
            idx = new_value
            
        # Remove current layers
        for layer in current_layers:
            if layer in m.layers:
                m.remove_layer(layer)
        current_layers.clear()
        
        # Create and add new layers for the selected date
        data = processed_data_list[idx]
        
        # Convert classification to RGB image
        rgb_image = custom_colormap(data['classification'])
        img = Image.fromarray(rgb_image)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        layer = ImageOverlay(
            url=f"data:image/png;base64,{img_str}",
            bounds=bounds,
            name=f'Land Use Classification - {data["date_str"]}'
        )
        m.add_layer(layer)
        current_layers.append(layer)
    
    # Register the update function
    slider.observe(update_layers, names='value')
    
    # Add legend
    legend_dict = {
        'Barren Land': '#FFD700',
        'Dense Vegetation': '#006400',
        'Moderate Vegetation': '#90EE90',
        'Urban Areas': '#FF0000',
        'Water Bodies': '#00008B'
    }
    m.add_legend(title="Land Use Classification", legend_dict=legend_dict)
    
    # Add slider to map
    slider_control = WidgetControl(widget=slider, position='topright')
    m.add_control(slider_control)
    
    # Initialize the first view
    update_layers({'new': (processed_data_list[0]['date_str'], 0)})
    
    return m

def create_prediction_map_with_slider(processed_data_list, bounds, roi_gdf):
    map_center = [26.907436, 75.794157]
    zoom_level = 10
    m = geemap.Map(center=map_center, zoom=zoom_level)
    
    # Add ROI
    m.add_gdf(roi_gdf, "Region of Interest")
    
    # Create custom colormap for classification
    classification_colors = {
        0: '#FFD700',  # Barren - Yellow
        1: '#006400',  # Dense Vegetation - Dark Green
        2: '#90EE90',  # Moderate Vegetation - Light Green
        3: '#FF0000',  # Urban - Red
        4: '#00008B'   # Water - Blue
    }
    
    # Create time slider with explicit prediction labels
    slider_options = []
    for i, data in enumerate(processed_data_list):
        date_str = data['date_str']
        # Check if this is a predicted date
        if i >= len([d for d in processed_data_list if 'classification' in d]):
            date_str = f"{date_str} (Predicted)"
        slider_options.append((date_str, i))
    
    slider = widgets.SelectionSlider(
        options=slider_options,
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    # Keep track of current layers
    current_layers = []
    
    # Custom colormap function with alpha for predictions
    def custom_colormap(classification, is_prediction=False):
        colored = np.zeros((*classification.shape, 4), dtype=np.uint8)  # Added alpha channel
        for class_val, color in classification_colors.items():
            mask = (classification == class_val)
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            for i, c in enumerate(rgb):
                colored[mask, i] = c
            # Set alpha channel (more transparent for predictions)
            colored[mask, 3] = 180 if is_prediction else 255
        return colored
    
    # Function to update visible layers
    def update_layers(change):
        nonlocal current_layers
        
        # Get the index of the selected date
        new_value = change['new']
        if isinstance(new_value, (list, tuple)) and len(new_value) > 1:
            idx = new_value[1]
        else:
            idx = new_value
            
        # Remove current layers
        for layer in current_layers:
            if layer in m.layers:
                m.remove_layer(layer)
        current_layers.clear()
        
        # Create and add new layers for the selected date
        data = processed_data_list[idx]
        
        # Check if this is a prediction
        is_prediction = 'Predicted' in slider_options[idx][0]
        
        # Convert classification to RGBA image
        rgb_image = custom_colormap(data['classification'], is_prediction)
        img = Image.fromarray(rgb_image)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        layer = ImageOverlay(
            url=f"data:image/png;base64,{img_str}",
            bounds=bounds,
            name=f'Land Use Classification - {data["date_str"]}'
        )
        m.add_layer(layer)
        current_layers.append(layer)
        
        # Add prediction indicator if needed
        if is_prediction:
            prediction_text = widgets.HTML(
                value='<div style="background-color: rgba(255,255,0,0.2); padding: 8px; border-radius: 4px;">'
                      '<b>⚠️ Predicted Data</b></div>'
            )
            if not hasattr(m, 'prediction_control'):
                m.prediction_control = WidgetControl(widget=prediction_text, position='topright')
                m.add_control(m.prediction_control)
        else:
            if hasattr(m, 'prediction_control'):
                m.remove_control(m.prediction_control)
                delattr(m, 'prediction_control')
    
    # Register the update function
    slider.observe(update_layers, names='value')
    
    # Add legend with prediction note
    legend_dict = {
        'Barren Land': '#FFD700',
        'Dense Vegetation': '#006400',
        'Moderate Vegetation': '#90EE90',
        'Urban Areas': '#FF0000',
        'Water Bodies': '#00008B',
    }
    m.add_legend(title="Land Use Classification", legend_dict=legend_dict)
    
    # Add slider to map
    slider_control = WidgetControl(widget=slider, position='topright')
    m.add_control(slider_control)
    
    # Initialize the first view
    update_layers({'new': (processed_data_list[0]['date_str'], 0)})
    
    return m

def create_band_viewer(processed_data_dir):
    plt.ioff()
    
    # Create widgets
    date_slider = widgets.SelectionSlider(
        options=[(f.split('.')[0], f) for f in sorted(os.listdir(processed_data_dir)) if f.endswith('.npy')],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='150px')
    )
    
    # Create output widget for the plot
    plot_output = widgets.Output()
    
    # Band information
    band_info = {
        'Band 1': 'Ultra Blue',
        'Band 2': 'Blue',
        'Band 3': 'Green',
        'Band 4': 'Red',
        'Band 5': 'NIR',
        'Band 6': 'SWIR 1',
        'Band 7': 'SWIR 2',
        'DEM': 'Elevation',
        'Temperature': 'Air Temperature',
        'Precipitation': 'Precipitation'
    }
    
    def create_plot(filename):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            # Load the data
            filepath = os.path.join(processed_data_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            
            # Create figure
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
            
            # Plot each band
            for i, (band_name, description) in enumerate(band_info.items()):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                
                if 'Temperature' in band_name:
                    cmap = 'RdYlBu_r'
                    vmin = np.nanpercentile(data[i], 2)
                    vmax = np.nanpercentile(data[i], 98)
                elif 'Precipitation' in band_name:
                    cmap = 'Blues'
                    vmin = 0
                    vmax = np.nanpercentile(data[i], 98)
                elif 'DEM' in band_name:
                    cmap = 'terrain'
                    vmin = np.nanpercentile(data[i], 2)
                    vmax = np.nanpercentile(data[i], 98)
                else:
                    cmap = 'gray'
                    vmin = np.nanpercentile(data[i], 2)
                    vmax = np.nanpercentile(data[i], 98)
                
                # Create image
                im = ax.imshow(data[i], cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Set title with band name and description
                ax.set_title(f"{band_name}\n({description})")
                ax.axis('off')
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Calculate and display statistics
            stats_html = f"""
            <h4>Band Statistics for {filename.split('.')[0]}</h4>
            <table style="width:100%">
                <tr><th>Band</th><th>Mean</th><th>Min</th><th>Max</th><th>Std Dev</th></tr>
            """
            
            for i, (band_name, _) in enumerate(band_info.items()):
                band_data = data[i]
                stats_html += f"""
                <tr>
                    <td><b>{band_name}</b></td>
                    <td>{np.nanmean(band_data):.2f}</td>
                    <td>{np.nanmin(band_data):.2f}</td>
                    <td>{np.nanmax(band_data):.2f}</td>
                    <td>{np.nanstd(band_data):.2f}</td>
                </tr>
                """
            
            stats_html += "</table>"
            stats_text.value = stats_html
    
    def update_display(change):
        filename = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(filename)
    
    # Register the update function
    date_slider.observe(update_display, names='value')
    
    # Initial plot
    create_plot(date_slider.options[0][1])
    
    return widgets.VBox([
        widgets.HTML("<h2>Landsat Band Visualization</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])
    
def create_spectral_indices_viewer(processed_data_list):
    plt.ioff()
    
    # Create widgets
    date_slider = widgets.SelectionSlider(
        options=[(data['date_str'], i) for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='150px')
    )
    
    # Create output widget for the plot
    plot_output = widgets.Output()
    
    def create_plot(idx):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            data = processed_data_list[idx]
            
            # Create the figure and axes
            fig = plt.figure(figsize=(20, 20))
            gs = fig.add_gridspec(2, 2)
            
            # Configure plots for each index
            plots_config = {
                'NDVI': {'pos': (0, 0), 'cmap': 'RdYlGn'},
                'NDBI': {'pos': (0, 1), 'cmap': 'RdYlBu_r'},
                'NDWI': {'pos': (1, 0), 'cmap': 'RdYlBu'},
                'EBBI': {'pos': (1, 1), 'cmap': 'YlOrRd'}
            }
            
            for name, config in plots_config.items():
                ax = fig.add_subplot(gs[config['pos']])
                im = ax.imshow(data[name], cmap=config['cmap'], vmin=-1, vmax=1)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(name)
                ax.axis('off')
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Update statistics
            stats_html = f"""
            <h4>Statistics for {data['date_str']}</h4>
            <table>
                <tr><td><b>NDVI Mean:</b></td><td>{np.nanmean(data['NDVI']):.3f}</td></tr>
                <tr><td><b>NDBI Mean:</b></td><td>{np.nanmean(data['NDBI']):.3f}</td></tr>
                <tr><td><b>NDWI Mean:</b></td><td>{np.nanmean(data['NDWI']):.3f}</td></tr>
                <tr><td><b>EBBI Mean:</b></td><td>{np.nanmean(data['EBBI']):.3f}</td></tr>
            </table>
            """
            stats_text.value = stats_html
    
    def update_display(change):
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(idx)
    
    # Register the update function
    date_slider.observe(update_display, names='value')
    
    # Initial plot
    create_plot(0)
    
    return widgets.VBox([
        widgets.HTML("<h2>Spectral Indices Visualization</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])
    
def create_classification_viewer(processed_data_list):
    plt.ioff()
    
    # Create widgets
    date_slider = widgets.SelectionSlider(
        options=[(data['date_str'], i) for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='150px')
    )
    
    # Create output widget for the plot
    plot_output = widgets.Output()
    
    # Define class information
    class_info = {
        'Barren Land': {'value': 0, 'color': '#FFD700'},
        'Dense Vegetation': {'value': 1, 'color': '#006400'},
        'Moderate Vegetation': {'value': 2, 'color': '#90EE90'},
        'Urban': {'value': 3, 'color': '#FF0000'},
        'Water': {'value': 4, 'color': '#00008B'}
    }
    
    # Custom colormap
    from matplotlib.colors import ListedColormap
    colors = [info['color'] for info in class_info.values()]
    custom_cmap = ListedColormap(colors)
    
    def create_plot(idx):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            data = processed_data_list[idx]
            
            # Create figure
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 3)
            
            # Create subplots for each class
            for i, (label, info) in enumerate(class_info.items()):
                if i < 5:  # Only create 5 subplots
                    ax = fig.add_subplot(2, 3, i+1)
                    mask = data['classification'] == info['value']
                    ax.imshow(mask, cmap='binary')
                    ax.set_title(label)
                    ax.axis('off')
            
            # Combined map
            ax_combined = fig.add_subplot(2, 3, 6)
            combined_map = data['classification']
            im_combined = ax_combined.imshow(combined_map, cmap=custom_cmap, vmin=0, vmax=4)
            ax_combined.set_title('Combined Map')
            ax_combined.axis('off')
            
            # Add colorbar
            cbar = fig.colorbar(im_combined, ax=ax_combined, fraction=0.046, pad=0.04)
            cbar.set_ticks(np.arange(5))
            cbar.set_ticklabels(class_info.keys())
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Calculate statistics
            total_pixels = np.size(data['classification'])
            stats_html = f"""
            <h4>Land Cover Statistics for {data['date_str']}</h4>
            <table>
                <tr><td><b>Barren Land:</b></td><td>{np.sum(data['classification'] == 0) / total_pixels * 100:.1f}%</td></tr>
                <tr><td><b>Dense Vegetation:</b></td><td>{np.sum(data['classification'] == 1) / total_pixels * 100:.1f}%</td></tr>
                <tr><td><b>Moderate Vegetation:</b></td><td>{np.sum(data['classification'] == 2) / total_pixels * 100:.1f}%</td></tr>
                <tr><td><b>Urban:</b></td><td>{np.sum(data['classification'] == 3) / total_pixels * 100:.1f}%</td></tr>
                <tr><td><b>Water:</b></td><td>{np.sum(data['classification'] == 4) / total_pixels * 100:.1f}%</td></tr>
            </table>
            """
            stats_text.value = stats_html
    
    def update_display(change):
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(idx)
    
    # Register the update function
    date_slider.observe(update_display, names='value')
    
    # Initial plot
    create_plot(0)
    
    return widgets.VBox([
        widgets.HTML("<h2>Land Cover Classification Visualization</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])
    
def create_prediction_viewer(processed_data_list):
    plt.ioff()
    
    # Create widgets
    date_slider = widgets.SelectionSlider(
        options=[(f"{data['date_str']} {'(Predicted)' if 'Predicted' in data.get('type', '') else ''}", i) 
                for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='150px')
    )
    
    # Create output widget for the plot
    plot_output = widgets.Output()
    
    # Define class information with colors and labels
    class_info = {
        'Barren Land': {'value': 0, 'color': '#FFD700', 'cmap': 'YlOrBr'},
        'Dense Vegetation': {'value': 1, 'color': '#006400', 'cmap': 'Greens'},
        'Moderate Vegetation': {'value': 2, 'color': '#90EE90', 'cmap': 'YlGn'},
        'Urban': {'value': 3, 'color': '#FF0000', 'cmap': 'Reds'},
        'Water': {'value': 4, 'color': '#00008B', 'cmap': 'Blues'}
    }
    
    # Create custom colormap for combined view
    combined_colors = [info['color'] for info in class_info.values()]
    custom_cmap = ListedColormap(combined_colors)
    
    def create_plot(idx):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            data = processed_data_list[idx]
            is_prediction = 'Predicted' in data.get('type', '')
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Plot individual class maps
            for i, (label, info) in enumerate(class_info.items()):
                if i < 5:  # Only create 5 subplots (one for each class)
                    ax = fig.add_subplot(gs[i // 3, i % 3])
                    mask = data['classification'] == info['value']
                    
                    # Use specific colormap for each class
                    im = ax.imshow(mask, cmap=info['cmap'])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    # Add prediction indicator if needed
                    title = f"{label}\n"
                    if is_prediction:
                        title += "(Predicted)"
                    ax.set_title(title)
                    ax.axis('off')
            
            # Create combined view
            ax_combined = fig.add_subplot(gs[1, 2])
            im_combined = ax_combined.imshow(data['classification'], 
                                           cmap=custom_cmap, 
                                           vmin=0, 
                                           vmax=4)
            
            # Add colorbar for combined view
            cbar = fig.colorbar(im_combined, ax=ax_combined, 
                              fraction=0.046, 
                              pad=0.04)
            cbar.set_ticks(np.arange(5))
            cbar.set_ticklabels(class_info.keys())
            
            # Add title for combined view
            title = "Combined Classification\n"
            if is_prediction:
                title += "(Predicted)"
            ax_combined.set_title(title)
            ax_combined.axis('off')
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Update statistics
            total_pixels = np.size(data['classification'])
            stats_html = f"""
            <h4>Land Cover Statistics for {data['date_str']}</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f2f2f2;'>
                    <th style='padding: 8px; text-align: left;'>Class</th>
                    <th style='padding: 8px; text-align: right;'>Coverage</th>
                    <th style='padding: 8px; text-align: right;'>Pixels</th>
                </tr>
            """
            
            for label, info in class_info.items():
                class_pixels = np.sum(data['classification'] == info['value'])
                percentage = (class_pixels / total_pixels) * 100
                stats_html += f"""
                <tr>
                    <td style='padding: 8px;'>{label}</td>
                    <td style='padding: 8px; text-align: right;'>{percentage:.1f}%</td>
                    <td style='padding: 8px; text-align: right;'>{class_pixels:,}</td>
                </tr>
                """
            
            stats_html += "</table>"
            
            if is_prediction:
                stats_html += """
                <div style='margin-top: 10px; padding: 8px; background-color: #fff3cd; 
                           border: 1px solid #ffeeba; border-radius: 4px;'>
                    <strong>⚠️ Note:</strong> This is a predicted classification
                </div>
                """
            
            stats_text.value = stats_html
    
    def update_display(change):
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(idx)
    
    # Register the update function
    date_slider.observe(update_display, names='value')
    
    # Initial plot
    create_plot(0)
    
    return widgets.VBox([
        widgets.HTML("<h2>Land Cover Classification Viewer</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])
    
def analyze_time_series(data_dir):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    def process_file(file_path, data_dir):
        try:
            data = process_landsat_data(os.path.join(data_dir, file_path))
            
            # Calculate percentages efficiently using np.sum
            classification = data['classification']
            total_pixels = classification.size
            class_counts = np.bincount(classification.ravel(), minlength=5)
            
            return {
                'date_str': data['date_str'],
                'date_obj': data['date'],
                'urban_percent': class_counts[3] / total_pixels * 100,
                'vegetation_percent': (class_counts[1] + class_counts[2]) / total_pixels * 100,
                'water_percent': class_counts[4] / total_pixels * 100,
                'mean_ndvi': np.nanmean(data['NDVI']),
                'mean_ndbi': np.nanmean(data['NDBI']),
                'mean_ndwi': np.nanmean(data['NDWI']),
                'mean_temp': np.nanmean(data['air_temp']),
                'total_precipitation': np.nansum(data['precipitation'])
            }
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(partial(process_file, data_dir=data_dir), file_list))
    
    # Filter out None results and sort by date
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x['date_obj'])
    
    return results

def plot_time_series(results):
    plt.close('all')
    
    df = pd.DataFrame(results)
    df.set_index('date_str', inplace=True)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Spectral indices
    ax1 = fig.add_subplot(gs[0, 0])
    df[['mean_ndvi', 'mean_ndbi', 'mean_ndwi']].plot(ax=ax1)
    ax1.set_title('Spectral Indices Over Time')
    ax1.set_ylabel('Index Value')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # Land cover percentages
    ax2 = fig.add_subplot(gs[0, 1])
    df[['urban_percent', 'vegetation_percent', 'water_percent']].plot(ax=ax2)
    ax2.set_title('Land Cover Changes')
    ax2.set_ylabel('Percentage')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Temperature
    ax3 = fig.add_subplot(gs[1, 0])
    df['mean_temp'].plot(ax=ax3, color='red')
    ax3.set_title('Temperature Variation')
    ax3.set_ylabel('Temperature (K)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Precipitation
    ax4 = fig.add_subplot(gs[1, 1])
    df['total_precipitation'].plot(ax=ax4, color='blue', kind='bar')
    ax4.set_title('Precipitation')
    ax4.set_ylabel('Precipitation (mm)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    
def visualize_predictions(predictions: List[Dict[str, Any]]) -> None:
    """
    Visualize prediction results.
    
    Args:
        predictions: List of prediction dictionaries
    """
    if not predictions:
        print("No predictions to visualize")
        return
    
    # Set up the visualization
    n_predictions = len(predictions)
    fig = plt.figure(figsize=(15, 4 * n_predictions))
    gs = gridspec.GridSpec(n_predictions, 2, figure=fig)
    
    # Create custom colormap for classifications
    class_colors = ['yellow', 'darkgreen', 'red', 'darkblue']
    class_cmap = ListedColormap(class_colors)
    
    # Plot each prediction
    for idx, pred in enumerate(predictions):
        # Plot classification
        ax1 = fig.add_subplot(gs[idx, 0])
        im1 = ax1.imshow(pred['classification'], cmap=class_cmap)
        ax1.set_title(f"Predicted Land Use\n{pred['date_str']}")
        plt.colorbar(im1, ax=ax1, ticks=range(4),
                    label='Class',
                    boundaries=np.arange(-0.5, 5.5))
        
        # Plot confidence
        ax2 = fig.add_subplot(gs[idx, 1])
        im2 = ax2.imshow(pred['confidence'], cmap='RdYlGn')
        ax2.set_title(f"Prediction Confidence\n{pred['date_str']}")
        plt.colorbar(im2, ax=ax2, label='Confidence Score')
    
    plt.tight_layout()
    plt.show()

class ModelAnalysis:
    """Class for analyzing model performance and predictions."""
    
    @staticmethod
    def analyze_class_distribution(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze class distribution over time.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            DataFrame with class distribution analysis
        """
        results = []
        for pred in predictions:
            total_pixels = pred['classification'].size
            distribution = {
                'date': pred['date_str'],
                'total_pixels': total_pixels
            }
            
            # Calculate class percentages
            unique, counts = np.unique(pred['classification'], return_counts=True)
            for cls, count in zip(unique, counts):
                cls_name = f"class_{cls}_percent"
                distribution[cls_name] = (count / total_pixels) * 100
            
            results.append(distribution)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def analyze_confidence(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze prediction confidence.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            DataFrame with confidence analysis
        """
        results = []
        for pred in predictions:
            confidence = pred['confidence']
            results.append({
                'date': pred['date_str'],
                'mean_confidence': np.mean(confidence),
                'min_confidence': np.min(confidence),
                'max_confidence': np.max(confidence),
                'std_confidence': np.std(confidence)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def analyze_spatial_changes(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze spatial changes between predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            DataFrame with spatial change analysis
        """
        results = []
        for i in range(len(predictions) - 1):
            current = predictions[i]['classification']
            next_pred = predictions[i + 1]['classification']
            
            changes = current != next_pred
            change_percentage = np.mean(changes) * 100
            
            results.append({
                'from_date': predictions[i]['date_str'],
                'to_date': predictions[i + 1]['date_str'],
                'change_percentage': change_percentage,
                'pixels_changed': np.sum(changes),
                'total_pixels': changes.size
            })
        
        return pd.DataFrame(results)
    