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

def create_band_viewer(processed_data_dir):
    """
    Create an enhanced interactive viewer for Landsat bands visualization.
    
    Args:
        processed_data_dir: Directory containing the processed Landsat data files
    
    Returns:
        widgets.VBox: Interactive band viewer with enhanced visualization and statistics
    """
    plt.ioff()
    
    # Create widgets
    date_slider = widgets.SelectionSlider(
        options=[(f.split('.')[0], f) for f in sorted(os.listdir(processed_data_dir)) if f.endswith('.npy')],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    # Enhanced statistics text with scrollable area
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='200px', overflow='auto')
    )
    
    # Create output widget for the plot
    plot_output = widgets.Output()
    
    # Enhanced band information with wavelengths and applications
    band_info = {
        'Band 1': {
            'name': 'Ultra Blue (Coastal/Aerosol)',
            'wavelength': '0.43-0.45 µm',
            'applications': 'Coastal water and aerosol studies',
            'cmap': 'gray'
        },
        'Band 2': {
            'name': 'Blue',
            'wavelength': '0.45-0.51 µm',
            'applications': 'Bathymetric mapping, soil/vegetation discrimination',
            'cmap': 'Blues'
        },
        'Band 3': {
            'name': 'Green',
            'wavelength': '0.53-0.59 µm',
            'applications': 'Peak vegetation reflection, vigour assessment',
            'cmap': 'Greens'
        },
        'Band 4': {
            'name': 'Red',
            'wavelength': '0.64-0.67 µm',
            'applications': 'Vegetation absorption, species discrimination',
            'cmap': 'Reds'
        },
        'Band 5': {
            'name': 'Near Infrared (NIR)',
            'wavelength': '0.85-0.88 µm',
            'applications': 'Biomass content and shorelines',
            'cmap': 'RdYlGn'
        },
        'Band 6': {
            'name': 'Short-wave Infrared (SWIR) 1',
            'wavelength': '1.57-1.65 µm',
            'applications': 'Moisture content, soil and vegetation moisture',
            'cmap': 'YlOrBr'
        },
        'Band 7': {
            'name': 'Short-wave Infrared (SWIR) 2',
            'wavelength': '2.11-2.29 µm',
            'applications': 'Mineral and rock type discrimination',
            'cmap': 'terrain'
        },
        'DEM': {
            'name': 'Digital Elevation Model',
            'wavelength': 'N/A',
            'applications': 'Topography and terrain analysis',
            'cmap': 'terrain'
        },
        'Temperature': {
            'name': 'Air Temperature',
            'wavelength': 'N/A',
            'applications': 'Climate and environmental monitoring',
            'cmap': 'RdYlBu_r'
        },
        'Precipitation': {
            'name': 'Precipitation',
            'wavelength': 'N/A',
            'applications': 'Hydrological analysis',
            'cmap': 'Blues'
        }
    }
    
    def create_plot(filename):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            # Load the data
            filepath = os.path.join(processed_data_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            
            # Create figure with enhanced layout
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
            
            # Plot each band with enhanced visualization
            for i, (band_name, info) in enumerate(band_info.items()):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                
                # Configure visualization based on band type
                if 'Temperature' in band_name:
                    vmin = np.nanpercentile(data[i], 2)
                    vmax = np.nanpercentile(data[i], 98)
                    title_suffix = '(K)'
                elif 'Precipitation' in band_name:
                    vmin = 0
                    vmax = np.nanpercentile(data[i], 98)
                    title_suffix = '(mm)'
                elif 'DEM' in band_name:
                    vmin = np.nanpercentile(data[i], 2)
                    vmax = np.nanpercentile(data[i], 98)
                    title_suffix = '(m)'
                else:
                    vmin = np.nanpercentile(data[i], 2)
                    vmax = np.nanpercentile(data[i], 98)
                    title_suffix = f'\n{info["wavelength"]}'
                
                # Create image with enhanced colormap
                im = ax.imshow(data[i], cmap=info['cmap'], vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Enhanced title with wavelength and name
                title = f"{band_name}\n{info['name']}\n{title_suffix}"
                ax.set_title(title, fontsize=10, pad=10)
                ax.axis('off')
            
            plt.suptitle(f'Landsat 8 Band Analysis - {filename.split(".")[0]}', 
                        y=1.02, fontsize=16)
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Enhanced statistics with comprehensive metrics
            stats_html = f"""
            <h4>Band Statistics for {filename.split('.')[0]}</h4>
            <div style='height: 150px; overflow-y: auto;'>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f2f2f2; position: sticky; top: 0;'>
                    <th style='padding: 8px;'>Band</th>
                    <th style='padding: 8px;'>Mean</th>
                    <th style='padding: 8px;'>Median</th>
                    <th style='padding: 8px;'>Std Dev</th>
                    <th style='padding: 8px;'>Valid %</th>
                </tr>
            """
            
            for i, (band_name, info) in enumerate(band_info.items()):
                band_data = data[i]
                valid_data = band_data[~np.isnan(band_data)]
                valid_percentage = (len(valid_data) / band_data.size) * 100
                
                stats_html += f"""
                <tr>
                    <td style='padding: 8px;'><b>{band_name}</b><br>
                        <span style='font-size: 0.8em;'>{info['name']}</span></td>
                    <td style='padding: 8px;'>{np.nanmean(band_data):.3f}</td>
                    <td style='padding: 8px;'>{np.nanmedian(band_data):.3f}</td>
                    <td style='padding: 8px;'>{np.nanstd(band_data):.3f}</td>
                    <td style='padding: 8px;'>{valid_percentage:.1f}%</td>
                </tr>
                """
            
            stats_html += """
            </table>
            </div>
            <div style='margin-top: 10px; padding: 8px; background-color: #f8f9fa; 
                       border: 1px solid #dee2e6; border-radius: 4px;'>
                <h5>Band Applications:</h5>
                <div style='height: 100px; overflow-y: auto;'>
            """
            
            for band_name, info in band_info.items():
                stats_html += f"""
                <p style='margin: 5px 0;'><b>{band_name}:</b> {info['applications']}</p>
                """
            
            stats_html += """
                </div>
            </div>
            """
            
            stats_text.value = stats_html
    
    def update_display(change):
        filename = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(filename)
    
    # Register the update function
    date_slider.observe(update_display, names='value')
    
    # Initial plot
    create_plot(date_slider.options[0][1])
    
    # Create help text widget
    help_text = widgets.HTML("""
    <div style='margin-bottom: 10px; padding: 8px; background-color: #e9ecef; 
               border-radius: 4px;'>
        <h4>🔍 Band Viewer Guide:</h4>
        <ul>
            <li>Use the slider to navigate through different dates</li>
            <li>Each band is shown with its specific colormap and wavelength range</li>
            <li>Statistics table shows key metrics for each band</li>
            <li>Band applications section provides context for each band's use</li>
        </ul>
    </div>
    """)
    
    return widgets.VBox([
        help_text,
        date_slider,
        stats_text,
        plot_output
    ])

def create_map_with_slider(processed_data_list, bounds, roi_gdf):
    """Create interactive map with time slider for land use visualization."""
    map_center = [26.907436, 75.794157]
    zoom_level = 10
    m = geemap.Map(center=map_center, zoom=zoom_level)
    
    # Add ROI
    m.add_gdf(roi_gdf, "Region of Interest")
    
    # Enhanced classification colors
    classification_colors = {
        0: '#FFD700',  # Barren Land - Yellow
        1: '#006400',  # Dense Vegetation - Dark Green
        2: '#90EE90',  # Moderate Vegetation - Light Green
        3: '#FF0000',  # Urban Areas - Red
        4: '#00008B',  # Water Bodies - Dark Blue
        5: '#DEB887',  # Bare Soil - Burlywood
        6: '#FA8072'   # Mixed Urban - Salmon
    }
    
    # Create time slider
    slider = widgets.SelectionSlider(
        options=[(data['date_str'], i) for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    current_layers = []
    
    def custom_colormap(classification):
        colored = np.zeros((*classification.shape, 3), dtype=np.uint8)
        for class_val, color in classification_colors.items():
            mask = (classification == class_val)
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            for i, c in enumerate(rgb):
                colored[mask, i] = c
        return colored
    
    def update_layers(change):
        nonlocal current_layers
        
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        
        for layer in current_layers:
            if layer in m.layers:
                m.remove_layer(layer)
        current_layers.clear()
        
        data = processed_data_list[idx]
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
    
    slider.observe(update_layers, names='value')
    
    # Updated legend with new classes
    legend_dict = {
        'Barren Land': '#FFD700',
        'Dense Vegetation': '#006400',
        'Moderate Vegetation': '#90EE90',
        'Urban Areas': '#FF0000',
        'Water Bodies': '#00008B',
        'Bare Soil': '#DEB887',
        'Mixed Urban': '#FA8072'
    }
    m.add_legend(title="Land Use Classification", legend_dict=legend_dict)
    
    slider_control = WidgetControl(widget=slider, position='topright')
    m.add_control(slider_control)
    
    update_layers({'new': (processed_data_list[0]['date_str'], 0)})
    
    return m

def create_spectral_indices_viewer(processed_data_list):
    """Create viewer for enhanced spectral indices."""
    plt.ioff()
    
    date_slider = widgets.SelectionSlider(
        options=[(data['date_str'], i) for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='200px')
    )
    
    plot_output = widgets.Output()
    
    def create_plot(idx):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            data = processed_data_list[idx]
            
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            
            # Configure plots for each index
            plots_config = {
                'NDVI': {'pos': (0, 0), 'cmap': 'RdYlGn', 'title': 'NDVI\n(Vegetation)'},
                'MNDWI': {'pos': (0, 1), 'cmap': 'RdYlBu', 'title': 'MNDWI\n(Water)'},
                'NDBI': {'pos': (1, 0), 'cmap': 'RdYlBu_r', 'title': 'NDBI\n(Built-up)'},
                'UI': {'pos': (1, 1), 'cmap': 'YlOrRd', 'title': 'UI\n(Urban)'},
                'BSI': {'pos': (2, 0), 'cmap': 'YlOrBr', 'title': 'BSI\n(Bare Soil)'},
                'EBBI': {'pos': (2, 1), 'cmap': 'YlOrRd', 'title': 'EBBI\n(Enhanced Built-up)'}
            }
            
            for name, config in plots_config.items():
                ax = fig.add_subplot(gs[config['pos']])
                im = ax.imshow(data[name], cmap=config['cmap'], 
                             vmin=-1 if name != 'EBBI' else None, 
                             vmax=1 if name != 'EBBI' else None)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(config['title'])
                ax.axis('off')
            
            plt.suptitle(f'Spectral Indices Analysis - {data["date_str"]}', y=1.02, fontsize=16)
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Update statistics with more comprehensive metrics
            stats_html = f"""
            <h4>Index Statistics for {data['date_str']}</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f2f2f2;'>
                    <th style='padding: 8px;'>Index</th>
                    <th style='padding: 8px;'>Mean</th>
                    <th style='padding: 8px;'>Std Dev</th>
                    <th style='padding: 8px;'>Coverage*</th>
                </tr>
            """
            
            for name in plots_config.keys():
                arr = data[name]
                threshold = 0.2 if name in ['NDVI', 'MNDWI'] else 0
                coverage = np.mean(arr > threshold) * 100
                
                stats_html += f"""
                <tr>
                    <td style='padding: 8px;'>{name}</td>
                    <td style='padding: 8px;'>{np.nanmean(arr):.3f}</td>
                    <td style='padding: 8px;'>{np.nanstd(arr):.3f}</td>
                    <td style='padding: 8px;'>{coverage:.1f}%</td>
                </tr>
                """
            
            stats_html += """
            </table>
            <p style='font-size: 0.8em;'>*Coverage shows percentage of pixels above typical thresholds for each index</p>
            """
            
            stats_text.value = stats_html
    
    def update_display(change):
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(idx)
    
    date_slider.observe(update_display, names='value')
    create_plot(0)
    
    return widgets.VBox([
        widgets.HTML("<h2>Enhanced Spectral Indices Analysis</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])

def create_classification_viewer(processed_data_list):
    """Create viewer for enhanced land use classification."""
    plt.ioff()
    
    date_slider = widgets.SelectionSlider(
        options=[(data['date_str'], i) for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='200px')
    )
    
    plot_output = widgets.Output()
    
    class_info = {
        'Barren Land': {'value': 0, 'color': '#FFD700', 'cmap': 'YlOrBr'},
        'Dense Vegetation': {'value': 1, 'color': '#006400', 'cmap': 'Greens'},
        'Moderate Vegetation': {'value': 2, 'color': '#90EE90', 'cmap': 'YlGn'},
        'Urban Areas': {'value': 3, 'color': '#FF0000', 'cmap': 'Reds'},
        'Water Bodies': {'value': 4, 'color': '#00008B', 'cmap': 'Blues'},
        'Bare Soil': {'value': 5, 'color': '#DEB887', 'cmap': 'YlOrBr'},
        'Mixed Urban': {'value': 6, 'color': '#FA8072', 'cmap': 'RdGy'}
    }
    
    colors = [info['color'] for info in class_info.values()]
    custom_cmap = ListedColormap(colors)
    
    def create_plot(idx):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            data = processed_data_list[idx]
            
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Individual class maps
            for i, (label, info) in enumerate(class_info.items()):
                if i < 8:  # Now we have 7 classes
                    ax = fig.add_subplot(gs[i // 3, i % 3])
                    mask = data['classification'] == info['value']
                    im = ax.imshow(mask, cmap=info['cmap'])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(label)
                    ax.axis('off')
            
            # Combined map
            ax_combined = fig.add_subplot(gs[2, 2])
            im_combined = ax_combined.imshow(data['classification'], 
                                           cmap=custom_cmap, 
                                           vmin=0, 
                                           vmax=len(class_info)-1)
            cbar = fig.colorbar(im_combined, ax=ax_combined, 
                              fraction=0.046, 
                              pad=0.04)
            cbar.set_ticks(range(len(class_info)))
            cbar.set_ticklabels(class_info.keys())
            ax_combined.set_title('Combined Classification')
            ax_combined.axis('off')
            
            plt.suptitle(f'Land Use Classification - {data["date_str"]}', 
                        y=1.02, fontsize=16)
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Enhanced statistics
            total_pixels = np.size(data['classification'])
            stats_html = f"""
            <h4>Land Cover Statistics for {data['date_str']}</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f2f2f2;'>
                    <th style='padding: 8px;'>Class</th>
                    <th style='padding: 8px;'>Coverage</th>
                    <th style='padding: 8px;'>Pixels</th>
                    <th style='padding: 8px;'>Change*</th>
                </tr>
            """
            
            for label, info in class_info.items():
                class_pixels = np.sum(data['classification'] == info['value'])
                percentage = (class_pixels / total_pixels) * 100
                
                # Calculate change if not first image
                change_str = "N/A"
                if idx > 0:
                    prev_data = processed_data_list[idx - 1]
                    prev_pixels = np.sum(prev_data['classification'] == info['value'])
                    change = ((class_pixels - prev_pixels) / prev_pixels) * 100
                    change_str = f"{change:+.1f}%"
                
                stats_html += f"""
                <tr>
                    <td style='padding: 8px;'>{label}</td>
                    <td style='padding: 8px;'>{percentage:.1f}%</td>
                    <td style='padding: 8px;'>{class_pixels:,}</td>
                    <td style='padding: 8px;'>{change_str}</td>
                </tr>
                """
            
            stats_html += """
            </table>
            <p style='font-size: 0.8em;'>*Change shows percentage change from previous date</p>
            """
            
            stats_text.value = stats_html
    
    def update_display(change):
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(idx)
    
    date_slider.observe(update_display, names='value')
    create_plot(0)
    
    return widgets.VBox([
        widgets.HTML("<h2>Enhanced Land Cover Classification</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])

def plot_time_series(results):
    """Plot enhanced time series analysis."""
    plt.close('all')
    
    df = pd.DataFrame(results)
    df.set_index('date_str', inplace=True)
    
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Spectral indices subplot
    ax1 = fig.add_subplot(gs[0, 0])
    df[['mean_ndvi', 'mean_mndwi', 'mean_ndbi', 'mean_ui', 'mean_bsi']].plot(ax=ax1)
    ax1.set_title('Spectral Indices Trends')
    ax1.set_ylabel('Index Value')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(['NDVI', 'MNDWI', 'NDBI', 'UI', 'BSI'])
    ax1.grid(True)
    
    # Land cover percentages
    ax2 = fig.add_subplot(gs[0, 1])
    df[['barren_percent', 'dense_veg_percent', 'mod_veg_percent', 
        'urban_percent', 'water_percent', 'bare_soil_percent', 
        'mixed_urban_percent']].plot(ax=ax2)
    ax2.set_title('Land Cover Distribution')
    ax2.set_ylabel('Coverage (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(['Barren', 'Dense Veg.', 'Mod. Veg.', 'Urban', 'Water', 
               'Bare Soil', 'Mixed Urban'])
    ax2.grid(True)
    
    # Urban and vegetation comparison
    ax3 = fig.add_subplot(gs[1, 0])
    total_urban = df['urban_percent'] + df['mixed_urban_percent']
    total_veg = df['dense_veg_percent'] + df['mod_veg_percent']
    pd.DataFrame({
        'Total Urban': total_urban,
        'Total Vegetation': total_veg,
        'Urban Ratio': total_urban / total_veg
    }).plot(ax=ax3, secondary_y='Urban Ratio')
    ax3.set_title('Urban vs Vegetation Trends')
    ax3.set_ylabel('Coverage (%)')
    ax3.right_ax.set_ylabel('Urban/Vegetation Ratio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True)
    
    # Environmental parameters
    ax4 = fig.add_subplot(gs[1, 1])
    df['mean_temp'].plot(ax=ax4, color='red', label='Temperature')
    ax4_twin = ax4.twinx()
    df['total_precipitation'].plot(ax=ax4_twin, color='blue', 
                                 label='Precipitation', kind='bar', alpha=0.3)
    ax4.set_title('Environmental Parameters')
    ax4.set_ylabel('Temperature (K)')
    ax4_twin.set_ylabel('Precipitation (mm)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True)
    
    # Built-up area change rate
    ax5 = fig.add_subplot(gs[2, 0])
    urban_change = total_urban.pct_change() * 100
    urban_change.plot(kind='bar', ax=ax5, color='red', alpha=0.6)
    ax5.set_title('Urban Expansion Rate')
    ax5.set_ylabel('Change Rate (%)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True)
    
    # Cumulative changes
    ax6 = fig.add_subplot(gs[2, 1])
    cumulative_changes = pd.DataFrame({
        'Urban': (total_urban - total_urban.iloc[0]),
        'Vegetation': (total_veg - total_veg.iloc[0]),
        'Water': (df['water_percent'] - df['water_percent'].iloc[0])
    }).cumsum()
    cumulative_changes.plot(ax=ax6)
    ax6.set_title('Cumulative Land Cover Changes')
    ax6.set_ylabel('Cumulative Change (%)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True)
    
    plt.suptitle('Enhanced Time Series Analysis of Land Use Changes', 
                 y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_time_series(data_dir):
    """Enhanced time series analysis with additional metrics."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    def process_file(file_path, data_dir):
        try:
            data = process_landsat_data(os.path.join(data_dir, file_path))
            
            # Calculate class percentages
            classification = data['classification']
            total_pixels = classification.size
            class_counts = np.bincount(classification.ravel(), minlength=7)
            
            return {
                'date_str': data['date_str'],
                'date_obj': data['date'],
                'barren_percent': class_counts[0] / total_pixels * 100,
                'dense_veg_percent': class_counts[1] / total_pixels * 100,
                'mod_veg_percent': class_counts[2] / total_pixels * 100,
                'urban_percent': class_counts[3] / total_pixels * 100,
                'water_percent': class_counts[4] / total_pixels * 100,
                'bare_soil_percent': class_counts[5] / total_pixels * 100,
                'mixed_urban_percent': class_counts[6] / total_pixels * 100,
                'mean_ndvi': np.nanmean(data['NDVI']),
                'mean_mndwi': np.nanmean(data['MNDWI']),
                'mean_ndbi': np.nanmean(data['NDBI']),
                'mean_ui': np.nanmean(data['UI']),
                'mean_bsi': np.nanmean(data['BSI']),
                'mean_ebbi': np.nanmean(data['EBBI']),
                'mean_temp': np.nanmean(data['air_temp']),
                'total_precipitation': np.nansum(data['precipitation'])
            }
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(partial(process_file, data_dir=data_dir), 
                                  file_list))
    
    # Filter out None results and sort by date
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x['date_obj'])
    
    return results

def create_prediction_map_with_slider(processed_data_list, bounds, roi_gdf):
    """Create interactive map with time slider for predictions."""
    map_center = [26.907436, 75.794157]
    zoom_level = 10
    m = geemap.Map(center=map_center, zoom=zoom_level)
    
    # Add ROI
    m.add_gdf(roi_gdf, "Region of Interest")
    
    # Enhanced classification colors
    classification_colors = {
        0: '#FFD700',  # Barren Land - Yellow
        1: '#006400',  # Dense Vegetation - Dark Green
        2: '#90EE90',  # Moderate Vegetation - Light Green
        3: '#FF0000',  # Urban Areas - Red
        4: '#00008B',  # Water Bodies - Dark Blue
        5: '#DEB887',  # Bare Soil - Burlywood
        6: '#FA8072'   # Mixed Urban - Salmon
    }
    
    # Create slider with prediction labels
    slider_options = []
    for i, data in enumerate(processed_data_list):
        date_str = data['date_str']
        if 'type' in data and data['type'] == 'Predicted':
            date_str = f"{date_str} (Predicted)"
        slider_options.append((date_str, i))
    
    slider = widgets.SelectionSlider(
        options=slider_options,
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    current_layers = []
    
    def custom_colormap(classification, is_prediction=False):
        colored = np.zeros((*classification.shape, 4), dtype=np.uint8)  # Added alpha channel
        for class_val, color in classification_colors.items():
            mask = (classification == class_val)
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            for i, c in enumerate(rgb):
                colored[mask, i] = c
            colored[mask, 3] = 180 if is_prediction else 255  # Transparency for predictions
        return colored
    
    def update_layers(change):
        nonlocal current_layers
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        
        for layer in current_layers:
            if layer in m.layers:
                m.remove_layer(layer)
        current_layers.clear()
        
        data = processed_data_list[idx]
        is_prediction = 'type' in data and data['type'] == 'Predicted'
        
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
        
        if is_prediction:
            prediction_text = widgets.HTML(
                value='<div style="background-color: rgba(255,255,0,0.2); padding: 8px; '
                      'border-radius: 4px;"><b>⚠️ Predicted Data</b></div>'
            )
            if not hasattr(m, 'prediction_control'):
                m.prediction_control = WidgetControl(widget=prediction_text, position='topright')
                m.add_control(m.prediction_control)
        else:
            if hasattr(m, 'prediction_control'):
                m.remove_control(m.prediction_control)
                delattr(m, 'prediction_control')
    
    slider.observe(update_layers, names='value')
    
    # Updated legend with new classes
    legend_dict = {
        'Barren Land': '#FFD700',
        'Dense Vegetation': '#006400',
        'Moderate Vegetation': '#90EE90',
        'Urban Areas': '#FF0000',
        'Water Bodies': '#00008B',
        'Bare Soil': '#DEB887',
        'Mixed Urban': '#FA8072'
    }
    m.add_legend(title="Land Use Classification", legend_dict=legend_dict)
    
    slider_control = WidgetControl(widget=slider, position='topright')
    m.add_control(slider_control)
    
    update_layers({'new': (processed_data_list[0]['date_str'], 0)})
    
    return m

def create_prediction_viewer(processed_data_list):
    """Create viewer for predicted land use classifications."""
    plt.ioff()
    
    date_slider = widgets.SelectionSlider(
        options=[(f"{data['date_str']} {'(Predicted)' if 'type' in data and data['type'] == 'Predicted' else ''}", i) 
                for i, data in enumerate(processed_data_list)],
        description='Date:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    stats_text = widgets.HTML(
        layout=widgets.Layout(width='500px', height='250px')
    )
    
    plot_output = widgets.Output()
    
    class_info = {
        'Barren Land': {'value': 0, 'color': '#FFD700', 'cmap': 'YlOrBr'},
        'Dense Vegetation': {'value': 1, 'color': '#006400', 'cmap': 'Greens'},
        'Moderate Vegetation': {'value': 2, 'color': '#90EE90', 'cmap': 'YlGn'},
        'Urban Areas': {'value': 3, 'color': '#FF0000', 'cmap': 'Reds'},
        'Water Bodies': {'value': 4, 'color': '#00008B', 'cmap': 'Blues'},
        'Bare Soil': {'value': 5, 'color': '#DEB887', 'cmap': 'YlOrBr'},
        'Mixed Urban': {'value': 6, 'color': '#FA8072', 'cmap': 'RdGy'}
    }
    
    combined_colors = [info['color'] for info in class_info.values()]
    custom_cmap = ListedColormap(combined_colors)
    
    def create_plot(idx):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            data = processed_data_list[idx]
            is_prediction = 'type' in data and data['type'] == 'Predicted'
            
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Individual class maps
            for i, (label, info) in enumerate(class_info.items()):
                if i < 8:  # 7 classes + combined view
                    ax = fig.add_subplot(gs[i // 3, i % 3])
                    mask = data['classification'] == info['value']
                    im = ax.imshow(mask, cmap=info['cmap'])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    title = f"{label}\n"
                    if is_prediction:
                        title += "(Predicted)"
                        if 'confidence' in data:
                            conf_mask = data['confidence'][mask]
                            avg_conf = np.mean(conf_mask) if len(conf_mask) > 0 else 0
                            title += f"\nConf: {avg_conf:.2f}"
                    ax.set_title(title)
                    ax.axis('off')
            
            # Combined view
            ax_combined = fig.add_subplot(gs[2, 2])
            im_combined = ax_combined.imshow(data['classification'], 
                                           cmap=custom_cmap, 
                                           vmin=0, 
                                           vmax=len(class_info)-1)
            cbar = fig.colorbar(im_combined, ax=ax_combined, 
                              fraction=0.046, 
                              pad=0.04)
            cbar.set_ticks(range(len(class_info)))
            cbar.set_ticklabels(class_info.keys())
            
            title = "Combined Classification\n"
            if is_prediction:
                title += "(Predicted)"
            ax_combined.set_title(title)
            ax_combined.axis('off')
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Enhanced statistics
            total_pixels = np.size(data['classification'])
            stats_html = f"""
            <h4>Land Cover Statistics for {data['date_str']}</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color: #f2f2f2;'>
                    <th style='padding: 8px;'>Class</th>
                    <th style='padding: 8px;'>Coverage</th>
                    <th style='padding: 8px;'>Pixels</th>
                    <th style='padding: 8px;'>Confidence*</th>
                </tr>
            """
            
            for label, info in class_info.items():
                class_pixels = np.sum(data['classification'] == info['value'])
                percentage = (class_pixels / total_pixels) * 100
                
                confidence_str = "N/A"
                if is_prediction and 'confidence' in data:
                    mask = data['classification'] == info['value']
                    conf_values = data['confidence'][mask]
                    if len(conf_values) > 0:
                        confidence_str = f"{np.mean(conf_values):.2f}"
                
                stats_html += f"""
                <tr>
                    <td style='padding: 8px;'>{label}</td>
                    <td style='padding: 8px;'>{percentage:.1f}%</td>
                    <td style='padding: 8px;'>{class_pixels:,}</td>
                    <td style='padding: 8px;'>{confidence_str}</td>
                </tr>
                """
            
            stats_html += "</table>"
            
            if is_prediction:
                stats_html += f"""
                <div style='margin-top: 10px; padding: 8px; background-color: #fff3cd; 
                           border: 1px solid #ffeeba; border-radius: 4px;'>
                    <strong>⚠️ Predicted Classification</strong><br>
                    Mean Confidence: {np.mean(data['confidence']):.3f}
                </div>
                """
            
            stats_text.value = stats_html
    
    def update_display(change):
        idx = change['new'][1] if isinstance(change['new'], tuple) else change['new']
        create_plot(idx)
    
    date_slider.observe(update_display, names='value')
    create_plot(0)
    
    return widgets.VBox([
        widgets.HTML("<h2>Land Cover Prediction Viewer</h2>"),
        date_slider,
        stats_text,
        plot_output
    ])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

def visualize_predictions(predictions):
    """
    Visualize prediction results with enhanced visualization and analysis.
    
    Args:
        predictions: List of prediction dictionaries containing:
            - classification: numpy array of predicted classes
            - confidence: numpy array of prediction confidences
            - date_str: string representation of prediction date
    """
    if not predictions:
        print("No predictions to visualize")
        return
        
    n_predictions = len(predictions)
    fig = plt.figure(figsize=(20, 6 * n_predictions))
    gs = gridspec.GridSpec(n_predictions, 3, figure=fig)
    
    # Maintain consistency with established color scheme
    class_colors = [
        '#FFD700',  # Barren Land - Yellow
        '#006400',  # Dense Vegetation - Dark Green
        '#90EE90',  # Moderate Vegetation - Light Green
        '#FF0000',  # Urban Areas - Red
        '#00008B',  # Water Bodies - Dark Blue
        '#DEB887',  # Bare Soil - Burlywood
        '#FA8072'   # Mixed Urban - Salmon
    ]
    class_cmap = ListedColormap(class_colors)
    
    # Use consistent class mapping from the model
    class_mapping = {
        0: 'Barren Land',
        1: 'Dense Vegetation',
        2: 'Moderate Vegetation',
        3: 'Urban Areas',
        4: 'Water Bodies',
        5: 'Bare Soil',
        6: 'Mixed Urban'
    }
    
    for idx, pred in enumerate(predictions):
        try:
            # Get unique classes in this prediction
            unique_classes = np.unique(pred['classification'])
            
            # Classification plot with consistent coloring
            ax1 = fig.add_subplot(gs[idx, 0])
            im1 = ax1.imshow(pred['classification'], cmap=class_cmap)
            ax1.set_title(f"Predicted Land Use\n{pred['date_str']}", pad=10)
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_ticks(range(len(class_mapping)))
            cbar1.set_ticklabels([class_mapping[i] for i in range(len(class_mapping))])
            ax1.axis('off')
            
            # Confidence plot with RdYlGn colormap for consistency
            ax2 = fig.add_subplot(gs[idx, 1])
            im2 = ax2.imshow(pred['confidence'], cmap='RdYlGn')
            ax2.set_title(f"Prediction Confidence\n{pred['date_str']}", pad=10)
            plt.colorbar(im2, ax=ax2)
            ax2.axis('off')
            
            # Distribution plot with enhanced statistics
            ax3 = fig.add_subplot(gs[idx, 2])
            counts = np.bincount(pred['classification'].ravel(), minlength=len(class_mapping))
            total = counts.sum()
            percentages = counts / total * 100
            
            # Only plot bars for classes that are present
            present_classes = unique_classes
            present_labels = [class_mapping[cls] for cls in present_classes]
            present_colors = [class_colors[cls] for cls in present_classes]
            present_percentages = [percentages[cls] for cls in present_classes]
            
            x_positions = np.arange(len(present_classes))
            bars = ax3.bar(x_positions, present_percentages, color=present_colors)
            
            ax3.set_title(f"Class Distribution and Confidence\n{pred['date_str']}", pad=10)
            ax3.set_ylabel('Coverage (%)')
            
            # Set x-axis labels only for present classes
            ax3.set_xticks(x_positions)
            ax3.set_xticklabels(present_labels, rotation=45, ha='right')
            
            # Add confidence statistics on bars
            for bar, cls in zip(bars, present_classes):
                mask = pred['classification'] == cls
                conf_values = pred['confidence'][mask]
                mean_conf = np.mean(conf_values)
                std_conf = np.std(conf_values)
                
                # Add mean and std confidence on bars
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'Conf: {mean_conf:.2f}\n±{std_conf:.2f}',
                        ha='center', va='bottom', fontsize=8)
            
            # Add grid for better readability
            ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Add analysis text box
            class_changes = []
            if idx > 0:
                prev_pred = predictions[idx - 1]
                for cls in present_classes:
                    prev_count = np.sum(prev_pred['classification'] == cls)
                    curr_count = np.sum(pred['classification'] == cls)
                    if prev_count > 0:
                        change_pct = ((curr_count - prev_count) / prev_count) * 100
                        class_changes.append(f"{class_mapping[cls]}: {change_pct:+.1f}%")
            
            analysis_text = "Monthly Changes:\n" + "\n".join(class_changes) if class_changes else ""
            if analysis_text:
                ax3.text(1.15, 0.5, analysis_text,
                        transform=ax3.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                        verticalalignment='center',
                        fontsize=8)
        
        except Exception as e:
            print(f"Error plotting prediction {idx}: {str(e)}")
            continue
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nDetailed Prediction Statistics:")
    for idx, pred in enumerate(predictions):
        print(f"\nPrediction {idx+1} - {pred['date_str']}:")
        unique, counts = np.unique(pred['classification'], return_counts=True)
        total = counts.sum()
        
        print("\nClass Distribution and Confidence:")
        for cls in sorted(unique):
            mask = pred['classification'] == cls
            conf_values = pred['confidence'][mask]
            count = np.sum(mask)
            percentage = (count / total) * 100
            
            print(f"\n{class_mapping[cls]}:")
            print(f"  Coverage: {percentage:.1f}% ({count:,} pixels)")
            print(f"  Confidence: {np.mean(conf_values):.3f} ± {np.std(conf_values):.3f}")
            print(f"  Range: {np.min(conf_values):.3f} - {np.max(conf_values):.3f}")
            
        print(f"\nOverall Confidence: {np.mean(pred['confidence']):.3f} ± "
              f"{np.std(pred['confidence']):.3f}")
        
        if idx > 0:
            prev_pred = predictions[idx - 1]
            changes = pred['classification'] != prev_pred['classification']
            change_percentage = np.mean(changes) * 100
            print(f"\nChanges from previous prediction: {change_percentage:.1f}% of pixels")
            
class ModelAnalysis:
    """Enhanced class for analyzing model performance and predictions."""
    
    @staticmethod
    def analyze_class_distribution(predictions):
        """Analyze class distribution over time with enhanced metrics."""
        results = []
        for pred in predictions:
            total_pixels = pred['classification'].size
            distribution = {
                'date': pred['date_str'],
                'total_pixels': total_pixels
            }
            
            # Calculate class percentages and confidence
            unique, counts = np.unique(pred['classification'], return_counts=True)
            for cls in range(7):  # 7 classes
                cls_name = f"class_{cls}_percent"
                mask = pred['classification'] == cls
                count = np.sum(mask)
                distribution[cls_name] = (count / total_pixels) * 100
                
                if 'confidence' in pred:
                    conf_name = f"class_{cls}_confidence"
                    conf_values = pred['confidence'][mask]
                    distribution[conf_name] = np.mean(conf_values) if len(conf_values) > 0 else 0
            
            results.append(distribution)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def analyze_confidence(predictions):
        """Analyze prediction confidence with detailed metrics."""
        results = []
        for pred in predictions:
            confidence = pred['confidence']
            class_conf = {}
            
            # Calculate overall confidence metrics
            overall_metrics = {
                'date': pred['date_str'],
                'mean_confidence': np.mean(confidence),
                'median_confidence': np.median(confidence),
                'min_confidence': np.min(confidence),
                'max_confidence': np.max(confidence),
                'std_confidence': np.std(confidence),
                'q25_confidence': np.percentile(confidence, 25),
                'q75_confidence': np.percentile(confidence, 75)
            }
            
            # Calculate class-specific confidence
            for cls in range(7):  # 7 classes
                mask = pred['classification'] == cls
                if np.any(mask):
                    cls_conf = confidence[mask]
                    class_conf.update({
                        f'class_{cls}_mean_conf': np.mean(cls_conf),
                        f'class_{cls}_std_conf': np.std(cls_conf),
                        f'class_{cls}_min_conf': np.min(cls_conf),
                        f'class_{cls}_max_conf': np.max(cls_conf)
                    })
                else:
                    class_conf.update({
                        f'class_{cls}_mean_conf': 0,
                        f'class_{cls}_std_conf': 0,
                        f'class_{cls}_min_conf': 0,
                        f'class_{cls}_max_conf': 0
                    })
            
            results.append({**overall_metrics, **class_conf})
        
        return pd.DataFrame(results)
    
    @staticmethod
    def analyze_spatial_changes(predictions):
        """Analyze spatial changes between predictions with enhanced metrics."""
        results = []
        for i in range(len(predictions) - 1):
            current = predictions[i]['classification']
            next_pred = predictions[i + 1]['classification']
            current_conf = predictions[i]['confidence']
            next_conf = predictions[i + 1]['confidence']
            
            # Calculate basic change metrics
            changes = current != next_pred
            change_percentage = np.mean(changes) * 100
            
            # Calculate transition matrix
            transition_matrix = np.zeros((7, 7))  # 7x7 for all classes
            for from_class in range(7):
                for to_class in range(7):
                    mask = (current == from_class) & (next_pred == to_class)
                    transition_matrix[from_class, to_class] = np.sum(mask)
            
            # Calculate confidence of changes
            change_conf_current = current_conf[changes]
            change_conf_next = next_conf[changes]
            
            result = {
                'from_date': predictions[i]['date_str'],
                'to_date': predictions[i + 1]['date_str'],
                'change_percentage': change_percentage,
                'pixels_changed': np.sum(changes),
                'total_pixels': changes.size,
                'mean_conf_changed_from': np.mean(change_conf_current),
                'mean_conf_changed_to': np.mean(change_conf_next),
                'stable_pixels': np.sum(~changes),
                'mean_conf_stable': np.mean(current_conf[~changes])
            }
            
            # Add transition matrix to results
            for from_class in range(7):
                for to_class in range(7):
                    key = f'transition_{from_class}_to_{to_class}'
                    result[key] = transition_matrix[from_class, to_class]
            
            # Calculate class-specific change rates
            for cls in range(7):
                mask_current = current == cls
                mask_next = next_pred == cls
                
                if np.any(mask_current):
                    stability = np.sum(mask_current & mask_next) / np.sum(mask_current) * 100
                    result[f'class_{cls}_stability'] = stability
                    result[f'class_{cls}_loss'] = 100 - stability
                else:
                    result[f'class_{cls}_stability'] = 0
                    result[f'class_{cls}_loss'] = 0
                
                if np.any(mask_next):
                    result[f'class_{cls}_gain'] = np.sum(mask_next & ~mask_current) / np.sum(mask_next) * 100
                else:
                    result[f'class_{cls}_gain'] = 0
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def generate_summary_report(predictions):
        """Generate a comprehensive analysis report."""
        class_names = ['Barren Land', 'Dense Vegetation', 'Moderate Vegetation',
                      'Urban Areas', 'Water Bodies', 'Bare Soil', 'Mixed Urban']
        
        # Get all analyses
        dist_df = ModelAnalysis.analyze_class_distribution(predictions)
        conf_df = ModelAnalysis.analyze_confidence(predictions)
        change_df = ModelAnalysis.analyze_spatial_changes(predictions)
        
        report = {
            'overview': {
                'total_predictions': len(predictions),
                'time_span': f"{predictions[0]['date_str']} to {predictions[-1]['date_str']}",
                'mean_overall_confidence': conf_df['mean_confidence'].mean()
            },
            'class_metrics': {},
            'change_metrics': {
                'mean_change_rate': change_df['change_percentage'].mean(),
                'total_changed_pixels': change_df['pixels_changed'].sum(),
                'mean_stability': change_df['stable_pixels'].mean() / change_df['total_pixels'].mean() * 100
            },
            'confidence_trends': {
                'overall_trend': conf_df['mean_confidence'].diff().mean(),
                'by_class': {}
            }
        }
        
        # Calculate class-specific metrics
        for i, class_name in enumerate(class_names):
            class_data = {
                'mean_coverage': dist_df[f'class_{i}_percent'].mean(),
                'coverage_trend': dist_df[f'class_{i}_percent'].diff().mean(),
                'mean_confidence': conf_df[f'class_{i}_mean_conf'].mean(),
                'stability': change_df[f'class_{i}_stability'].mean() if not change_df.empty else 0,
                'gain_rate': change_df[f'class_{i}_gain'].mean() if not change_df.empty else 0,
                'loss_rate': change_df[f'class_{i}_loss'].mean() if not change_df.empty else 0
            }
            report['class_metrics'][class_name] = class_data
            
            # Add confidence trends
            report['confidence_trends']['by_class'][class_name] = \
                conf_df[f'class_{i}_mean_conf'].diff().mean()
        
        return report
    
    @staticmethod
    def plot_analysis_results(predictions):
        """Create comprehensive visualization of analysis results."""
        # Create distribution analysis
        dist_df = ModelAnalysis.analyze_class_distribution(predictions)
        conf_df = ModelAnalysis.analyze_confidence(predictions)
        change_df = ModelAnalysis.analyze_spatial_changes(predictions)
        
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot class distribution over time
        ax1 = fig.add_subplot(gs[0, 0])
        class_cols = [col for col in dist_df.columns if 'class' in col and 'percent' in col]
        dist_df[class_cols].plot(ax=ax1)
        ax1.set_title('Class Distribution Over Time')
        ax1.set_ylabel('Coverage (%)')
        ax1.legend(['Barren', 'Dense Veg', 'Mod Veg', 'Urban', 'Water', 
                   'Bare Soil', 'Mixed Urban'], 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot confidence trends
        ax2 = fig.add_subplot(gs[0, 1])
        conf_cols = [col for col in conf_df.columns if 'mean_conf' in col]
        conf_df[conf_cols].plot(ax=ax2)
        ax2.set_title('Confidence Trends by Class')
        ax2.set_ylabel('Confidence Score')
        
        # Plot change rates
        ax3 = fig.add_subplot(gs[1, 0])
        change_df['change_percentage'].plot(kind='bar', ax=ax3)
        ax3.set_title('Change Rates Between Predictions')
        ax3.set_ylabel('Change (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot class stability
        ax4 = fig.add_subplot(gs[1, 1])
        stability_cols = [col for col in change_df.columns if 'stability' in col]
        change_df[stability_cols].mean().plot(kind='bar', ax=ax4)
        ax4.set_title('Class Stability')
        ax4.set_ylabel('Stability (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot urban growth trend
        ax5 = fig.add_subplot(gs[2, 0])
        urban_cols = ['class_3_percent', 'class_6_percent']  # Urban and Mixed Urban
        total_urban = dist_df[urban_cols].sum(axis=1)
        total_urban.plot(ax=ax5, marker='o')
        ax5.set_title('Urban Growth Trend')
        ax5.set_ylabel('Total Urban Coverage (%)')
        
        # Plot transition summary
        ax6 = fig.add_subplot(gs[2, 1])
        transition_cols = [col for col in change_df.columns if 'transition' in col]
        transition_means = change_df[transition_cols].mean()
        transition_means.plot(kind='bar', ax=ax6)
        ax6.set_title('Average Class Transitions')
        ax6.set_ylabel('Pixels')
        ax6.tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        plt.show()
        
