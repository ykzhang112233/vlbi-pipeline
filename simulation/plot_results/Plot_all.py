import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
import os
import csv
import pandas as pd

# Set Nature-style formatting
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (7.2, 7.2),  # Nature single column width is ~8.5 cm = 3.35 inches
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.25,
    'ytick.minor.width': 0.25,
})

def resolve_input_file(file_name):
    """
    Resolve the actual data file to use, preferring CSV when available.

    Parameters:
    ----------
    file_name : str
        Desired file name (may be .txt or .csv)

    Returns:
    -------
    resolved_path : str or None
        Existing path to use, preferring .csv over .txt if both are options.
    """
    # If the provided file exists, use it directly
    print(file_name)
    if os.path.exists(file_name):
        print(f"Using provided file: {file_name}")
        return file_name

    stem, ext = os.path.splitext(file_name)
    ext = ext.lower()

    # Prefer .csv if .txt requested but .csv exists
    if ext == ".txt":
        csv_path = stem + ".csv"
        if os.path.exists(csv_path):
            return csv_path
    # If .csv requested but not found, fall back to .txt when available
    if ext == ".csv":
        txt_path = stem + ".txt"
        if os.path.exists(txt_path):
            return txt_path

    # Try generic preference: csv then txt
    csv_path = stem + ".csv"
    if os.path.exists(csv_path):
        return csv_path
    txt_path = stem + ".txt"
    if os.path.exists(txt_path):
        return txt_path

    return None

def load_multi_column_data(file_path):
    """
    Load multi-column measurement data from a text file and convert units.
    
    Parameters:
    ----------
    file_path : str
        Path to the data file
    
    Returns:
    -------
    data_dict : dict or None
        Dictionary containing arrays for each column
        Keys: 'flux_mjy', 'x_mas', 'y_mas', 'major_fwhm_mas'
    """
    try:
        # Initialize lists for each column
        flux_data = []
        x_data = []
        y_data = []
        size_data = []
        
        ext = os.path.splitext(file_path)[1].lower()
        
        # Branch parsing based on extension (CSV vs whitespace text)
        if ext == ".csv":
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                for line_num, row in enumerate(reader, start=1):
                    # Skip empty rows
                    if not row:
                        continue
                    # Allow comments beginning with '#'
                    if row[0].strip().startswith('#'):
                        continue
                    # Strip whitespace and ignore empty strings
                    parts = [c.strip() for c in row if c is not None and str(c).strip() != ""]
                    if len(parts) < 4:
                        print(f"Warning: Row {line_num} in {file_path} has {len(parts)} columns, expected at least 4")
                        continue
                    try:
                        flux = float(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        size = float(parts[3])
                        
                        # Unit conversions
                        flux_data.append(flux * 1000.0)
                        x_data.append(x * 1000.0)
                        y_data.append(y * 1000.0)
                        size_data.append(size)
                    except ValueError:
                        # Likely a header row; skip gracefully
                        if line_num == 1:
                            continue
                        print(f"Warning: Row {line_num} in {file_path} has invalid data: {row}")
                        continue
        else:
            # Default: whitespace-separated text
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    
                    # Skip empty lines and comment lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Split line into columns
                    parts = line.split()
                    
                    if len(parts) >= 4:
                        try:
                            flux = float(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            size = float(parts[3])
                            
                            # Unit conversions
                            flux_data.append(flux * 1000.0)
                            x_data.append(x * 1000.0)
                            y_data.append(y * 1000.0)
                            size_data.append(size)
                            
                        except ValueError:
                            print(f"Warning: Line {line_num+1} in {file_path} has invalid data: {line}")
                            continue
                    else:
                        print(f"Warning: Line {line_num+1} in {file_path} has {len(parts)} columns, expected at least 4")
                        continue
        
        if len(size_data) == 0:
            print(f"Warning: No valid data in {file_path}")
            return None
        
        # Convert to numpy arrays with converted units
        data_dict = {
            'flux_mjy': np.array(flux_data),       # Now in mJy
            'x_mas': np.array(x_data),             # Now in mas
            'y_mas': np.array(y_data),             # Now in mas
            'major_fwhm_mas': np.array(size_data)  # Already in mas
        }
        
        return data_dict
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_multi_column_data_pandas(file_path):
    """
    Load multi-column measurement data using pandas and convert units.

    This function expects four columns in order:
      1) flux (Jy), 2) x (arcsec), 3) y (arcsec), 4) major FWHM (mas)
    It returns the same dictionary format as `load_multi_column_data`:
      {'flux_mjy', 'x_mas', 'y_mas', 'major_fwhm_mas'}

    Supports both CSV and whitespace-separated TXT. Lines starting with '#'
    are treated as comments and ignored. Non-numeric rows (e.g., headers)
    are dropped.

    Parameters:
    ----------
    file_path : str
        Path to the data file (.csv or .txt)

    Returns:
    -------
    dict or None
        Dictionary of numpy arrays with converted units, or None if failed.
    """

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            # CSV: auto-detect separator; ignore comments; no header assumed
            df = pd.read_csv(
                file_path,
                comment='#',
                sep=",",
                engine='python'
            )
        else:
            # TXT: whitespace-separated
            df = pd.read_csv(
                file_path,
                comment='#',
                header=None,
                delim_whitespace=True
            )

        # Ensure at least 4 columns
        if df.shape[1] < 4:
            print(f"Error: {file_path} has {df.shape[1]} columns, expected at least 4.")
            return None

        # Take first 4 columns and coerce to numeric; drop non-numeric rows
        # df = df.iloc[:, :4].apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) == 0:
            print(f"Warning: No valid numeric rows found in {file_path}")
            return None
        # print(df)
        flux = df['flux_jy']
        x = df['x_arcsec']
        y = df['y_arcsec']
        size = df['major_fwhm_mas']

        data_dict = {
            'flux_mjy': flux * 1000.0,
            'x_mas': x * 1000.0,
            'y_mas': y * 1000.0,
            'major_fwhm_mas': size
        }

        return data_dict

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading {file_path} with pandas: {e}")
        return None
def plot_epoch_histogram(ax, sizes, epoch_name, subplot_label):
    """
    Plot a histogram with Gaussian fit for a single epoch.
    
    Parameters:
    ----------
    ax : matplotlib axes object
        Axes to plot on
    sizes : numpy array
        Size measurements (major_fwhm_mas)
    epoch_name : str
        Name of the epoch (for title)
    subplot_label : str
        Label for subplot (a, b, c, ...)
    """
    if sizes is None or len(sizes) == 0:
        ax.text(0.5, 0.5, f"No data for\n{epoch_name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Size (mas)')
        ax.set_ylabel('Probability Density')
        return
    
    # Calculate statistics
    mean_val = np.mean(sizes)
    std_val = np.std(sizes)
    median_val = np.median(sizes)
    min_val = np.min(sizes)
    max_val = np.max(sizes)
    
    # Plot histogram
    n, bins, patches = ax.hist(sizes, bins=25, edgecolor='black', 
                               alpha=0.7, density=True, 
                               color='steelblue', linewidth=0.5)
    
    # Plot Gaussian fit
    x = np.linspace(np.min(sizes), np.max(sizes), 1000)
    pdf = norm.pdf(x, mean_val, std_val)
    ax.plot(x, pdf, 'r-', linewidth=1.5) #label='Gaussian fit')
    
    # Plot mean and std lines
    ax.axvline(x=mean_val, color='red', linestyle='--', 
               linewidth=1.5, alpha=0.8) #, label=f'Mean = {mean_val:.3f}')
    ax.axvline(x=mean_val + std_val, color='green', linestyle=':', 
               linewidth=1.2, alpha=0.7) #, label=f'±1σ = {std_val:.3f}')
    ax.axvline(x=mean_val - std_val, color='green', linestyle=':', 
               linewidth=1.2, alpha=0.7)
    
    # Fill 1σ region
    ax.fill_betweenx([0, np.max(pdf)], 
                     mean_val - std_val, mean_val + std_val, 
                     alpha=0.2, color='gray')
    
    # Add epoch name and statistics as text
    stats_text = f'{epoch_name}\nμ = {mean_val:.3f}\nσ = {std_val:.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # Add subplot label (a, b, c, ...) in top-left corner
    ax.text(0.02, 0.98, f'{subplot_label}', transform=ax.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top')

    ax.set_xlim(left=min_val * 0.9, right=max_val * 1.2)
    
    # Set labels and grid
    ax.set_xlabel('Time-varied Size (mas)', fontsize=9)
    ax.set_ylabel('Probability Density', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

def create_nature_style_plot(
        file_names: list = None
        ):
    """
    Create a 3x3 grid of histograms for all epochs.
    Returns figure and a list of data dictionaries for each file.
    """
    # Define file lists
    # freq_15ghz_files = ['BA161a.txt', 'BA161b.txt', 'BA161c.txt']
    # freq_8ghz_files = ['BL307b.txt', 'BL307c.txt', 'BL307d.txt', 
    #                    'BL307e.txt', 'BL307f.txt', 'BL307g.txt']
    
    # # Combine all files (desired names)
    # all_files = freq_15ghz_files + freq_8ghz_files
    all_files = file_names
    all_epoch_names = ['BA161a', 'BA161b', 'BA161c',
                      'BL307b', 'BL307c', 'BL307d',
                      'BL307e', 'BL307f', 'BL307g']
    
    # Resolve to actual existing paths (prefer CSV when present)
    resolved_paths = []
    missing_files = []
    for f in all_files:
        rp = resolve_input_file(f)
        if rp is None:
            missing_files.append(f)
        else:
            resolved_paths.append(rp)
    if missing_files:
        print(f"Warning: The following files are missing: {missing_files}")
        print("Please ensure all data files (.csv or .txt) are in the current directory.")
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 7.2))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration
    
    # List to store all data for statistics output
    all_data = []
    
    # Load data and create plots
    for i, (file_name, epoch_name) in enumerate(zip(all_files, all_epoch_names)):
        if i >= len(axes):
            break
        
        actual_path = resolve_input_file(file_name)
        if actual_path is None:
            print(f"Skipping {file_name}: no .csv or .txt found.")
            sizes = None
            data_dict = None
        else:
            print(f"Processing {file_name} (using {os.path.basename(actual_path)})...")
            
            # Load data (returns dictionary with 4 arrays, already converted)
            # data_dict = load_multi_column_data(actual_path)
            data_dict = load_multi_column_data_pandas(actual_path)
            if data_dict is not None:
                sizes = data_dict['major_fwhm_mas']
            else:
                sizes = None
        if data_dict is not None:
            # Extract size data for plotting
            
            # Store all data for later statistics output
            all_data.append({
                'file_name': actual_path if actual_path else file_name,
                'epoch_name': epoch_name,
                'data_dict': data_dict
            })
            
            # Plot histogram (using only size data)
            plot_epoch_histogram(axes[i], sizes, epoch_name, chr(97+i))  # 97 = 'a' in ASCII
        else:
            # If no data, turn off the axis
            axes[i].text(0.5, 0.5, f"No data for\n{epoch_name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xlabel('Size (mas)')
            axes[i].set_ylabel('Probability Density')
            axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Add subplot label
            axes[i].text(0.02, 0.98, f'{chr(97+i)}', transform=axes[i].transAxes,
                        fontsize=10, fontweight='bold', verticalalignment='top')
    
    # Remove any unused axes (if we have fewer than 9 files)
    for i in range(len(all_files), len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    return fig, all_data

def calculate_column_statistics(data_array, column_name):
    """
    Calculate statistics for a single column of data.
    
    Parameters:
    ----------
    data_array : numpy array
        Array of data values
    column_name : str
        Name of the column (for display purposes)
    
    Returns:
    -------
    stats_dict : dict
        Dictionary containing statistics
    """
    if len(data_array) == 0:
        return None
    
    stats_dict = {
        'column': column_name,
        'n': len(data_array),
        'mean': np.mean(data_array),
        'std': np.std(data_array),
        'median': np.median(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'range': np.max(data_array) - np.min(data_array)
    }
    
    # Add coefficient of variation (if mean is not zero)
    if stats_dict['mean'] != 0:
        stats_dict['cv'] = (stats_dict['std'] / stats_dict['mean']) * 100
    else:
        stats_dict['cv'] = float('inf')
    
    return stats_dict

def save_comprehensive_statistics(all_data):
    """
    Save comprehensive statistics for all four columns to a text file.
    
    Parameters:
    ----------
    all_data : list
        List of dictionaries containing data for each file
    """
    with open('multi_column_statistics_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE STATISTICS SUMMARY FOR MULTI-COLUMN DATA\n")
        f.write("=" * 80 + "\n\n")
        f.write("NOTE: Units have been converted:\n")
        f.write("  - Flux: Jy → mJy (×1000)\n")
        f.write("  - Position offsets: arcsec → mas (×1000)\n")
        f.write("  - Size: already in mas\n")
        f.write("=" * 80 + "\n\n")
        
        for file_data in all_data:
            file_name = file_data['file_name']
            epoch_name = file_data['epoch_name']
            data_dict = file_data['data_dict']
            
            f.write(f"File: {file_name} (Epoch: {epoch_name})\n")
            f.write("-" * 80 + "\n")
            
            # Calculate and write statistics for each column
            columns = ['flux_mjy', 'x_mas', 'y_mas', 'major_fwhm_mas']
            column_display_names = ['Flux (mJy)', 'X offset (mas)', 'Y offset (mas)', 'Major FWHM (mas)']
            
            for col_key, display_name in zip(columns, column_display_names):
                if col_key in data_dict and len(data_dict[col_key]) > 0:
                    stats = calculate_column_statistics(data_dict[col_key], display_name)
                    
                    if stats:
                        f.write(f"  {display_name}:\n")
                        f.write(f"    Number of measurements: {stats['n']}\n")
                        f.write(f"    Mean: {stats['mean']:.6f}\n")
                        f.write(f"    Standard deviation: {stats['std']:.6f}\n")
                        f.write(f"    Median: {stats['median']:.6f}\n")
                        f.write(f"    Minimum: {stats['min']:.6f}\n")
                        f.write(f"    Maximum: {stats['max']:.6f}\n")
                        f.write(f"    Range: {stats['range']:.6f}\n")
                        
                        if 'cv' in stats and stats['cv'] != float('inf'):
                            f.write(f"    Coefficient of variation: {stats['cv']:.2f}%\n")
                        
                        f.write("\n")
                else:
                    f.write(f"  {display_name}: No data available\n\n")
            
            f.write("=" * 80 + "\n\n")
        
        # Add summary table for size column (major_fwhm_mas) across all files
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY OF SIZE MEASUREMENTS (Major FWHM in mas)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':<10} {'N':<6} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        
        for file_data in all_data:
            epoch_name = file_data['epoch_name']
            data_dict = file_data['data_dict']
            
            if 'major_fwhm_mas' in data_dict and len(data_dict['major_fwhm_mas']) > 0:
                sizes = data_dict['major_fwhm_mas']
                f.write(f"{epoch_name:<10} {len(sizes):<6} {np.mean(sizes):<12.6f} {np.std(sizes):<12.6f} "
                       f"{np.median(sizes):<12.6f} {np.min(sizes):<12.6f} {np.max(sizes):<12.6f}\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Add summary table for flux measurements across all files
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY OF FLUX MEASUREMENTS (Flux in mJy)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':<10} {'N':<6} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        
        for file_data in all_data:
            epoch_name = file_data['epoch_name']
            data_dict = file_data['data_dict']
            
            if 'flux_mjy' in data_dict and len(data_dict['flux_mjy']) > 0:
                fluxes = data_dict['flux_mjy']
                f.write(f"{epoch_name:<10} {len(fluxes):<6} {np.mean(fluxes):<12.6f} {np.std(fluxes):<12.6f} "
                       f"{np.median(fluxes):<12.6f} {np.min(fluxes):<12.6f} {np.max(fluxes):<12.6f}\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Add note about data format and unit conversions
        f.write("\nDATA FORMAT AND UNIT CONVERSIONS:\n")
        f.write("-" * 80 + "\n")
        f.write("Input data files contain four columns:\n")
        f.write("  1. flux_jy: Flux density in Jansky (Jy)\n")
        f.write("  2. x_arcsec: X offset in arcseconds (arcsec)\n")
        f.write("  3. y_arcsec: Y offset in arcseconds (arcsec)\n")
        f.write("  4. major_fwhm_mas: Major axis FWHM in milliarcseconds (mas)\n\n")
        f.write("Unit conversions applied by the program:\n")
        f.write("  1. flux_jy → flux_mjy: Multiply by 1000 (1 Jy = 1000 mJy)\n")
        f.write("  2. x_arcsec → x_mas: Multiply by 1000 (1 arcsec = 1000 mas)\n")
        f.write("  3. y_arcsec → y_mas: Multiply by 1000 (1 arcsec = 1000 mas)\n")
        f.write("  4. major_fwhm_mas: No conversion needed (already in mas)\n")
        f.write("=" * 80 + "\n")
    
    print("Comprehensive statistics summary saved to: multi_column_statistics_summary.txt")

def main():
    """
    Main function to create and save the plot.
    """
    print("="*80)
    print("Creating Nature-style 3x3 histogram plot for VLBI multi-column data")
    print("="*80)
    print("NOTE: Unit conversions will be applied:")
    print("  - Flux: Jy → mJy (×1000)")
    print("  - Position offsets: arcsec → mas (×1000)")
    print("  - Size: already in mas, no conversion")
    print("="*80)
    
    # Define desired files (will resolve to .csv when present)
    all_files = ['simulated_source_parms_GRB221009A-ba161a1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-ba161b1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-ba161c1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-bl307bx1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-bl307cx1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-bl307dx1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-bl307ex1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-bl307fx1_jk_drop_timeblock_2000.csv',
                'simulated_source_parms_GRB221009A-bl307gx1_jk_drop_timeblock_2000.csv']

    # Resolve and report missing
    missing_files = []
    for f in all_files:
        if resolve_input_file(f) is None:
            missing_files.append(f)
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("Please ensure data files (.csv or .txt) are in the current directory.")
        print("Continuing with available files...\n")
    
    # Create the figure and get data
    fig, all_data = create_nature_style_plot(file_names=all_files)
    
    # Save as PDF with high quality
    pdf_filename = "Time_MC.pdf"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    print(f"\nPlot saved as: {pdf_filename}")
    
    # Save statistics summary
    if all_data:
        save_comprehensive_statistics(all_data)
    else:
        print("Warning: No data available for statistics summary.")
    
    # Print completion message
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"- Histograms of size measurements saved to {pdf_filename}")
    print("- Comprehensive statistics for all four columns saved to multi_column_statistics_summary.txt")
    print("- Units converted: flux (Jy→mJy), positions (arcsec→mas), size (already mas)")
    print("="*80)

if __name__ == "__main__":
    main()