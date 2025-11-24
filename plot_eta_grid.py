import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
from pathlib import Path

def plot_eta_files(eta_path: str | os.PathLike):
    """
    Finds all 'eta_*.npy' files in the current directory,
    loads them, and creates polar plots based on the [:, 0, 0] slice.
    
    The filename format is assumed to be 'eta_{fam}_{m}{n}{p}_{pol}.npy'
    """
    
    # Regex to parse the filename
    # It captures 'fam', 'm', 'n', 'p', and 'pol' parts
    filename_pattern = re.compile(
        r'eta_(?P<fam>.*?)_(?P<m>\d)(?P<n>\d)(?P<p>\d)_(?P<pol>.*?)\.npy'
    )
    
    # Find all files matching the pattern
    eta_files = glob.glob(str(eta_path / 'eta_*.npy'))  # type: ignore
    
    if not eta_files:
        print(f"No 'eta_*.npy' files found in the directory {eta_path}.")
        return

    print(f"Found {len(eta_files)} eta file(s). Processing...")

    for filepath in eta_files:
        filename = os.path.basename(filepath)
        match = filename_pattern.match(filename)
        
        if not match:
            print(f"Skipping '{filename}': Filename does not match expected pattern.")
            continue
            
        # Extract info from the filename
        file_info = match.groupdict()
        p_value = file_info['p']
        title = f"p = {p_value} (fam={file_info['fam']}, pol={file_info['pol']})"
        
        try:
            # Load the data
            eta_data = np.load(filepath)
            
            if eta_data.ndim != 3:
                print(f"Skipping '{filename}': Expected 3D array, but got {eta_data.ndim}D.")
                continue
            
            # --- Get the data slice to plot ---
            # This is the eta(beta) data for the first alpha and psi
            data_to_plot = eta_data[:, 0, 0]

            # manually remove large values
            filtered_data = np.copy(data_to_plot)
            large_value_indices = filtered_data > 1.0
            filtered_data[large_value_indices] = 0.0
            data_to_plot = filtered_data
            
            Nbeta = data_to_plot.shape[0]
            if Nbeta == 0:
                print(f"Skipping '{filename}': Data slice is empty.")
                continue
            
            # --- Prepare data for a closed polar plot ---
            # Create the angle array (theta)
            # We go from 0 to 2*pi with Nbeta points, not including 2*pi itself
            theta = np.linspace(0, 2 * np.pi, Nbeta, endpoint=False)
            
            # To make the plot "closed" (connect end to start),
            # we must append the first data point to the end of the data
            # and append the first angle (0, or 2*pi) to the end of the angles.
            plot_data_closed = np.append(data_to_plot, data_to_plot[0])
            theta_closed = np.append(theta, theta[0]) # Appending 0 is same as 2*pi
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            
            ax.plot(theta_closed, plot_data_closed)
            
            ax.set_title(title, pad=20)
            
            ax.set_theta_zero_location('E')      # type: ignore
            ax.set_theta_direction(1)            # type: ignore
            ax.set_rlabel_position(90)           # type: ignore

            ax.grid(True)
            
            output_filename = f"plot_{filename.replace('.npy', '.png')}"
            plt.savefig(output_filename)
            plt.close(fig)
            
            print(f"Successfully processed '{filename}' -> Saved plot as '{output_filename}'")

        except Exception as e:
            print(f"Error processing '{filename}': {e}")

if __name__ == "__main__":
    plot_eta_files(Path('/data/sguotong/projects/CaGe/data/etas'))
