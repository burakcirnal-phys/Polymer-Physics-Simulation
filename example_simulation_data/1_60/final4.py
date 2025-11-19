import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures3 directory
os.makedirs("figures4", exist_ok=True)

# Read ALL timesteps from dump file
def read_all_timesteps(filename):
    """Read all timesteps from LAMMPS dump file"""
    all_timesteps = []
    all_coords = []
    all_types = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if "ITEM: TIMESTEP" in line:
            # Read timestep number
            i += 1
            timestep = int(lines[i].strip())
            all_timesteps.append(timestep)
            i += 1
            
            # Skip ITEM: NUMBER OF ATOMS
            i += 2  # Skip "ITEM: NUMBER OF ATOMS" and the number itself
            
            # Skip ITEM: BOX BOUNDS (3 lines)
            i += 4  # Skip "ITEM: BOX BOUNDS" and 3 box boundary lines
            
            # Now we should be at "ITEM: ATOMS"
            if i < len(lines) and "ITEM: ATOMS" in lines[i]:
                i += 1
                coords, types = [], []
                
                # Read atom data until next ITEM or EOF
                while i < len(lines) and not lines[i].startswith("ITEM:"):
                    parts = lines[i].split()
                    if len(parts) >= 5:
                        types.append(int(parts[1]))
                        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    i += 1
                
                if coords:
                    all_coords.append(np.array(coords))
                    all_types.append(np.array(types))
            else:
                i += 1
        else:
            i += 1
    
    return all_timesteps, all_coords, all_types

# Slab profile functions with 0.5σ bin size
def slab_profile_number_density(coords, mask, axis=0, R=25.0):
    """Formula 1: 1D Number Density ρ_α(x) = N_α(x) / (A * Δx)"""
    dx = 0.5  # σ = 0.5
    bins = np.arange(-R, R + dx, dx)
    
    # Cross-sectional area
    area = (2 * R) * (2 * R)
    
    vals = coords[:, axis]
    cnt, _ = np.histogram(vals[mask], bins=bins)
    density = cnt / (area * dx)  # Number density
    
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, density

def slab_profile_probability_density(coords, mask, total_particles, axis=0, R=25.0):
    """Formula 2: 1D Probability Density p_α(x) = N_α(x) / (N_total,α * A * Δx)"""
    dx = 0.5  # σ = 0.5
    bins = np.arange(-R, R + dx, dx)
    
    # Cross-sectional area
    area = (2 * R) * (2 * R)
    
    vals = coords[:, axis]
    cnt, _ = np.histogram(vals[mask], bins=bins)
    probability_density = cnt / (total_particles * area * dx)  # Probability density
    
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, probability_density

def calculate_composition_fraction_probability(p_A, p_B):
    """Calculate composition fraction using probability density: φ_A = p_A / (p_A + p_B)"""
    phi_A = np.zeros_like(p_A)
    phi_B = np.zeros_like(p_B)
    nonzero = (p_A + p_B) > 0
    phi_A[nonzero] = p_A[nonzero] / (p_A[nonzero] + p_B[nonzero])
    phi_B[nonzero] = p_B[nonzero] / (p_A[nonzero] + p_B[nonzero])
    return phi_A, phi_B

# Parameters
R = 25.0
dx = dy = dz = 0.5  # σ = 0.5
bins_2d = np.arange(-R, R + dx, dx)
# Define angle coefficients for this case
a1_value = 1  # Change this for each case: 1, 1, 1, 20, 20, 20
a2_value = 60  # Change this for each case: 1, 20, 60, 20, 40, 60

# Read ALL timesteps data
print("Reading all timesteps from dump file...")
timesteps, all_coords, all_types = read_all_timesteps("dump.task")
print(f"Found {len(timesteps)} frames")

# Use data for time averaging
coords_instant = (all_coords[-1] - 0.5) * 2 * R
types_instant = all_types[-1]

# Classify types
A_mask_instant = np.isin(types_instant, [1, 2, 3])
B_mask_instant = ~A_mask_instant

# Calculate total particles
N_total_A_instant = len(coords_instant[A_mask_instant])
N_total_B_instant = len(coords_instant[B_mask_instant])
TOTAL_PARTICLES_instant = len(coords_instant)

print(f"\nSystem information:")
print(f"Total particles in system: {TOTAL_PARTICLES_instant}")
print(f"Type A particles: {N_total_A_instant} ({N_total_A_instant/TOTAL_PARTICLES_instant*100:.1f}%)")
print(f"Type B particles: {N_total_B_instant} ({N_total_B_instant/TOTAL_PARTICLES_instant*100:.1f}%)")

# Initialize arrays for time averaging
centers_template, _ = slab_profile_number_density(coords_instant, A_mask_instant, axis=0, R=R)
n_bins = len(centers_template)

time_avg_data = {
    'x': {'rho_A': np.zeros(n_bins), 'rho_B': np.zeros(n_bins), 
          'p_A': np.zeros(n_bins), 'p_B': np.zeros(n_bins)},
    'y': {'rho_A': np.zeros(n_bins), 'rho_B': np.zeros(n_bins), 
          'p_A': np.zeros(n_bins), 'p_B': np.zeros(n_bins)},
    'z': {'rho_A': np.zeros(n_bins), 'rho_B': np.zeros(n_bins), 
          'p_A': np.zeros(n_bins), 'p_B': np.zeros(n_bins)}
}

# Time averaging loop over ALL timesteps
print(f"\nCalculating time averages over {len(timesteps)} frames...")
for i, (coords, types) in enumerate(zip(all_coords, all_types)):
    if i % 10 == 0:  # Progress indicator
        print(f"Processing frame {i+1}/{len(timesteps)}...")
    
    coords_transformed = (coords - 0.5) * 2 * R
    A_mask = np.isin(types, [1, 2, 3])
    B_mask = ~A_mask
    
    N_total = len(coords_transformed[A_mask]) + len(coords_transformed[B_mask])
    
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        # Calculate instantaneous profiles
        centers, rho_A_inst = slab_profile_number_density(coords_transformed, A_mask, axis=axis_idx, R=R)
        centers, rho_B_inst = slab_profile_number_density(coords_transformed, B_mask, axis=axis_idx, R=R)
        centers, p_A_inst = slab_profile_probability_density(coords_transformed, A_mask, N_total, axis=axis_idx, R=R)
        centers, p_B_inst = slab_profile_probability_density(coords_transformed, B_mask, N_total, axis=axis_idx, R=R)
        
        # Accumulate for time averaging
        time_avg_data[axis_name]['rho_A'] += rho_A_inst
        time_avg_data[axis_name]['rho_B'] += rho_B_inst
        time_avg_data[axis_name]['p_A'] += p_A_inst
        time_avg_data[axis_name]['p_B'] += p_B_inst

# Calculate final time averages
for axis_name in ['x', 'y', 'z']:
    for key in ['rho_A', 'rho_B', 'p_A', 'p_B']:
        time_avg_data[axis_name][key] /= len(timesteps)

print("Time averaging complete!")

# Analyze for each coordinate axis - ONLY TIME-AVERAGED PLOTS (3 subplots)
for axis, coord_name in enumerate(['x', 'y', 'z']):
    print(f"\n=== Analyzing {coord_name}-axis ===")
    
    # Get TIME-AVERAGED data
    centers, _ = slab_profile_number_density(coords_instant, A_mask_instant, axis=axis, R=R)
    rho_A_avg = time_avg_data[coord_name]['rho_A']
    rho_B_avg = time_avg_data[coord_name]['rho_B']
    p_A_avg = time_avg_data[coord_name]['p_A']
    p_B_avg = time_avg_data[coord_name]['p_B']
    
    # Calculate TIME-AVERAGED composition fraction
    # New way (using probability density):
    phi_A_avg, phi_B_avg = calculate_composition_fraction_probability(p_A_avg, p_B_avg)
    
    # Normalize time-averaged probability density
    p_A_avg_norm = p_A_avg / p_A_avg.max() if p_A_avg.max() > 0 else p_A_avg
    p_B_avg_norm = p_B_avg / p_B_avg.max() if p_B_avg.max() > 0 else p_B_avg
    
    # Create subplots for this coordinate - 1 ROW, 3 COLUMNS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Time-Averaged Number Density
    y_max_rho = max(rho_A_avg.max(), rho_B_avg.max())
    axes[0].plot(centers, rho_A_avg, label='ρ_A', color='blue', linewidth=2)
    axes[0].plot(centers, rho_B_avg, label='ρ_B', color='red', linewidth=2)
    axes[0].set_ylim(0, 0.1)
    axes[0].set_xlabel(f'{coord_name} position')
    axes[0].set_ylabel('Number density ρ (particles/σ³)')
    axes[0].set_title(f'Time-Averaged: Number Density\n({len(timesteps)} frames)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Time-Averaged Probability Density (Y-limits 0.5 to 1)
    axes[1].plot(centers, p_A_avg_norm, label='p_A', color='blue', linewidth=2)
    axes[1].plot(centers, p_B_avg_norm, label='p_B', color='red', linewidth=2)
    axes[1].set_ylim(0.5, 1.0)  # Fixed y-limits 0.5 to 1
    axes[1].set_xlabel(f'{coord_name} position')
    axes[1].set_ylabel('Normalized probability density')
    axes[1].set_title(f'Time-Averaged: Probability Density\n({len(timesteps)} frames)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Time-Averaged Composition Fraction φ (ZOOMED)
    axes[2].plot(centers, phi_A_avg, '-o', label='φ_A', color='blue', markersize=2, linewidth=2)
    axes[2].plot(centers, phi_B_avg, '-o', label='φ_B', color='red', markersize=2, linewidth=2)
    axes[2].axhline(0.5, color='k', linestyle='--', label='mixed (0.5)', alpha=0.7)
    
    # Aggressive zoom - focus on the actual data range
    phi_min = min(phi_A_avg.min(), phi_B_avg.min())
    phi_max = max(phi_A_avg.max(), phi_B_avg.max())
    data_range = phi_max - phi_min
    
    if data_range > 0:
        # Only 5% padding for tight zoom
        y_min_zoom = max(0, phi_min - data_range * 0.05)
        y_max_zoom = min(1, phi_max + data_range * 0.05)
        axes[2].set_ylim(0.3, 0.65)
    else:
        # If data is constant, use a small range around the value
        center_val = (phi_A_avg[0] + phi_B_avg[0]) / 2
        axes[2].set_ylim(center_val - 0.1, center_val + 0.1)
    
    axes[2].set_xlabel(f'{coord_name} position')
    axes[2].set_ylabel('Composition fraction φ')
    axes[2].set_title(f'Time-Averaged: Composition Fraction (Zoomed)\n({len(timesteps)} frames)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures4/slab_profiles_{coord_name}_time_avg.png", dpi=300, bbox_inches='tight')
    plt.show()

# 2D Probability Density Difference Map - TIME-AVERAGED ONLY
print("\n=== Creating 2D Time-Averaged Probability Density Difference Map ===")

voxel_volume_2d = dx * dy * (2 * R)  # Δx * Δy * L_z

# Initialize 2D arrays for time averaging
H_A_accum = np.zeros((len(bins_2d)-1, len(bins_2d)-1))
H_B_accum = np.zeros((len(bins_2d)-1, len(bins_2d)-1))

print("Calculating 2D time averages...")
for i, (coords, types) in enumerate(zip(all_coords, all_types)):
    if i % 10 == 0:
        print(f"Processing 2D frame {i+1}/{len(timesteps)}...")
    
    coords_transformed = (coords - 0.5) * 2 * R
    A_mask = np.isin(types, [1, 2, 3])
    B_mask = ~A_mask
    
    # 2D histograms for probability density
    H_A, xedges, yedges = np.histogram2d(coords_transformed[A_mask, 0], coords_transformed[A_mask, 1], bins=[bins_2d, bins_2d])
    H_B, _, _ = np.histogram2d(coords_transformed[B_mask, 0], coords_transformed[B_mask, 1], bins=[bins_2d, bins_2d])
    
    H_A_accum += H_A
    H_B_accum += H_B

# Calculate time-averaged 2D probability density
H_A_avg = H_A_accum / len(timesteps)
H_B_avg = H_B_accum / len(timesteps)

# Convert to probability density
p_A_2d_avg = H_A_avg / (N_total_A_instant * voxel_volume_2d)
p_B_2d_avg = H_B_avg / (N_total_B_instant * voxel_volume_2d)

# Calculate normalized probability density difference
p_diff_avg = p_B_2d_avg - p_A_2d_avg 
p_diff_norm_avg = p_diff_avg / np.max(np.abs(p_diff_avg))  # Normalize to [-1, 1]

# Create 2D map
plt.figure(figsize=(8, 6))
im = plt.imshow(p_diff_norm_avg.T, origin='lower', extent=[-R, R, -R, R], 
                cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
plt.colorbar(im, label='Normalized Probability Density Difference (p_B - p_A)')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title(f'2D Time-Averaged Probability Density Difference\np_B(x,y) - p_A(x,y) ({len(timesteps)} frames)')
plt.tight_layout()
plt.savefig("figures4/2d_probability_density_difference_time_avg.png", dpi=300, bbox_inches='tight')
# Save composition data for summary plots
np.savez(f"composition_data_a1_{a1_value}_a2_{a2_value}.npz", 
         centers=centers, 
         phi_A=phi_A_avg, 
         phi_B=phi_B_avg)
plt.show()

print("\n=== All analyses complete ===")
print(f"Analyzed {len(timesteps)} frames total")
print("All plots saved to 'figures3' folder:")
print("- figures3/slab_profiles_x_time_avg.png")
print("- figures3/slab_profiles_y_time_avg.png") 
print("- figures3/slab_profiles_z_time_avg.png")
print("- figures3/2d_probability_density_difference_time_avg.png")