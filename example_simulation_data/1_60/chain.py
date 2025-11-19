import numpy as np # type: ignore
from itertools import count
from pathlib import Path

# --- Parameters ---
box_size = 10  # box dimensions (Lx = Ly = Lz = 50)
num_beads_per_chain = 15  # 1 head + 13 middle + 1 tail
num_chains = 1000 # fixed number of chains
monomer_diameter = 1.0  # sigma
bond_length = 1  # slightly shorter than diameter

# --- Constants ---
masses = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # arbitrary but identical

# --- ID Counters ---
atom_id_counter = count(1)
bond_id_counter = count(1)
angle_id_counter = count(1)

# --- Containers ---
atoms = []
bonds = []
angles = []

box_size = 30.0  # Now box spans -15 to 15 in all directions
max_chain_half_extent = (num_beads_per_chain - 1) * bond_length / 2  # Half-length of chain

def generate_chain(chain_id):
    # Choose axis-aligned direction
    axis = np.random.choice(3)
    sign = np.random.choice([-1, 1])
    direction = np.zeros(3)
    direction[axis] = sign

    # Limit origin so full chain fits inside [-15, 15]
    origin = np.zeros(3)
    for i in range(3):
        if direction[i] != 0:
            origin[i] = np.random.uniform(-15 + max_chain_half_extent, 15 - max_chain_half_extent)
        else:
            origin[i] = np.random.uniform(-15, 15)

    positions = [origin + i * bond_length * direction for i in range(num_beads_per_chain)]

    local_atoms = []
    local_bonds = []
    local_angles = []
    
    for i, pos in enumerate(positions):
        atom_id = next(atom_id_counter)
        if chain_id <= num_chains // 2:
            atom_type = 1 if i == 0 else 3 if i == num_beads_per_chain - 1 else 2
        else:
            atom_type = 4 if i == 0 else 6 if i == num_beads_per_chain - 1 else 5
        x, y, z = pos
        image_flags = (0, 0, 0)
        local_atoms.append((atom_id, chain_id, atom_type, x, y, z, *image_flags))

        if i > 0:
            bond_id = next(bond_id_counter)
            bond_type = 1 if chain_id <= num_chains // 2 else 2
            local_bonds.append((bond_id, bond_type, atom_id - 1, atom_id))

        if i > 1:
            angle_id = next(angle_id_counter)
            angle_type = 1 if chain_id <= num_chains // 2 else 2
            local_angles.append((angle_id, angle_type, atom_id - 2, atom_id - 1, atom_id))
    
    return local_atoms, local_bonds, local_angles


# --- Generate chains ---
for chain_id in range(1, num_chains + 1):
    a, b, ag = generate_chain(chain_id)
    atoms.extend(a)
    bonds.extend(b)
    angles.extend(ag)

# --- Write to LAMMPS data file ---
data_file_path = Path("solution.data")

with open(data_file_path, "w") as f:
    f.write("LAMMPS data file for semiflexible chains, 50^3 box, 50 chains of 15 beads\n\n")
    f.write(f"{len(atoms)} atoms\n")
    f.write(f"{len(bonds)} bonds\n")
    f.write(f"{len(angles)} angles\n")
    f.write("6 atom types\n2 bond types\n2 angle types\n\n")
    f.write(f"{-25.000} {25.000} xlo xhi\n")
    f.write(f"{-25.000} {25.000} ylo yhi\n")
    f.write(f"{-25.000} {25.000} zlo zhi\n\n")

    f.write("Masses\n\n")
    for t, m in masses.items():
        f.write(f"{t} {m}\n")

    f.write("\nAtoms\n\n")
    for atom in atoms:
        atom_id, mol_id, atom_type, x, y, z, ix, iy, iz = atom
        f.write(f"{atom_id} {mol_id} {atom_type} {x:.5f} {y:.5f} {z:.5f} {ix} {iy} {iz}\n")

    f.write("\nBonds\n\n")
    for bond in bonds:
        bond_id, bond_type, a1, a2 = bond
        f.write(f"{bond_id} {bond_type} {a1} {a2}\n")

    f.write("\nAngles\n\n")
    for angle in angles:
        angle_id, angle_type, a1, a2, a3 = angle
        f.write(f"{angle_id} {angle_type} {a1} {a2} {a3}\n")

print(f"Wrote file to {data_file_path.resolve()}")

