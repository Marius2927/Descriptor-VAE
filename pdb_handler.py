import Bio.PDB as bio
import numpy as np
import torch
from Bio.PDB import PDBParser


def load_all_atoms(pdb_path: str):
    """
    Parse the input PDB file and return:
      - X         : torch.Tensor of shape [N_atoms, 3], in the exact order atoms appear
      - info_list : list of length N_atoms, where each entry is a dict containing:
            {
              'atom_name': str,
              'res_name' : str,
              'chain_id' : str,
              'res_id'   : int,
              'altloc'   : str,
              'i_code'   : str,
              'occupancy': float,
              'tempfactor': float,
              'element'  : str,
              'charge'   : str,
            }
    Loads both ATOM and HETATM records.
    """
    parser    = bio.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model     = next(structure.get_models())

    atom_coords = []
    info_list   = []

    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            hetflag, resseq, icode = residue.get_id()
            res_name = residue.get_resname()
            for atom in residue:
                coord = atom.get_coord().astype(np.float32)
                atom_name = atom.get_name()
                altloc    = atom.get_altloc()
                element   = atom.element
                occ       = atom.get_occupancy() if atom.get_occupancy() is not None else 1.00
                tfac      = atom.get_bfactor()   if atom.get_bfactor()   is not None else 0.00
                charge    = atom.get_fullname().strip()[-2:].strip()

                atom_coords.append(coord)
                info_list.append({
                    'atom_name' : atom_name,
                    'res_name'  : res_name,
                    'chain_id'  : chain_id,
                    'res_id'    : resseq,
                    'altloc'    : altloc,
                    'i_code'    : icode,
                    'occupancy' : occ,
                    'tempfactor': tfac,
                    'element'   : element.strip(),
                    'charge'    : charge
                })

    if len(atom_coords) == 0:
        raise ValueError(f"No atom records found in {pdb_path}. Please check the file path or its contents.")

    X_np = np.vstack(atom_coords)
    X    = torch.from_numpy(X_np)
    return X, info_list

_ATOM_LINE_FMT = "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"

def get_all_atom_line(idx_atom: int,
                      atom_name: str,
                      res_name: str,
                      chain_id: str,
                      res_id: int,
                      altloc: str,
                      i_code: str,
                      x: float, y: float, z: float,
                      occupancy: float,
                      tempfactor: float,
                      element: str,
                      charge) -> str:
    record   = "ATOM"
    atom_num = idx_atom
    atom_name_field = atom_name.rjust(4)[:4]
    res_name_field  = res_name.ljust(3)[:3]
    chain_field     = (chain_id or " ").ljust(1)
    i_code_field    = (i_code   or " ").ljust(1)
    element_field   = (element or " ").strip().rjust(2)[:2]
    # ensure charge is string of length ≤2
    charge_str      = str(charge) if charge is not None else ""
    charge_field    = charge_str.strip().rjust(2)[:2]

    return _ATOM_LINE_FMT % (
        record,
        atom_num,
        atom_name_field,
        altloc[:1] if altloc else " ",
        res_name_field,
        chain_field,
        res_id,
        i_code_field,
        x, y, z,
        occupancy,
        tempfactor,
        element_field,
        charge_field
    )


def save_backbone_pdb(X_bb: torch.Tensor,
                      info_list: list,
                      out_path: str):
    """
    Write a single‐model PDB containing only the backbone atoms N, CA, C.
    - X_bb: [N_bb, 3] torch.Tensor of reconstructed coords
    - info_list: list of length N_bb of dicts with keys
        'atom_name','res_name','chain_id','res_id',
        'altloc','i_code','occupancy','tempfactor','element','charge'
    """
    assert X_bb.shape[0] == len(info_list), \
        f"Mismatched atom count: coords have {X_bb.shape[0]} rows, info_list has {len(info_list)}"
    with open(out_path, "w") as fp:
        fp.write("MODEL        1\n")
        for idx, info in enumerate(info_list, start=1):
            x, y, z = X_bb[idx-1].tolist()
            line = get_all_atom_line(
                idx_atom   = idx,
                atom_name  = info['atom_name'],
                res_name   = info['res_name'],
                chain_id   = info['chain_id'],
                res_id     = info['res_id'],
                altloc     = info.get('altloc'," "),
                i_code     = info.get('i_code'," "),
                x=x, y=y, z=z,
                occupancy  = info.get('occupancy',1.00),
                tempfactor = info.get('tempfactor',0.00),
                element    = info.get('element'," "),
                charge     = info.get('charge'," ")
            )
            fp.write(line)
        fp.write("ENDMDL\n")


def save_all_atom_pdb(X: torch.Tensor,
                      info_list: list,
                      out_path: str):
    assert X.shape[0] == len(info_list), "Mismatch between X rows and info_list entries."

    with open(out_path, "w") as fp:
        fp.write("MODEL        1\n")
        for idx in range(len(info_list)):
            info = info_list[idx]
            x, y, z = X[idx].tolist()
            line = get_all_atom_line(
                idx_atom   = idx+1,
                atom_name  = info['atom_name'],
                res_name   = info['res_name'],
                chain_id   = info['chain_id'],
                res_id     = info['res_id'],
                altloc     = info['altloc'],
                i_code     = info['i_code'],
                x=x, y=y, z=z,
                occupancy  = info['occupancy'],
                tempfactor = info['tempfactor'],
                element    = info['element'],
                charge     = info['charge']
            )
            fp.write(line)
        fp.write("ENDMDL\n")


def load_backbone_atoms(pdb_path: str):
    """
    Parse a single‐model PDB (or one model of an NMR ensemble),
    return only the backbone atoms (N, CA, C) in sequence order.
    """
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    coords    = []
    info_list = []

    # Here we assume you only care about the *first* model:
    model = next(structure.get_models())

    for chain in model:
        for residue in chain:
            for atom in residue:
                name = atom.get_name()
                if name in ("N", "CA", "C"):
                    coords.append(atom.get_coord().tolist())
                    info_list.append({
                        "atom_name": atom.get_name(),
                        "res_name":  residue.get_resname(),
                        "chain_id":  chain.get_id(),
                        "res_id":    residue.get_id()[1],
                        "altloc":    atom.get_altloc(),
                        "i_code":    residue.get_id()[2],
                        "occupancy": atom.get_occupancy(),
                        "tempfactor": atom.get_bfactor(),
                        "element":   atom.element,
                        "charge":    atom.get_full_id()[-1]
                    })

    if not coords:
        raise ValueError(f"No backbone atoms (N/CA/C) found in {pdb_path}")
    X_bb = torch.tensor(coords, dtype=torch.float32)  # [N_bb, 3]
    return X_bb, info_list