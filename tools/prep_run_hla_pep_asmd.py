#!/usr/bin/env python

from pathlib import Path

import numpy as np

from biomol import protein
from biomol.args import asmd_args
from biomol.md_utils import base_md_utils


def prep_md_sim_base_process(
    *,
    inpdb: str,
    md_sys_name: str = "md_system",
    fftype: str = "ff14SB",
    water_model: str = "tip3p",
    water_box_size: float = 15.0,
    ewald_cut: float = 12.0,
    target_temp: float = 300.0,
    prep_dir: str = "1_prep",
    min_dir: str = "2_min",
    equil_dir: str = "3_equilibration",
):
    # NOTE:
    # Preparing MD system
    print("@ Preparation of MD system")
    # NOTE:
    # check length of antigen peptide to set water_box size properly.
    chains = {}
    template_chain_ids = ["A", "B"]
    with open(inpdb, "r") as hla_pep_com:
        this_chid = "@"
        chain_order = 0
        for line in hla_pep_com:
            if line.startswith("ATOM"):
                if line[21] == " ":
                    this_chid = template_chain_ids[chain_order]
                elif line[21] != " " and line[21] != this_chid:
                    # has a chain id and be a new chain
                    this_chid = line[21]

                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    if this_chid not in chains:
                        chains[this_chid] = []
                    chains[this_chid].append(
                        np.array(
                            [
                                line[30:38].strip(),
                                line[38:46].strip(),
                                line[46:54].strip(),
                            ],
                            dtype=np.float64,
                        )
                    )
            elif line.startswith("TER"):
                chain_order += 1
    chain_length = {chid: len(reslist) for chid, reslist in chains.items()}
    n_pro_res = 0
    for _, length in chain_length.items():
        n_pro_res += length
    pep_chid = min(chain_length, key=chain_length.get)  # pyright: ignore[reportUnusedVariable, reportCallIssue, reportArgumentType]
    hla_chid = max(chain_length, key=chain_length.get)  # pyright: ignore[reportUnusedVariable, reportCallIssue, reportArgumentType]
    pep_residue = chains[pep_chid]
    nt_pep_ca = pep_residue[0]
    ct_pep_ca = pep_residue[-1]
    peptide_length = np.linalg.norm(nt_pep_ca - ct_pep_ca)
    if water_box_size <= float(peptide_length) + 10.0:
        water_box_size = float(peptide_length) + 10.0

    # NOTE: Teleap process
    init_solv_pdb, solv_top, init_solv_crd = base_md_utils.setup_md_system(
        inpdb=inpdb,
        md_sys_name=md_sys_name,
        fftype=fftype,
        water_model=water_model,
        water_box_size=water_box_size,
        prep_dir=prep_dir,
    )
    if Path(init_solv_pdb).exists():
        Path(init_solv_pdb).unlink()
    Path(init_solv_pdb).symlink_to(f"./{prep_dir}/{init_solv_pdb}")

    if Path(solv_top).exists():
        Path(solv_top).unlink()
    Path(solv_top).symlink_to(f"./{prep_dir}/{solv_top}")

    if Path(init_solv_crd).exists():
        Path(init_solv_crd).unlink()
    Path(init_solv_crd).symlink_to(f"./{prep_dir}/{init_solv_crd}")

    print("@ Setup minimization step")
    final_min_rst = base_md_utils.setup_conventional_minimization_steps(
        init_str_crd=init_solv_crd,
        md_top=solv_top,
        min_dir=min_dir,
        ewald_cut=ewald_cut,
        protein_residue=[1, n_pro_res],
        write_pdb_each_step=True,
    )

    print("@ Setup equilibration steps (heating and density equilibration)")
    final_equil_rst = base_md_utils.setup_conventional_equilibration_steps(
        system_name=md_sys_name,
        init_str_crd=f"../{min_dir}/{final_min_rst}",
        md_top=solv_top,
        protein_residue=[1, n_pro_res],
        equil_dir=equil_dir,
        ewald_cut=ewald_cut,
        target_temp=target_temp,
        raising_temp_interval=50.0,
        write_pdb_each_step=True,
    )

    print("@ Setup ASMD replicas")
    print(
        "  - Finding steering points: antigen peptide's C-Term CA and its nearest CA atoms"
    )
    with open(init_solv_pdb, "r") as preped_pdb:
        for line in preped_pdb:
            if line.startswith("ATOM"):
                atmnam = line[12:16].strip()
                atmid = int(line[6:11].strip())
                resid = int(line[22:26].strip())
                if atmnam == "CA":
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())


def parse_hla_pep_system(*, inpdb: str):
    # NOTE:
    # Detect antigen peptides
    print("@ Detecting HLA and antigen part")
    hla_pep = protein.Protein(infile=inpdb)

    residues = hla_pep.get_residues()
    ch_nres = {chid: len(reslist) for chid, reslist in residues.items()}
    hla_chid = max(ch_nres, key=ch_nres.get)  # pyright: ignore[reportUnusedVariable, reportCallIssue, reportArgumentType]
    pep_chid = min(ch_nres, key=ch_nres.get)  # pyright: ignore[reportUnusedVariable, reportCallIssue, reportArgumentType]

    hla_residues = residues[hla_chid]
    hla_res_ids = hla_pep.get_residue_ids(chainid=hla_chid)
    pep_residues = residues[pep_chid]
    pep_res_ids = hla_pep.get_residue_ids(chainid=pep_chid)

    print(f" - HLA     Chain id: {hla_chid} - {len(hla_res_ids)} residues")
    print(f" - Antigen Chain id: {pep_chid} - {len(pep_res_ids)} residues")
    # print(pep_res_ids)
    # print(pep_residues)
    # print(hla_res_ids)
    # print(hla_residues)

    print("@ Pulling point: C-terminal residue of antigen peptide")
    antigen_cterm_res = pep_residues[pep_res_ids[-1]]
    print(antigen_cterm_res["CA"], antigen_cterm_res["CA"].get_coord())


def run_asmd():
    pass


def main():
    parser = asmd_args.parse_args()
    args = parser.parse_args()

    prep_dir = "1_prep"
    min_dir = "2_min"
    equil_dir = "3_equil"
    prep_md_sim_base_process(
        inpdb=args.in_rec_pdb,
        md_sys_name=args.md_sys_name,
        fftype=args.fftype,
        water_model=args.water_model,
        ewald_cut=args.ewald_cut,
        water_box_size=args.water_box_size,
        prep_dir=prep_dir,
        min_dir=min_dir,
        equil_dir=equil_dir,
    )

    # parse_hla_pep_system(inpdb=init_prep_pdb)

    run_asmd()


if __name__ == "__main__":
    print("""
    #############################################################
    ##   Pulling MD simulation of HLA-antigen peptide system   ##
    #############################################################
    """)
    main()
