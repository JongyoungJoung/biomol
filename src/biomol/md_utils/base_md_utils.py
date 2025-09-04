#
# Base of MD utils
#
import os
import shutil
from pathlib import Path

from biomol.md_utils import md_calc_func, tleap_utils


def setup_md_system(
    *,
    inpdb: str = "in.pdb",
    md_sys_name: str = "md_system",
    fftype: str = "ff14SB",
    water_model: str = "tip3p",
    water_box_size: float = 15.0,
    prep_dir: str = "prep",
):
    pwd = Path.cwd()
    if Path(prep_dir).exists():
        shutil.rmtree(prep_dir)
    Path(prep_dir).mkdir()

    if not Path(inpdb).exists():
        raise FileNotFoundError(f"input pdb {inpdb} not found here.")
    else:
        shutil.copy(inpdb, prep_dir)

    os.chdir(prep_dir)
    # check SSBOND
    ssbond = []
    ssbond_cys_ids = []
    with open(inpdb) as input_pdb:
        for line in input_pdb:
            if line.startswith("SSBOND"):
                resi = int(line[17:21].strip())
                resj = int(line[31:35].strip())
                ssbond.append((resi, resj))
                ssbond_cys_ids.append(resi)
                ssbond_cys_ids.append(resj)
    # If SSBOND exists, change the residue name CYS -> CYX
    if len(ssbond_cys_ids) != 0:
        shutil.move(inpdb, f"{Path(inpdb).stem}_org.pdb")
        new_lines = []
        with open(f"{Path(inpdb).stem}_org.pdb", "r") as in_pdb:
            for line in in_pdb:
                if line.startswith("ATOM"):
                    resid = int(line[22:26].strip())
                    resnam = line[17:20]
                    if resnam == "CYS" and resid in ssbond_cys_ids:
                        new_lines.append(f"{line[:17]}CYX{line[20:]}")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
        new_inpdb = open(inpdb, "w")
        for nline in new_lines:
            new_inpdb.write(nline)
        new_inpdb.close()

    init_solv_pdb, init_solv_top, init_solv_crd = tleap_utils.teleap_protein(
        inpdb=inpdb,
        md_sys_name=md_sys_name,
        fftype=fftype,
        water_model=water_model,
        water_box_size=water_box_size,
        ssbond=ssbond,
    )

    os.chdir(pwd)

    return init_solv_pdb, init_solv_top, init_solv_crd


def setup_conventional_minimization_steps(
    *,
    init_str_crd: str,
    md_top: str,
    protein_residue: list[int],
    min_dir: str = "2_min",
    ewald_cut: float = 12.0,
    ligand_residue: list[int] = [],
    write_pdb_each_step: bool = False,
):
    """
    Conventional mimization steps.

    5 sequential minimizations of gradually decreasing restraints.

    Args:
        init_str_crd: initial starting coordinates of 1st minimization.
        md_top: MD topology file
        min_dir: minimization directory
        ewald_cut: distance cuf for Ewald summation
    """
    if len(protein_residue) != 0:
        assert len(protein_residue) == 2, (
            "About protein residue for masking, first residue id and last residue id are required."
        )

    pwd = Path.cwd()
    if Path(min_dir).exists():
        shutil.rmtree(min_dir)
    Path(min_dir).mkdir()

    if not Path(init_str_crd).exists():
        raise FileNotFoundError(
            f"Starting structure for minimization {init_str_crd} not found."
        )
    if not Path(md_top).exists():
        raise FileNotFoundError(f"Topology of simulation system {md_top} not found.")

    os.chdir(min_dir)
    # make symbolic links in this directory
    if Path(init_str_crd).exists():
        Path(Path(init_str_crd).name).symlink_to(init_str_crd)
    else:
        raise FileNotFoundError(
            f"In current dir {Path.cwd()}, {init_str_crd} not found."
        )
    if Path(md_top).exists():
        Path(Path(md_top).name).symlink_to(md_top)
    else:
        raise FileNotFoundError(f"In current dir {Path.cwd()}, {md_top} not found.")

    min_cmds = []
    # min-1. minimizing hydrogen
    min1_out = f"{Path(init_str_crd).stem}_1_min"
    min1_cmd, min1_rst = md_calc_func.minimization(
        start_crd=init_str_crd,
        top=md_top,
        min_input="1_min.in",
        outname=min1_out,
        min_max_cyc=3000,
        min_ncyc=1500,
        with_restraint=True,
        restraint_fc=10.0,
        restraint_mask="&!@H=",
        ewald_cut=ewald_cut,
        step_description="Minimization step-1. Only Hydrogen",
    )
    min_cmds.append(min1_cmd)
    if write_pdb_each_step:
        min_cmds.append(f"ambpdb -p {md_top} -c {min1_rst} > {min1_out}.pdb\n")

    # min-2. minimizing hydrogens and waters
    min2_out = f"{Path(init_str_crd).stem}_2_min"
    min2_mask = f":{protein_residue[0]}-{protein_residue[1]}"
    # If ligands exist, include them.
    if len(ligand_residue) != 0:
        for l_resid in ligand_residue:
            min2_mask += f",{l_resid}"
    min2_mask += "&!@H="
    min2_cmd, min2_rst = md_calc_func.minimization(
        start_crd=min1_rst,
        top=md_top,
        min_input="2_min.in",
        outname=min2_out,
        min_max_cyc=3000,
        min_ncyc=1500,
        with_restraint=True,
        restraint_fc=5.0,
        restraint_mask=min2_mask,
        ewald_cut=ewald_cut,
        step_description="Miminization step-2. Only waters",
    )
    min_cmds.append(min2_cmd)
    if write_pdb_each_step:
        min_cmds.append(f"ambpdb -p {md_top} -c {min2_rst} > {min2_out}.pdb")

    # min-3. minimizing side-chain part of protein, not ligand (if ligands exist)
    min3_out = f"{Path(init_str_crd).stem}_3_min"
    min3_mask = f":{protein_residue[0]}-{protein_residue[1]}@C,CA,N"
    if len(ligand_residue) != 0:
        min3_mask += ",:"
        for l_resid in ligand_residue:
            min3_mask += f",{l_resid}"
    min3_cmd, min3_rst = md_calc_func.minimization(
        start_crd=min2_rst,
        top=md_top,
        min_input="3_min.in",
        outname=min3_out,
        min_max_cyc=3000,
        min_ncyc=1500,
        with_restraint=True,
        restraint_fc=5.0,
        restraint_mask=min3_mask,
        ewald_cut=ewald_cut,
        step_description="Mimization step-3. Side chains and all waters (not ligands (if existed))",
    )
    min_cmds.append(min3_cmd)

    # min-4. minimizing side-chain and ligands (if existed) with lowering restraint force
    min4_out = f"{Path(init_str_crd).stem}_4_min"
    min4_mask = f":{protein_residue[0]}-{protein_residue[1]}@C,CA,N"
    min4_cmd, min4_rst = md_calc_func.minimization(
        start_crd=min3_rst,
        top=md_top,
        min_input="4_min.in",
        outname=min4_out,
        min_max_cyc=3000,
        min_ncyc=1500,
        with_restraint=True,
        restraint_fc=1.0,
        restraint_mask=min4_mask,
        ewald_cut=ewald_cut,
        step_description="Minimization step-4. All but not backbone of the proteins",
    )
    min_cmds.append(min4_cmd)

    # min-5. final minimization without restraints
    min5_out = f"{Path(init_str_crd).stem}_5_min"
    min5_cmd, min5_rst = md_calc_func.minimization(
        start_crd=min4_rst,
        top=md_top,
        outname=min5_out,
        min_input="5_min.in",
        min_max_cyc=5000,
        min_ncyc=2500,
        with_restraint=False,
        ewald_cut=ewald_cut,
        step_description="Miminization step-5. Without restraints",
    )
    min_cmds.append(min5_cmd)

    # run script
    with open("run_min.sh", "w") as run_min_sh:
        run_min_sh.write("#!/bin/sh\n")
        run_min_sh.write("GPUID=$1\n")
        run_min_sh.write('export CUDA_VISIBLE_DEVICES="${GPUID}"\n\n')
        run_min_sh.write("\n".join(min_cmds))

    os.chdir(pwd)

    return min5_rst


def setup_conventional_equilibration_steps(
    *,
    system_name: str,
    init_str_crd: str,
    md_top: str,
    protein_residue: list[int],
    equil_dir: str = "3_equilibration",
    ewald_cut: float = 12.0,
    target_temp: float = 300.0,
    raising_temp_interval: float = 50.0,
    ligand_residue: list[int] = [],
    write_pdb_each_step: bool = False,
):
    """
    Setup conventional equilibration steps.

    gradually heating steps and following density, pressure equilibration steps.
    """
    pwd = Path.cwd()
    if Path(equil_dir).exists():
        shutil.rmtree(equil_dir)
    Path(equil_dir).mkdir()

    os.chdir(equil_dir)
    Path(Path(init_str_crd).name).symlink_to(init_str_crd)
    Path(Path(md_top).name).symlink_to(md_top)

    equil_cmds = []
    # NOTE: Heating process
    heating_steps = [
        i * raising_temp_interval
        for i in range(int(target_temp / raising_temp_interval) + 1)
    ]
    heating_steps.append(target_temp)
    heat_mask = f"{protein_residue[0]}-{protein_residue[1]}"
    if len(ligand_residue) != 0:
        for l_id in ligand_residue:
            heat_mask += f",{l_id}"
    heat_mask += "&!@H="
    eq_step_id: int = 0
    prev_start_crd = ""
    for t_i in range(1, len(heating_steps)):
        if t_i == 1:
            restart = False
            start_crd = init_str_crd
        else:
            restart = True
            start_crd = prev_start_crd
        temp_start = heating_steps[t_i - 1]  # starting temperature
        temp_end = heating_steps[t_i]  # target temperature
        heat_cmd, heat_rst = md_calc_func.dynamics(
            start_crd=start_crd,
            top=md_top,
            outname=f"{system_name}_heat_{t_i:0d}",
            md_input=f"heat_{t_i:02d}.in",
            restart=restart,
            nstlim=50000,
            dt=0.001,
            ntwr=50000,
            ntpr=1000,
            ntwx=10000,
            heating=True,
            start_temp=temp_start,
            target_temp=temp_end,
            ewald_cut=ewald_cut,
            with_restraint=True,
            restraint_fc=10.0,
            restraint_mask=heat_mask,
            step_description=f"heating system from {temp_start:.1f} K to {temp_end:.1f} K",
        )
        equil_cmds.append(heat_cmd)
        prev_start_crd = heat_rst
        eq_step_id = t_i
    # NOTE: Equilibration process
    #
    # equilibration - 1.
    eq_step_id += 1
    eq1_cmd, eq1_rst = md_calc_func.dynamics(
        start_crd=prev_start_crd,
        top=md_top,
        outname=f"{system_name}_equil_{eq_step_id:02d}",
        restart=True,
        md_input=f"equil_{eq_step_id:02d}.in",
        nstlim=1000000,
        dt=0.001,
        ntpr=10000,
        ntwx=100000,
        target_temp=target_temp,
        ewald_cut=ewald_cut,
        with_restraint=True,
        restraint_fc=5.0,  # decreased force constant: 10.0 -> 5.0
        restraint_mask=heat_mask,
        ensemble="NVT",
        step_description=f"NVT Equilibration step-{eq_step_id} water and hydrogen atoms",
    )
    equil_cmds.append(eq1_cmd)
    # equilibration - 2.
    eq_step_id += 1
    eq2_cmd, eq2_rst = md_calc_func.dynamics(
        start_crd=eq1_rst,
        top=md_top,
        outname=f"{system_name}_equil_{eq_step_id:02d}",
        restart=True,
        md_input=f"equil_{eq_step_id:02d}.in",
        nstlim=1000000,
        dt=0.001,
        ntpr=10000,
        ntwx=100000,
        target_temp=target_temp,
        ewald_cut=ewald_cut,
        with_restraint=True,
        restraint_fc=2.5,  # decreased force constant: 5.0 -> 2.5
        restraint_mask=heat_mask,
        ensemble="NVT",
        step_description=f"NVT Equilibration step-{eq_step_id} water and hydrogen atoms",
    )
    equil_cmds.append(eq2_cmd)
    # equilibration - 3.
    eq_step_id += 1
    backbone_mask = f"{protein_residue[0]}-{protein_residue[1]}&@C,CA,N"
    eq3_cmd, eq3_rst = md_calc_func.dynamics(
        start_crd=eq2_rst,
        top=md_top,
        outname=f"{system_name}_equil_{eq_step_id:02d}",
        restart=True,
        md_input=f"equil_{eq_step_id:02d}.in",
        nstlim=1000000,
        dt=0.001,
        ntpr=10000,
        ntwx=100000,
        target_temp=target_temp,
        ewald_cut=ewald_cut,
        with_restraint=True,
        restraint_fc=1.0,
        restraint_mask=backbone_mask,
        ensemble="NVT",
        step_description=f"NVT Equilibration step-{eq_step_id} protein side chains,"
        " ligand, water and hydrogen atoms",
    )
    equil_cmds.append(eq3_cmd)
    # equilibration - 4.
    eq_step_id += 1
    eq4_cmd, eq4_rst = md_calc_func.dynamics(
        start_crd=eq3_rst,
        top=md_top,
        outname=f"{system_name}_equil_{eq_step_id:02d}",
        restart=True,
        md_input=f"equil_{eq_step_id:02d}.in",
        nstlim=1000000,
        dt=0.001,
        ntpr=10000,
        ntwx=100000,
        target_temp=target_temp,
        ewald_cut=ewald_cut,
        with_restraint=True,
        restraint_fc=0.5,
        restraint_mask=backbone_mask,
        ensemble="NPT",
        step_description=f"NPT Equilibration step-{eq_step_id} protein side chains, "
        " ligand, water and hydrogen atoms",
    )
    equil_cmds.append(eq4_cmd)
    # equilibration - 5.
    eq_step_id += 1
    eq5_cmd, eq5_rst = md_calc_func.dynamics(
        start_crd=eq4_rst,
        top=md_top,
        outname=f"{system_name}_equil_{eq_step_id:02d}",
        restart=True,
        md_input=f"equil_{eq_step_id:02d}.in",
        nstlim=5000000,
        dt=0.001,
        ntpr=10000,
        ntwx=100000,
        target_temp=target_temp,
        ewald_cut=ewald_cut,
        with_restraint=False,
        ensemble="NPT",
        step_description="NPT Equilibration step-{eq_step_id} without restraints",
    )

    equil_cmds.append(eq5_cmd)
    final_rst = eq5_rst
    # run script
    with open("run_equil.sh", "w") as run_equil_sh:
        run_equil_sh.write("#!/bin/sh\n")
        run_equil_sh.write("GPUID=$1\n")
        run_equil_sh.write('export CUDA_VISIBLE_DEVICES="${GPUID}"\n\n')
        run_equil_sh.write("\n".join(equil_cmds))

    return final_rst
