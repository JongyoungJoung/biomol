import subprocess
import sys

# from pathlib import Path


def teleap_protein(
    *,
    md_sys_name: str = "md_system",
    inpdb: str = "in.pdb",
    fftype: str = "ff14SB",
    water_model: str = "tip3p",
    water_box_size: float = 15.0,
    ssbond: list[tuple[int, int]] = [],
    dont_run_tleap: bool = False,
):
    tleap_in = [f"source leaprc.protein.{fftype}"]
    tleap_in += [f"source leaprc.water.{water_model}"]
    tleap_in += [f"protein = loadpdb {inpdb}"]
    for ss_pair in ssbond:
        tleap_in += [f"bond protein.{ss_pair[0]}.SG protein.{ss_pair[1]}.SG"]
    tleap_in += [
        f"solvateBox protein {water_model.upper()}BOX {water_box_size:.1f} iso"
    ]
    tleap_in += ["addIonsRand protein Na+ 0 3.5"]
    tleap_in += ["addIonsRand protein Cl- 0 3.5"]
    tleap_in += [
        f"saveamberparm protein {md_sys_name}_solv.prmtop {md_sys_name}_solv.inpcrd"
    ]
    tleap_in += [f"savepdb protein {md_sys_name}_init_solv.pdb"]
    tleap_in += ["quit"]

    tleap_infile = open("tleap_protein.in", "w")
    tleap_infile.write("\n".join(tleap_in))
    tleap_infile.close()

    if not dont_run_tleap:
        subprocess.run(["tleap", "-f", "tleap_protein.in"])

    return (
        f"{md_sys_name}_init_solv.pdb",
        f"{md_sys_name}_solv.prmtop",
        f"{md_sys_name}_solv.inpcrd",
    )
