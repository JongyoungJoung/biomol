#
# MD calculation functions utilities
#
from pathlib import Path

from biomol.md_utils import md_input_util


def minimization(
    *,
    start_crd: str,
    top: str,
    min_input: str = "min.in",
    outname: str = "min_out",
    min_max_cyc: int = 3000,
    min_ncyc: int = 1500,
    ewald_cut: float = 12.0,
    with_restraint: bool = False,
    restraint_fc: float = 10.0,
    restraint_mask: str = "!@H=",
    write_crd: bool = False,
    step_description: str = "minimizing only H",
):
    """
    Minimization step.

    Args:
        start_crd: starting coordinates (crd, rst)
        top: topology files
        min_input: pmemd input for minimization
        outname: basename of minimization output
        min_max_cyc: total number of minimization step
        min_ncyc: num of minimization step for steepest descent.
        ewald_cut: distance cut of Ewald summation for electrostatic
        with_restraint: use restraint
        restraint_fc: force constant for restraint
        restraint_mask: mask where restraint force is applied
        write_crd: writing minmization steps into coordinate files
        step_description: description of minimization steps

        write_crd:

    Returns:
        MD run command
        finally stored restart (rst) file.
    """
    # min_input
    md_input_util.write_min_input(
        input_name=min_input,
        max_cyc=min_max_cyc,
        ncyc=min_ncyc,
        ewald_cut=ewald_cut,
        with_restraint=with_restraint,
        restraint_fc=restraint_fc,
        restraint_mask=restraint_mask,
        step_description=step_description,
    )

    # command
    cmd = ["pmemd.cuda", "-O"]
    cmd.extend(["-i", min_input])
    cmd.extend(["-o", f"{Path(min_input).stem}.out"])
    cmd.extend(["-p", top])
    cmd.extend(["-c", start_crd])
    if with_restraint:
        cmd.extend(["-ref", start_crd])
    if write_crd:
        cmd.extend(["-x", f"{outname}.nc"])
    cmd.extend(["-r", f"{outname}.rst"])

    return " ".join(cmd), f"{outname}.rst"


def dynamics(
    *,
    start_crd: str,
    top: str,
    outname: str,
    md_input: str = "md.in",
    restart: bool = False,
    dt: float = 0.002,
    nstlim: int = 500000,
    ntpr: int = 1000,
    ntwx: int = 1000,
    ntwr: int = -1,
    heating: bool = False,
    start_temp: float = 0.0,
    target_temp: float = 300.0,
    ewald_cut: float = 12.0,
    with_restraint: bool = False,
    restraint_fc: float = 10.0,
    restraint_mask: str = "!@H=",
    step_description: str = "molecular dynamics simulation only H",
    ensemble: str = "NVT",
):
    """
    Conventional dynamics step.

    Args:
        start_crd
        top
        outname

    Returns:
        MD run command
        finally stored restart (rst) file.
    """
    ensemble_set = ["NVE", "NVT", "NPT"]
    assert ensemble.upper() in ensemble_set, (
        f"{ensemble.upper()} is undefined ensemble condition."
    )
    md_input_util.write_md_input(
        input_name=md_input,
        restart=restart,
        step_description=step_description,
        dt=dt,
        nstlim=nstlim,
        ntpr=ntpr,
        ntwx=ntwx,
        ntwr=ntwr,
        raise_temperature=heating,
        start_temp=start_temp,
        target_temp=target_temp,
        ewald_cut=ewald_cut,
        with_restraint=with_restraint,
        restraint_fc=restraint_fc,
        restraint_mask=restraint_mask,
        ensemble=ensemble.upper(),
    )

    cmd = ["pmemd.cuda", "-O"]
    cmd.extend(["-i", md_input])
    cmd.extend(["-o", f"{Path(md_input).stem}.out"])
    cmd.extend(["-p", top])
    cmd.extend(["-c", start_crd])
    cmd.extend(["-x", f"{outname}.nc"])
    cmd.extend(["-r", f"{outname}.rst"])
    if with_restraint:
        cmd.extend(["-ref", start_crd])

    return " ".join(cmd), f"{outname}.rst"
