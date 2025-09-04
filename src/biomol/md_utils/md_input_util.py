# MD input utilities


def write_min_input(
    *,
    input_name: str = "min.in",
    step_description: str = "minimization",
    max_cyc: int = 3000,
    ncyc: int = 1500,
    ewald_cut: float = 12.0,
    with_restraint: bool = False,
    restraint_fc: float = 10.0,
    restraint_mask: str = "!@H=",
):
    """
    Write input option file for minimization.
    """
    input_cont = [f"{step_description}"]
    input_cont.append("&cntrl")
    input_cont.append(" imin = 1")  # minimization option
    input_cont.append(f" maxcyc = {max_cyc}")
    input_cont.append(f" ncyc = {ncyc}")
    input_cont.append(" ntb = 1")
    input_cont.append(f" cut = {ewald_cut:.1f}")
    if with_restraint:
        input_cont.append(" ntr = 1")
        input_cont.append(f" restraint_wt = {restraint_fc:.1f}")
        input_cont.append(f' restraintmask = "{restraint_mask}"')
    else:
        input_cont.append(" ntr = 0")
    input_cont.append("/\nEND")

    with open(input_name, "w") as min_in:
        min_in.write("\n".join(input_cont))


def write_md_input(
    *,
    input_name: str = "md.in",
    step_description: str = "molecular dynamics",
    dt: float = 0.002,
    nstlim: int = 500000,
    ntpr: int = 1000,
    ntwx: int = 1000,
    ntwr: int = -1,  # only write restart at the last snapshot
    restart: bool = False,
    raise_temperature: bool = False,  # for heating
    start_temp: float = 0.0,  # target temperature
    target_temp: float = 300.0,  # target temperature
    ewald_cut: float = 12.0,
    seed_number: int = -1,
    with_restraint: bool = False,
    restraint_fc: float = 10.0,
    restraint_mask: str = "!@H=",
    ensemble: str = "NVT",
):
    """
    Write input option file for conventional molecular dynamics simulation.
    """
    input_cont = [f"({ensemble.upper()}): {step_description}"]
    input_cont.append("&cntrl")
    input_cont.append(" imin = 0")  # dynamics option
    if restart:
        # restart simulation from previous steps
        input_cont.append(" irest = 1")
        input_cont.append(" ntx = 5")
    else:
        input_cont.append(" irest = 0")
        input_cont.append(" ntx = 1")
    if raise_temperature:
        input_cont.append(f" tempi = {start_temp:.1f}")
    input_cont.append(f" temp0 = {target_temp:.1f}")
    # Ewald summation distance cut
    input_cont.append(f" cut = {ewald_cut}")
    # for SHAKE algorithm
    input_cont.append(" ntc = 2, ntf = 2")
    # Langevin dynamics to control temperature
    input_cont.append(" ntt = 3, gamma_ln = 1.0")
    # ensemble conditions
    # TODO: add another  ensemble conditions
    if ensemble.upper() == "NVT":
        input_cont.append(" ntb = 1")
    elif ensemble.upper() == "NPT":
        input_cont.append(" ntb = 2, pres0 = 1.0, ntp = 1, taup = 2.0")
    # initiation of starting force
    input_cont.append(f" ig = {seed_number}")  # random number
    input_cont.append(" iwrap = 1")
    # Writing frequencies
    input_cont.append(f" dt = {dt:.3f}")
    input_cont.append(f" nstlim = {nstlim}")
    input_cont.append(f" ntpr = {ntpr}")
    input_cont.append(f" ntwx = {ntwx}")
    if ntwr == -1:
        input_cont.append(f" ntwr = {nstlim}")
    else:
        input_cont.append(f" ntwr = {ntwr}")
    # positional restraint
    if with_restraint:
        input_cont.append(" ntr = 1")
        input_cont.append(f" restraint_wt = {restraint_fc:.1f}")
        input_cont.append(f" restraintmask = {restraint_mask}")
    else:
        input_cont.append(" ntr = 0")
    input_cont.append("/\nEND")

    with open(input_name, "w") as md_in:
        md_in.write("\n".join(input_cont))
