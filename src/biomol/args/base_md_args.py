import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="AMBER MD simulation Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # NOTE:
    # I/O related
    io_opt = parser.add_argument_group("IO Options")
    io_opt.add_argument(
        "-inrec",
        "--input-receptor",
        dest="in_rec_pdb",
        type=str,
        required=True,
        help="Input receptor pdb file (protein alone or protein-ligand complex.",
    )
    io_opt.add_argument(
        "-inlig",
        "--input-ligand",
        dest="in_lig_pdb",
        type=str,
        default="",
        help="Input ligand pdb file. (optional)",
    )
    io_opt.add_argument(
        "-n",
        "--num-traj",
        dest="num_traj",
        type=int,
        default=1,
        help="Number of trajectories.",
    )
    io_opt.add_argument(
        "-t",
        "--md-system-name",
        dest="md_sys_name",
        type=str,
        default="md_system",
        help="COMMON Name of MD system used to assign composite files in this MD system.",
    )
    io_opt.add_argument(
        "-mol2",
        "--mol2-files",
        dest="mol2_files",
        type=list[str],
        nargs="*",
        help="Existing (pre-built) mol2 files for bound ligands",
    )

    # NOTE:
    # MD Condition related
    md_condition_opt = parser.add_argument_group("MD Setup-related Options")
    md_condition_opt.add_argument(
        "-fftype",
        "--force-field-type",
        dest="fftype",
        type=str,
        default="ff14SB",
        choices=[
            "ff99SB",
            "ff99SBildn",
            "ff99SBnmr",
            "ff03.r1",
            "ff03ua",
            "ff14SB",
            "ff14SB_modAA",
            "ff14SBonlysc",
            "sb15",
            "ff15ipq",
            "ff15ipq-vac",
            "ff19SB",
            "ff19SB_modAAff19ipq",
        ],
        help="Force field type",
    )
    md_condition_opt.add_argument(
        "-lig-at",
        "--ligand-atom-type",
        dest="ligand_at",
        type=str,
        choices=["gaff", "gaff2", "amber", "mixed"],
        default="gaff2",
        help="Atom type of ligands",
    )
    md_condition_opt.add_argument(
        "-phos",
        "--use-phospho-ff",
        dest="use_phosphoAA",
        action="store_true",
        help="Use force field for phosphorylated amino acid",
    )
    md_condition_opt.add_argument(
        "-water-model",
        dest="water_model",
        type=str,
        choices=[
            "fb3",
            "fb3mod",
            "fb4",
            "opc3",
            "opc3pol",
            "spce",
            "spceb",
            "tip3p",
            "tip4pd",
            "tip4pd-a99SBdisp",
            "tip4pew",
        ],
        default="tip3p",
        help="Water model type",
    )
    md_condition_opt.add_argument(
        "-temp",
        "--temperature",
        dest="temp",
        type=float,
        default=300.0,
        help="Simulation temperature (Kelvin)",
    )
    md_condition_opt.add_argument(
        "-ec",
        "--ewald-cut-dist",
        dest="ewald_cut",
        type=float,
        default=12.0,
        help="Distance cut-off for Ewald Summation (Angstrom)",
    )

    # NOTE:
    # MD control option for time, restraint, output logs
    md_control_opt = parser.add_argument_group(
        "MD Control Options for Running Time or Constraints"
    )
    md_control_opt.add_argument(
        "-dt",
        "--sim-step-time",
        dest="dt",
        type=float,
        default=1.0,
        help="Unit time step for simulation run (fs)",
    )
    md_control_opt.add_argument(
        "-heat-time",
        "--heat-time-per-step",
        dest="heat_time",
        type=float,
        default=0.05,
        help="Simulation time to heat the system in one heating step "
        "for gradually heating (ns)",
    )
    md_control_opt.add_argument(
        "-equil-time",
        "--equil-time-per-step",
        dest="equil_time",
        type=float,
        default=1.0,
        help="Equilibration time in one step for gradually equilibrating the system (ns)",
    )
    md_control_opt.add_argument(
        "-equil-last-time",
        "--equil-time-for-last-step",
        dest="equil_last_time",
        type=float,
        default=5.0,
        help="Equilibration time in the last step, non-restrainted equilibration (ns)",
    )
    md_control_opt.add_argument(
        "-prod-one-time",
        "--production-one-step-time",
        dest="prod_one_time",
        type=float,
        default=1.0,
        help="One production step' running time (ns)",
    )
    md_control_opt.add_argument(
        "-prod-repeat-num",
        "--production-repeating-num",
        dest="prod_repeat_num",
        type=int,
        default=100,
        help="Repeating number of one production run step (num_repeat)",
    )
    md_control_opt.add_argument(
        "-wrt-eq-out-freq",
        "--write-eq-out-frequency",
        dest="wrt_eq_out_freq",
        type=float,
        default=0.01,
        help="Frequency time to write logs in .out during equilibration (ns)",
    )
    md_control_opt.add_argument(
        "-wrt-eq-crd-freq",
        "--write-eq-crd-frequency",
        dest="wrt_eq_crd_freq",
        type=float,
        default=0.01,
        help="Frequency time to write structures into crd files (ns)",
    )
    md_control_opt.add_argument(
        "-wrt-eq-rst-freq-ratio",
        "--write-eq-rst-frequency-ratio",
        dest="wrt_eq_rst_freq_ratio",
        type=float,
        default=1.0,
        help="Frequency ratio to write restart file during equilibration (ratio: 0.0 ~ 1.0)",
    )
    md_control_opt.add_argument(
        "-wrt-pd-out-freq",
        "--write-prod-out-frequency",
        dest="wrt_pd_out_freq",
        type=float,
        default=0.01,
        help="Frequency time to write logs in .out during production (ns)",
    )
    md_control_opt.add_argument(
        "-wrt-pd-crd-freq",
        "--write-prod-crd-frequency",
        dest="wrt_pd_crd_freq",
        type=float,
        default=0.01,
        help="Frequency time to write structures into crd files during production (ns)",
    )
    md_control_opt.add_argument(
        "-wrt-pd-rst-freq-ratio",
        "--write-prod-rst-frequency-ratio",
        dest="wrt_pd_rst_freq_ratio",
        type=float,
        default=1.0,
        help="Frequency ratio to write restart file during production (ratio: 0.0 ~ 1.0)",
    )

    # NOTE:
    # Ligand preparation related
    lig_prep_opt = parser.add_argument_group("Ligand Preparation Options")
    lig_prep_opt.add_argument(
        "-ct",
        "--charge-type-ligand",
        dest="charge_type",
        type=str,
        choices=["am1bcc", "abcg2"],
        help="Method to determine ligands' atomic charges",
    )

    # NOTE:
    # Target system composition related
    target_system_opt = parser.add_argument_group("Target System Related Options")
    target_system_opt.add_argument(
        "-wb",
        "--water-box-size",
        dest="water_box_size",
        type=float,
        default=15.0,
        help="Water box size. Distance from surface of system to box boundary (Angstrom)",
    )

    return parser
