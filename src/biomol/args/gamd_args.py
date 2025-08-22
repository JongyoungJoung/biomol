from biomol.args import base_md_args


def parse_args():
    parser = base_md_args.parse_args()

    gamd_setup_opt = parser.add_argument_group("GAMD Setup Options")
    gamd_setup_opt.add_argument(
        "-gamd",
        "--run-gamd",
        dest="gamd",
        action="store_ture",
        help="Run simulations as Gaussian Accelerated MD",
    )
    gamd_option_desc = {
        0: "no boost",
        1: "total potential energy only",
        2: "dihedral energy only",
        3: "dual on dihedral and total potential energy",
        4: "non-bonded potential energy only",
        5: "dual on dihedral and non-bonded potential energy",
        10: "non-bonded potential of selected region"
        " (timask1 and scmask1) as for a ligand (LiGaMD)",
        11: "dual on both non-bonded potential of bound ligand and "
        "remaining potential energy of the rest of system, (LiGaMD_Dual",
        14: "total potential energy of selected region as for peptide (Pep-GaMD)",
        15: "dual on both peptide potential energy and "
        "total system potential energy other than peptide (Pep-GaMD_Dual)",
        16: "interaion between protein partners, "
        "1st protein - timask1, scmask1, "
        "2nd protein - bgpro2atm (first atom number of protein), edpro2atm (end atom number of protein) "
        "for protein-protein interaction GaMD (PPI-GaMD)",
        17: "dual on both protein-protein interactions and "
        "remaining potential energy of the entire system (PPI-GaMD_Dual)",
        18: "dual on both the essential peptide potential (only peptide dihedral energy) "
        "and total system potential energy other than peptide potential (Variant of Pep-GaMD_Dual)",
    }
    gamd_setup_opt.add_argument(
        "-igamd",
        "--gamd-type",
        dest="igamd",
        type=int,
        choices=gamd_option_desc.keys(),
        default=11,
        help="Boost potential (igamd option in AMBER) for GAMD. Options:\n"
        + "\n".join([f"   {k:>3d}: {v}" for k, v in gamd_option_desc.items()]),
    )
    gamd_setup_opt.add_argument(
        "-timask1-gamd",
        dest="gamd_ti_maks1",
        type=str,
        default="",
        help="Atom or Residue mask of the bound ligand.",
    )
    gamd_setup_opt.add_argument(
        "-scmask1-gamd",
        dest="gamd_sc_mask1",
        type=str,
        default="",
        help="Atom or Residue mask of the bound ligand described using soft core",
    )
    gamd_setup_opt.add_argument(
        "-ibblig",
        dest="ibblig",
        type=int,
        choices=[0, 1],
        default=0,
        help="Boost the bound ligand selectively with numLig (> 1) "
        "option => 0: no selective boost, "
        "1: boost the bound ligand selectively out of numLig ligand molecules in the system.",
    )
    gamd_setup_opt.add_argument(
        "-atom-p",
        dest="atom_p",
        type=int,
        default=0,
        help="Serial number of a protein atom (starting from 1 for a protein's the first atom) "
        "used to calculate the distance to the ligand. "
        "Only used when ibblig = 1",
    )
    gamd_setup_opt.add_argument(
        "-atom-l",
        dest="atom_l",
        type=int,
        default=0,
        help="Serial number of a ligand atom (starting from 1 for a ligand's the first atom) "
        "used to calculate the distance to the protein. "
        "Only used when ibblig = 1",
    )
    gamd_setup_opt.add_argument(
        "-dblig",
        dest="dblig",
        type=float,
        default=4.0,
        help="Cutoff distance between atom_p and atom_l for determining the boundness "
        "of the ligand to the ptoein. Only used when ibblig = 1",
    )
    gamd_setup_opt.add_argument(
        "-bgpro2atm",
        dest="bgpro2atm",
        type=int,
        default=-1,
        help="Starting atomic number of the second protein, -1: None",
    )
    gamd_setup_opt.add_argument(
        "-edpro2atm",
        dest="edpro2atm",
        type=int,
        default=-1,
        help="Ending atomic number of the second protein, -1: None",
    )

    gamd_control_opt = parser.add_argument_group("GAMD Control Options")
    gamd_control_opt.add_argument(
        "-sigma0p",
        dest="simga0p",
        type=float,
        default=6.0,
        help="Upper limit of sigma of first potential boost. (kcal/mol)",
    )
    gamd_control_opt.add_argument(
        "-sigma0d",
        dest="sigma0d",
        type=float,
        default=6.0,
        help="Upper limit of sigma of second potential boost. (kcal/mol)",
    )
    gamd_control_opt.add_argument(
        "-iEP",
        dest="iEP",
        type=int,
        default=1,
        help="Threshold energy E for applying first potential in dual-boost (kcal/mol)",
    )
    gamd_control_opt.add_argument(
        "-iED",
        dest="iED",
        type=int,
        default=1,
        help="Threshold energy E for applying second potential in dual-boost (kcal/mol)",
    )
    gamd_control_opt.add_argument(
        "-ntcmd",
        dest="ntcmd",
        type=int,
        default=1000000,
        help="Number of initial cMD steps [ntmcd] (steps) -> time: ntcmd * 2fs",
    )
    gamd_control_opt.add_argument(
        "-ntcmdprep",
        dest="ntcmdprep",
        type=int,
        default=200000,
        help="Number of preparation of cMD steps [ntcmdprep] (steps) -> time: ntcmdprep * 2fs",
    )
    gamd_control_opt.add_argument(
        "-nteb",
        dest="nteb",
        type=int,
        default=1000000,
        help="Number of biasing MD steps [nteb] (steps) -> time: nteb * 2fs",
    )
    gamd_control_opt.add_argument(
        "-ntebprep",
        dest="ntebprep",
        type=int,
        default=200000,
        help="Number of preparation of biasing MD steps [ntebprep] (steps) -> time: ntebprep * 2fs",
    )
    gamd_control_opt.add_argument(
        "-ntave",
        dest="ntave",
        type=int,
        default=50000,
        help="Number of simulating steps to update mean and sigma (std)"
        "of potential energies [ntave] (steps) -> time: ntave * 2fs",
    )

    return parser
