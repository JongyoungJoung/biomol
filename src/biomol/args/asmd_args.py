from biomol.args import base_md_args


def parse_args():
    parser = base_md_args.parse_args()

    # NOTE:
    # ASMD setup
    asmd_setup_opt = parser.add_argument_group("ASMD Setup Option")
    asmd_setup_opt.add_argument(
        "-asmd",
        "--run-asmd",
        dest="asmd",
        action="store_true",
        help="Run ASMD simulation to pull the moleculess",
    )
    asmd_setup_opt.add_argument(
        "-nr-asmd",
        "--nreplica-asmd",
        dest="nreplica_asmd",
        type=int,
        default=40,
        help="Num of replica of steered MD",
    )
    asmd_setup_opt.add_argument(
        "-ns-asmd",
        "--nstage-asmd",
        dest="nstage_asmd",
        type=int,
        default=10,
        help="Num of stages of adaptive steered MD = Num of divided sectors of total pulling distance",
    )
    asmd_setup_opt.add_argument(
        "-pull-dist",
        "--total-pulling-distance",
        dest="total_pull_distance",
        type=float,
        default=-1.0,
        help="Pulling distance (total) (Angtrom), -1: automatically adjust",
    )

    # NOTE:
    # Controling option for ASMD
    asmd_control_opt = parser.add_argument_group("ASMD Control Option")
    asmd_pull_speed = {
        "ultra_fast": 100.0,
        "highly_fast": 25.0,
        "fast": 10.0,
        "medium": 5.0,
        "slow": 1.0,
        "very_slow": 0.5,
        "custom": -1.0,
    }
    asmd_control_opt.add_argument(
        "-pv",
        "--pulling-velocity",
        dest="pull_velocity",
        type=str,
        default="slow",
        help="Pulling spped (or velocity) (Angstrom/ns): \n"
        + "\n".join(
            f"   {k}: {v:>6.1f} Ansgrom/nm" for k, v in asmd_pull_speed.items()
        ),
    )
    asmd_control_opt.add_argument(
        "-pfc",
        "--pulling-force-constant",
        dest="pull_force_constant",
        type=float,
        default=50.0,
        help="Force constant to pull the target points (kcal/mol/A^2)",
    )
    asmd_control_opt.add_argument(
        "-pull-one-time",
        "--pulling-sim-time-one-stage",
        dest="pull_one_time",
        type=float,
        default=-1.0,
        help="Simulation time for one pulling stage. (ns) "
        "-1: adjusted according to pulling velocity and pull distance in one stage",
    )

    return parser
