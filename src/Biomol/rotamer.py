import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

rotamer_lib = f'{os.environ["HOME"]}/rotamerBB/Everything-5_rotamerBB/Extended2-5/ALL.bbdep.rotamers.lib'


@dataclass
class RotamerDataPoint:
    """
    dataclass for one line of rotamer library
    """

    count: int
    r1: int
    r2: int
    r3: int
    r4: int
    probabil: float
    chi1Val: float
    chi2Val: float
    chi3Val: float
    chi4Val: float
    chi1Sig: float
    chi2Sig: float
    chi3Sig: float
    chi4Sig: float


@dataclass
class SideChainRotamer:
    phi_psi: list[float]
    chi_data: list[RotamerDataPoint]


@dataclass
class RotamerLibrary:
    residue_name: str
    chi_vals: list[SideChainRotamer]


def init_rotamer():
    assert Path(rotamer_lib).exists(), f"FileNotFoundError: {rotamer_lib} not exist"

    with open(rotamer_lib) as rotamer_file:
        for line in rotamer_file:
            if not line.startswith("#"):
                (
                    resname,
                    phi,
                    psi,
                    count,
                    r1,
                    r2,
                    r3,
                    r4,
                    prob,
                    chi1val,
                    chi2val,
                    chi3val,
                    chi4val,
                    chi1sig,
                    chi2sig,
                    chi3sig,
                    chi4sig,
                ) = line.split()
