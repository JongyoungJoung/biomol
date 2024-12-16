from biomol import protein
from biomol.libs import pdbutil


class Multimer(protein.Protein):
    def __init__(self, pdbfile_path: str):
        super().__init__(pdbfile_path)
        self.interface_pairs = {}

        assert self.get_num_chain() >= 2, "This Protein has a single chain"

    # for PPI
    def calc_interface_residues(
        self,
        ch1: str | None = None,
        ch2: str | None = None,
        cut: float = 5.0,
    ) -> None:
        if ch1 is None:
            chain_list1 = self.get_chain_ids()
        else:
            chain_list1 = [ch1]
        if ch2 is None:
            chain_list2 = self.get_chain_ids()
        else:
            chain_list2 = [ch2]

        for c1 in chain_list1:
            chain1_crd = self.get_atom_crds_and_their_resids(chainid=c1)
            for c2 in chain_list2:
                if c1 == c2:
                    continue
                if (c2, c1) in self.interface_pairs:
                    continue
                chain2_crd = self.get_atom_crds_and_their_resids(chainid=c2)

                pairs, npairs = pdbutil.get_interface_residues(
                    chain1_crd,
                    chain2_crd,
                    cut,
                    # len(chain1_crd), len(chain2_crd)
                )
                if (c1, c2) not in self.interface_pairs:
                    self.interface_pairs[(c1, c2)] = []
                self.interface_pairs[(c1, c2)] = pairs[:npairs]

    def check_residue_in_interface(
        self, resid: int, *, reschid: str | None = None, verbose=False
    ) -> tuple[bool, str]:
        # resid   : residue that you want to check its residence in the interface
        # reschid : residue's chain id if you know
        # verbose : show detail
        if len(self.interface_pairs) == 0:
            self.calc_interface_residues()

        check_all_interface = False
        if reschid is None:
            if verbose:
                print(f"WARNING: chain id for residue-{resid} is missed.")
                print("Possible chain pairs below:")
                for chain_pair, interfaces in self.interface_pairs.items():
                    if len(interfaces) >= 1:
                        print(
                            f" * [{chain_pair[0]} - {chain_pair[1]}] : {len(interfaces)}"
                        )
            check_all_interface = True

        is_in_interface = False
        locate_chain_pairs = ""
        for chain_pair, interfaces in self.interface_pairs.items():
            if (reschid in chain_pair or check_all_interface) and resid in interfaces:
                if reschid is not None:
                    # if reschid is specified, check whether resid is in "that" chain.
                    which_chain = chain_pair.index(reschid)
                    if resid in interfaces[:, which_chain]:
                        is_in_interface = True
                elif resid in interfaces:
                    # if not specified, just check resid is found in the interface region
                    is_in_interface = True

                if is_in_interface:
                    # accumulate chain_pairs for return
                    if len(locate_chain_pairs) != 0:
                        locate_chain_pairs += ","
                    locate_chain_pairs += "".join(chain_pair)

                if is_in_interface and verbose:
                    print(
                        f"res-{resid} in interface of pairs: {chain_pair[0]}-{chain_pair[1]}"
                    )
                if not check_all_interface:
                    break
            is_in_interface = False
        if len(locate_chain_pairs) != 0:
            is_in_interface = True

        return is_in_interface, locate_chain_pairs
