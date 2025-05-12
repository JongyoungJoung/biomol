#!/usr/bin/env python
import argparse
import os
import pdb as ptr
import shutil
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyemma  # type: ignore
from deeptime.clustering import KMeans as dt_kmeans
from deeptime.decomposition import TICA, vamp_score_cv
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.cluster import KMeans as sk_kmeans
from sklearn.metrics import silhouette_score as silh_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from biomol.protein import Protein as biomol_protein
from biomol.protein_ligand_complex import ProteinLigandComplex as PL_Com

# import mdshare


def check_existance_or_load_default(
    *, inpdb: str | None = None, intop: str | None = None
) -> tuple[str, int]:
    """
    If input pdb or input topfile is not given explicitely, find the default initial pdb or topfile.

    Default initial pdb:
        -> ./prep/*_initial_solv.pdb
        : convert this *_inital_solv.pdb into "desolvated" form of a system
    Default initial topfile:
        -> ./prep/*solv.prmtop

    """
    input_pdb_or_top = ""
    num_atoms = 0
    if inpdb is not None and intop is None:
        # NOTE: When a pdb is given,
        if Path(inpdb).exists():
            input_pdb_or_top = inpdb
        else:
            # NOTE: default. find prep/*_initial_solv.pdb as the initial pdb
            default_inpdb = [
                str(file_) for file_ in Path("./prep").rglob("*_initial_solv.pdb")
            ]
            if len(default_inpdb) == 0:
                raise FileNotFoundError(
                    "-inpdb is not given and also default ./prep/*_initial_solv.pdb does not exist"
                )
            input_pdb_or_top = "desolvated_input.pdb"
            # NOTE: When loading a pdb file into biomol.protein.Protein object,
            #       all the solvent residues are excluded by default.
            inpdb_obj = biomol_protein(default_inpdb[0])
            inpdb_obj.extract_as_one_pdbfile(outpdb_name=input_pdb_or_top)
            num_atoms = inpdb_obj.get_num_atoms()
    elif inpdb is None and intop is not None:
        if Path(intop).exists():
            if Path(intop).suffix[1:] != "prmtop":
                shutil.copy(intop, f"{Path(intop).stem}.prmtop")
                input_pdb_or_top = f"{Path(intop).stem}.prmtop"
            else:
                input_pdb_or_top = intop
        else:
            default_intop = [
                str(file_) for file_ in Path("./prep").rglob("*_solv.prmtop")
            ]
            if len(default_intop) == 0:
                raise FileNotFoundError(
                    "-top is not given and also default ./prep/*_solv.prmtop does not exist"
                )
            input_pdb_or_top = default_intop[0]

    return input_pdb_or_top, num_atoms


def extract_geometric_features(*, complex_obj: PL_Com, feature_type: list[str]) -> dict:
    features = {}
    if "atom_contact" in feature_type:
        complex_obj.find_protein_ligand_contact_atom_pairs()
        features["atom_contact"] = complex_obj.get_atomic_contact_pairs_as_array_index()
    if "hbond" in feature_type:
        # TODO: needs to be implemented.
        print("Warning: features 'hbond' are not implmeneted yet.")
        features["hbond"] = []
    if "hydrophobic" in feature_type:
        # TODO: needs to be implemented.
        print("Warning: feature 'hydrophobic' are not implemented yet.")
        features["hydrophobic"] = []

    return features


def collect_crds_of_trajectories(*, first_traj: int = 1, last_traj: int = -1) -> list:
    """
    Colloect trajectories' coordinates files.
    """
    coord_list = []
    traj_dirs = glob("traj_*")
    traj_dirs.sort()
    if len(traj_dirs) == 0:
        raise FileNotFoundError(
            "Trajectory directories not found under current position"
        )

    if last_traj == -1:
        last_traj = first_traj
    elif last_traj < first_traj:
        print(
            f"Warning: first traj id is {first_traj}, and last traj id is {last_traj}"
        )
        last_traj = first_traj
    file_ext = ["crd", "nc"]
    for tid in range(first_traj - 1, last_traj):
        print(f" -> Scanning Trajectory - {tid + 1}")
        traj_coords = []
        for ext in file_ext:
            traj_coords.extend(glob(f"{traj_dirs[tid]}/production/*.{ext}"))
        if len(traj_coords) == 0:
            print(f"Warning: trajectory - {tid + 1} does not have simulatioed data.")
            continue
        else:
            print(f"     - Num of coordinates: {len(traj_coords)}")
            coord_list.extend(traj_coords)

    abspath_coord_list = []  # Absolute path of coordinate files
    if len(coord_list) == 0:
        raise FileNotFoundError("There are no trajectory coordinate files under here.")
    else:
        coord_list.sort()
        abspath_coord_list = [str(Path(p).resolve()) for p in coord_list]

    return abspath_coord_list


def create_featurizers(
    *, topfile: str, feature_type: list[str], feature_data: dict
) -> dict:
    featurizers = {}
    topfile_found = ""
    topfile_found, _ = check_existance_or_load_default(intop=topfile)
    for feat_typ in feature_type:
        featurizers[feat_typ] = pyemma.coordinates.featurizer(topfile_found)
        if feat_typ == "atom_contact":
            featurizers[feat_typ].add_distances(feature_data[feat_typ], periodic=False)
        # FIX:
        # for other feature type, e.g. hbond, hydrophobic..
        # It is needed to idealize how to incorporate their geometric data into featurizer

    return featurizers


def load_crd_data(*, featurizers: dict, feature_type: list, crds: list, n_atoms: int):
    """
    Loading coordinate files to process the geoemtric features based on feature_type.
    """
    loaded_crd_data = {}
    for feat_type in feature_type:
        loaded_crd_data[feat_type] = pyemma.coordinates.load(
            crds,
            featurizers[feat_type],
            n_atoms=n_atoms,
        )

    return loaded_crd_data


def plot_feature_distribution(
    *, featurizers: dict, geometric_data: dict, show_plots: bool = False
):
    """
    Distribution of features' raw data.

    If the dimension of geometric data is larger than max_dim, which is restricted as 50 in PyEMMA,
    Only show distributions of data that have non-gaussian-like distribution.
    """
    max_dim = 50
    x_labels = {
        "atom_contact": "Atomic distance between contacted atom pairs",
        "hbond": "Hydrogen bond formation",
        "hydrophobic": "Hydrophobic cluster size",
    }
    # TODO:
    # When dim of data > 50,
    # only select non-gaussian-like data to be shown in histogram for visuality.
    for feat_type, featurizer in featurizers.items():
        # generate new plots
        fig, ax = plt.subplots(figsize=(10, featurizer.dimension()))
        pyemma.plots.plot_feature_histograms(
            np.concatenate(geometric_data[feat_type])[:, :max_dim],
            # feature_labels=featurizer,
            feature_labels=featurizer.describe()[:max_dim],
            ax=ax,
        )
        ax.set_xlabel(x_labels[feat_type])
        ax.set_ylabel("Histogram per dimension (normalized)")
        fig.tight_layout()
        if show_plots:
            plt.show()
        plt.savefig(f"dist_feature_{feat_type}.png")


def estimate_tica(
    *,
    geometric_data: dict,
    max_lag_time: int = 100,
    lag_time_interval: int = 10,
    num_cv: int = 5,
    show_plots: bool = False,
):
    """
    Esitamte TICA and display free energy landscape of geometric data.

    Argument:
        geomtric_data: dict, Extracted geometric data
        show_plots: bool, Showing free energy landscape according to configurational density

    """
    feat_types = list(geometric_data.keys())
    best_feature = ""
    best_score_of_best_feature = -1
    best_dim_of_best_feature = -1
    best_lag_time_index = -1
    best_dim_index = -1
    # TODO:
    # When I want to include all geometric data together, How can I merge data that has different dimensions?
    # if "all" not in geometric_data:
    #     feat_types.append("all")
    #     all_data = []
    #     for feat, data in geometric_data.items():
    #         all_data.append(data)
    lag_times = [1] + list(
        range(lag_time_interval, max_lag_time + 1, lag_time_interval)
    )

    #
    # Step 1. Estimation of optimal lag time and dimensions according to VAMP2 score
    #
    for feat in tqdm(feat_types, desc="Featrue type"):
        # For each feature data, maximum dimenstion is set to its dimension
        # TODO: if geometric_data's dimension is larger than 20, divde max_dim by 10 to get dimension search interval
        # 20, 10 : hard-coded -> needs to be changed
        #
        if geometric_data[feat][0].shape[1] > 20:
            dim_interval = geometric_data[feat][0].shape[1] // 10
        else:
            dim_interval = 1
        dims = list(range(1, geometric_data[feat][0].shape[1] + 1, dim_interval))

        all_scores = []
        fig, ax = plt.subplots(figsize=(12, 12))
        # for i, lag_t in tqdm(enumerate(lag_times), desc="Lag time"):
        for i in tqdm(range(len(lag_times)), desc="Lag time"):
            lag_t = lag_times[i]
            scores = []
            errors = []
            tica_estimator = TICA(lagtime=lag_t)
            tica_estimator.fit(np.concatenate(geometric_data[feat])).fetch_model()
            for dim in tqdm(dims, desc="Dimension", leave=False):
                tica_estimator.dim = dim
                vamp2_score = vamp_score_cv(
                    tica_estimator,
                    trajs=np.concatenate(geometric_data[feat]),
                    blocksize=lag_t,
                    blocksplit=True,
                    n=num_cv,
                )
                scores.append(vamp2_score.mean())
                errors.append(vamp2_score.std())
            all_scores.append(scores)
            scores = np.array(scores)
            errors = np.array(errors)

            # plot
            color = f"C{i}"
            ax.fill_between(
                dims, scores - errors, scores + errors, alpha=0.3, facecolor=color
            )
            # TODO:
            # change formatting of lag_time legend as automatically determining units given simulation data
            ax.plot(
                dims, scores, "--o", color=color, label=f"lag_{lag_t * 0.02:.2f} ns"
            )
        ax.legend()
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("VAMP2 Score")

        fig.tight_layout()
        if show_plots:
            plt.show()
        plt.savefig(f"vamp2_scores_{feat}.png")
        # Find optimal values of lag_time and dimension via all_scores
        # TODO:
        # Check meaning of VAMP2 score and
        # how I can interpreet scores to detemine optimal hyper-parameters,
        # that are lag time and dimension
        all_scores = np.array(all_scores)
        best_score_index = np.unravel_index(np.argmax(all_scores), all_scores.shape)
        best_score = all_scores[best_score_index]
        # if best score of this features is all the best, update.
        # NOTE: -> could be changed according to the meaning of VAMP2 score.
        if best_feature is None or best_score > best_score_of_best_feature:
            best_feature = feat
            best_score_of_best_feature = best_score
            best_lag_time_index = best_score_index[0]
            best_dim_of_best_feature = dims[best_score_index[1]]
        # TODO:
        # How to check converged area of dimension in terms of VAMP2 score
        # When I found the first converged regions of dimension, it need to investigate finely dimenstions in that area
        # to get optimal dimensions..
    print(f"Best features for TICA analysis: {best_feature}")
    print(f"Best lag Time: {lag_times[best_lag_time_index]}")
    print(f"Best dimension : {best_dim_of_best_feature}")

    # TICA estimator with the optimal lag time about best feature's geometric data
    # TODO: var_cutoff ?? what is the best?
    opt_tica_estimator = TICA(
        lagtime=lag_times[best_lag_time_index],
        dim=best_dim_of_best_feature,
        var_cutoff=0.95,
    )
    opt_tica_model = opt_tica_estimator.fit(
        np.concatenate(geometric_data[best_feature])
    ).fetch_model()
    transformed_geometric_data = opt_tica_model.transform(
        np.concatenate(geometric_data[best_feature])
    )

    return transformed_geometric_data, best_feature


def clustering_kmeans_and_plot(
    *,
    transformed_geometric_data: list | np.ndarray,
    best_feature: str,
    max_cluster: int = 100,
    max_tica_for_cluster: int = 2,
    silhouette_convergence_threshold: float = 0.001,
    show_plots: bool = False,
):
    """
    Perform KMeans clustering by fitting given geometric data to the optimal number of clusters.

    Args:
        transformed_geometric_data: list|np.ndarray
    Returns:
        best_k: int, number of clusters by optimally divide dataset.
    """
    #
    # Step 2. Evaluation of optimal number of clusters for K-Means clustering
    #
    kcenter_range = range(2, max_cluster + 1)
    inertias = []
    silhouettes = []
    prev_silhouette_score = -1.0
    i = 0
    for k in tqdm(kcenter_range, desc="Evaluating optimal cluster number using KMeans"):
        kmeans = sk_kmeans(n_clusters=k, random_state=1234)
        # if max_tica_for_cluster = 2 -> only consider 1st and 2nd TICA
        labels = kmeans.fit_predict(
            transformed_geometric_data[:, 0:max_tica_for_cluster]  # type: ignore
        )
        inertias.append(kmeans.inertia_)
        silhouette_for_this_n_cluster = silh_score(
            transformed_geometric_data[:, 0:max_tica_for_cluster],  # type: ignore
            labels,
        )
        silhouettes.append(silhouette_for_this_n_cluster)
        i += 1
        if (
            abs(prev_silhouette_score - silhouette_for_this_n_cluster)
            < silhouette_convergence_threshold
            and prev_silhouette_score > silhouette_for_this_n_cluster
            and prev_silhouette_score > 0.0
        ):
            print(
                "  -> In evaluating optimal cluster number, Silhouette score looks convergned."
            )
            break
        prev_silhouette_score = silhouette_for_this_n_cluster

    # plot inertia and silhouettes scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(kcenter_range[:i], inertias, marker="o")
    plt.title("Elbow Method\nCompactness of clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(kcenter_range[:i], silhouettes, marker="o")
    plt.title("Silhouette Score\nQualities of clustering")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Scores")
    plt.grid()

    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(f"clusetering_qualtiy_kmeans_{best_feature}.png")

    best_k = kcenter_range[np.argmax(silhouettes)]
    print(f"Optimal number of cluster : {best_k}")

    return best_k


def free_energy_landscape(
    *,
    transformed_geometric_data: list | np.ndarray,
    best_feature: str,
    best_k: int,
    max_tica_for_cluster: int = 2,
    n_cv: int = 5,
    n_eval_bandwidth_kernel: int = 100,
    n_samples_per_cluster: int = 100,
    show_plots: bool = False,
    ratio_of_density_in_closeness_distance: float = 0.9,
):
    fig, ax = plt.subplots()

    #
    # Step 3. Visualization of Free Energy Landscape of configurations upon TICA spaces
    #
    kmeans_estimator = dt_kmeans(n_clusters=best_k, progress=tqdm)  # deeptime Kmeans
    clustering = kmeans_estimator.fit(
        transformed_geometric_data[:, 0:max_tica_for_cluster]  # type: ignore
    ).fetch_model()
    if clustering.cluster_centers is None:  # type: ignore
        raise RuntimeError("Error: deeptime KMeans clustering failed.")

    #  3-1. Free Energy Landscape with centers
    pyemma.plots.plot_free_energy(
        transformed_geometric_data[:, 0],  # type: ignore
        transformed_geometric_data[:, 1],  # type: ignore
        ax=ax,
    )
    # for labeling center of clusters
    center_label = [
        f"Cluster-{i + 1}"
        for i in range(clustering.cluster_centers.shape[0])  # type: ignore
    ]
    ax.plot(*clustering.cluster_centers.T, "X", color="black")  # type: ignore
    for i in range(len(center_label)):
        ax.text(
            clustering.cluster_centers.T[0][i],  # type: ignore
            clustering.cluster_centers.T[1][i] + 0.5,  # type: ignore
            center_label[i],
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="darkblue",
        )
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(f"free_energy_landscape_{best_feature}.png")

    # 3-2. Evaludation of optimal Bandwidth
    bandwidths = np.linspace(0.01, 1.0, n_eval_bandwidth_kernel)
    param_grid = {"bandwidth": bandwidths}
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=123)
    kernel_scores = []
    best_kernel_bandwidth_score = -np.inf
    best_kernel_model = None
    for bw in tqdm(
        bandwidths, desc="Evaluation of optimal bandwith for Kernel Density"
    ):
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        fold_scores = []

        for train_idx, test_idx in cv.split(
            transformed_geometric_data[:, 0:max_tica_for_cluster]  # type: ignore
        ):  # type: ignore
            kde_fold = clone(kde)
            kde_fold.fit(  # type: ignore
                transformed_geometric_data[:, 0:max_tica_for_cluster][train_idx]  # type: ignore
            )
            score = kde_fold.score(  # type: ignore
                transformed_geometric_data[:, 0:max_tica_for_cluster][test_idx]  # type: ignore
            )
            fold_scores.append(score)
        mean_score = np.mean(fold_scores)
        kernel_scores.append(mean_score)

        if mean_score > best_kernel_bandwidth_score:
            best_kernel_bandwidth_score = mean_score
            best_kernel_model = clone(kde)
    best_bandwidth = best_kernel_model.bandwidth  # type: ignore
    print(f"Optimal bandwidth : {best_bandwidth}")

    # 3-3. Set optimal Kernel density model and collect near center samples of each cluster
    opt_kde = KernelDensity(bandwidth=best_bandwidth, kernel="gaussian")
    opt_kde.fit(transformed_geometric_data[:, 0:max_tica_for_cluster])  # type: ignore
    density = np.exp(
        opt_kde.score_samples(transformed_geometric_data[:, 0:max_tica_for_cluster])  # type: ignore
    )  # type: ignore
    print("density:", density)
    # calculated distances between samples and cluster centers
    distances = cdist(
        transformed_geometric_data[:, 0:max_tica_for_cluster],  # type: ignore
        clustering.cluster_centers,  # type: ignore
    )
    # collect cluster's near center samples
    selected_points = []
    selected_indice_sets = []
    for cluster_idx in range(clustering.cluster_centers.shape[0]):  # type: ignore
        # NOTE: closeness_score
        # The closeness of a sample with respect to the cluster center is determined
        # not only by density of a sample but also by its distance to the center.
        # Density is 90% and distance 10%.
        closeness_score = (
            ratio_of_density_in_closeness_distance * density
            - (1.0 - ratio_of_density_in_closeness_distance) * distances[:, cluster_idx]
        )

        top_indices = np.argsort(closeness_score[-n_samples_per_cluster:])
        selected_points.append(transformed_geometric_data[top_indices])
        selected_indice_sets.append(top_indices)
    # write out the indices of selected sample points into data files.
    for i, indices in enumerate(selected_indice_sets):
        outfile = open(f"selected_config_snapshots_cluster_{i:02d}.dat", "w")
        for id_ in indices:
            outfile.write(f"{id_ + 1}\n")
        outfile.close()

    # 3-3. Free Energy Landscape with centers and their nearest neighbors
    fig, ax = plt.subplots()
    for point in selected_points:
        ax.plot(*point[:0:max_tica_for_cluster].T, "k.", alpha=0.05)
    ax.plot(*clustering.cluster_centers.T, "y", markersize=10)  # type: ignore
    pyemma.plots.plot_free_energy(
        transformed_geometric_data[:, 0],  # type: ignore
        transformed_geometric_data[:, 1],  # type: ignore
        ax=ax,
    )
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(f"free_energy_landscape_{best_feature}_with_samples.png")


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(
        "Clustering protein-ligand binding poses from MD trajectories"
    )
    parser.add_argument(
        "-inpdb",
        dest="inpdb",
        type=str,
        default="initial.pdb",
        help="Initial pdb structures to detect structural features.",
    )
    parser.add_argument(
        "-top",
        dest="topology",
        type=str,
        default="amber.top",
        help="Amber topology files for analyzing trajectories",
    )
    parser.add_argument(
        "-t0",
        "--traj-0",
        dest="start_traj",
        type=int,
        default=1,
        help="First trajectory id for analysis, default: 1 (traj_01)",
    )
    parser.add_argument(
        "-t1",
        "--traj-1",
        dest="end_traj",
        type=int,
        default=-1,
        help="Last trajectory id for analysis, default: -1 (not assigned. just do about -traj0)",
    )
    parser.add_argument(
        "-feat",
        "--feature",
        choices=["atom_contact", "hbond", "hydrophobic"],
        default=["atom_contact"],
        nargs="+",
        dest="md_features",
        help="Features for used protein-ligand poses analysis. default: atom_contact",
    )
    parser.add_argument(
        "-show-plot",
        dest="show_plots",
        action="store_true",
        help="Show plots of each steps",
    )
    parser.add_argument(
        "-wdir",
        "--workding-directory",
        type=str,
        default="msm_pro_lig",
        dest="wdir",
        help="Workding directory for MSM modeling, default='./msm'",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print("1. Generate Protein-Ligand Complex Object")
    # NOTE: loading initial pdb file
    # if not exist, loading ./prep/*_initial_solv.pdb by converting it into desolvated pdb
    input_pdb, num_atoms = check_existance_or_load_default(inpdb=args.inpdb)
    pl_complex = PL_Com(inpdb=input_pdb)

    print(f"2. Extracting Geometric Features from Initial Structure: {args.inpdb}")
    # NOTE: Get geometry features
    features_geometric_data = extract_geometric_features(
        complex_obj=pl_complex, feature_type=args.md_features
    )

    print("3. Collecting Coordinates from Selected Trajectories")
    # NOTE: Collect coordinate files
    # coord_list : absoulte paths of coordinate files
    coord_list = collect_crds_of_trajectories(
        first_traj=args.start_traj, last_traj=args.end_traj
    )

    print("4. Generate PyEMMA Featurizer")
    # NOTE: Featurizers of each feature type
    featurizers = create_featurizers(
        topfile=args.topology,
        feature_type=args.md_features,
        feature_data=features_geometric_data,
    )

    cwd = Path.cwd()
    # Moving into workding directory
    if not Path(args.wdir).exists():
        Path(args.wdir).mkdir()
    os.chdir(args.wdir)

    print("5. Load coordinate files")
    # NOTE: load top and datafiles
    crd_data = load_crd_data(
        featurizers=featurizers,
        feature_type=args.md_features,
        crds=coord_list,
        n_atoms=num_atoms,
    )

    print("6. Plot Distribution of Geometric Features for Protein-Ligand Binding")
    # NOTE: Plot distribution of raw data of geometric features
    plot_feature_distribution(
        featurizers=featurizers,
        geometric_data=crd_data,
        show_plots=args.show_plots,
    )

    print("7. Esitmation of TICA")
    # NOTE: TICA estimation
    transformed_geometric_data, best_feature_name = estimate_tica(
        geometric_data=crd_data,
        max_lag_time=100,
        lag_time_interval=10,
        num_cv=5,
        show_plots=args.show_plots,
    )

    print("8. Evaluation of K-Means Clustering for given Geometric Features")
    # NOTE: KMeans clustering and its quality assessment
    best_num_cluster = clustering_kmeans_and_plot(
        transformed_geometric_data=transformed_geometric_data,
        best_feature=best_feature_name,
        max_cluster=100,
        max_tica_for_cluster=2,
        show_plots=args.show_plots,
    )

    print("9. Plot Free Energy Landscape and Extracting Cluster-Center Near Samples")
    # NOTE: Free energy landscape with & without n_samples_per_cluster
    free_energy_landscape(
        transformed_geometric_data=transformed_geometric_data,
        best_feature=best_feature_name,
        best_k=best_num_cluster,
        max_tica_for_cluster=2,
        n_cv=5,
        n_eval_bandwidth_kernel=100,
        n_samples_per_cluster=100,
        ratio_of_density_in_closeness_distance=0.9,
        show_plots=args.show_plots,
    )

    os.chdir(cwd)


if __name__ == "__main__":
    main()
