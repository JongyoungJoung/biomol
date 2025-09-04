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
        # HACK: needs to be implemented.
        print("Warning: features 'hbond' are not implmeneted yet.")
        features["hbond"] = []
    if "hydrophobic" in feature_type:
        # HACK: needs to be implemented.
        print("Warning: feature 'hydrophobic' are not implemented yet.")
        features["hydrophobic"] = []

    return features


def collect_crds_of_trajectories(
    *,
    first_traj: int = 1,
    last_traj: int = -1,
    num_crd_per_traj: int = -1,
) -> tuple[list, dict]:
    """
    Colloect trajectories' coordinates files.
    """
    coord_list = []
    traj_dirs = glob("traj_*")  # NOTE: Do not change traj_* to other name.
    traj_dirs.sort()
    crd_count_per_traj = {}

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
            if num_crd_per_traj == -1:
                print("       All coordinates are selected.")
                coord_list.extend(traj_coords)
                # NOTE: for indexing frames from multiple trajectories
                crd_count_per_traj[tid + 1] = len(traj_coords)
            else:
                if len(traj_coords) >= num_crd_per_traj:
                    print(f"       {num_crd_per_traj} coordinates are selected.")
                    coord_list.extend(traj_coords[:num_crd_per_traj])
                    # NOTE: for indexing frames from multiple trajectories
                    crd_count_per_traj[tid + 1] = num_crd_per_traj
                else:
                    print("       All coordinates are selected.")
                    coord_list.extend(traj_coords)
                    # NOTE: for indexing frames from multiple trajectories
                    crd_count_per_traj[tid + 1] = len(traj_coords)

    abspath_coord_list = []  # Absolute path of coordinate files
    if len(coord_list) == 0:
        raise FileNotFoundError("There are no trajectory coordinate files under here.")
    else:
        coord_list.sort()
        abspath_coord_list = [str(Path(p).resolve()) for p in coord_list]

    return abspath_coord_list, crd_count_per_traj


def create_featurizers(
    *, topfile: str, feature_type: list[str], feature_data: dict
) -> dict:
    featurizers = {}
    topfile_found = ""
    topfile_found, _ = check_existance_or_load_default(intop=topfile)
    for feat_typ in feature_type:
        featurizers[feat_typ] = pyemma.coordinates.featurizer(topfile_found)
        if feat_typ == "atom_contact":
            featurizers[feat_typ].add_distances(feature_data[feat_typ], periodic=True)
        # HACK:
        # for other feature type, e.g. hbond, hydrophobic..
        # It is needed to idealize how to incorporate their geometric data into featurizer

    return featurizers


def load_crd_data(
    *, featurizers: dict, feature_type: list, crds: list, n_atoms: int, stride: int = 1
):
    """
    Loading coordinate files to process the geoemtric features based on feature_type.
    """
    loaded_crd_data = {}
    for feat_type in feature_type:
        loaded_crd_data[feat_type] = pyemma.coordinates.load(
            crds,
            featurizers[feat_type],
            n_atoms=n_atoms,
            stride=stride,
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
    max_tic_dim_for_cluster: int = 2,
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
        # if max_tic_dim_for_cluster = 2 -> only consider 1st and 2nd TICA
        labels = kmeans.fit_predict(
            transformed_geometric_data[:, 0:max_tic_dim_for_cluster]  # type: ignore
        )
        inertias.append(kmeans.inertia_)
        silhouette_for_this_n_cluster = silh_score(
            transformed_geometric_data[:, 0:max_tic_dim_for_cluster],  # type: ignore
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
    max_tic_dim_for_cluster: int = 2,
    n_cv: int = 5,
    n_eval_bandwidth_kernel: int = 100,
    sample_select_ratio_cluster: float = 0.1,
    sample_select_num_cluster: int = -1,
    min_sample_select_num_cluster: int = 30,
    ratio_of_density_in_closeness_distance: float = 0.9,
    seq_frame_id_to_traj_frame_id: dict = {0: []},
    show_plots: bool = False,
):
    """
    Free Energy Landscape.

    Args:
        transformed_geometric_data : list | np.ndarray, transformed geometric data by TICA model
        best_feature: str, Name of best feature
        best_k : int, Optimal number of K-means cluster
        max_tic_dim_for_cluster: int, dimension size to plot free energy landscape
        n_cv: int, num of cross validation
        n_eval_bandwidth_kernel: int, num of trial to determine optimal bandwidth of kernel density
        sample_select_ratio_cluster: float, 0.0 ~ 1.0, percentage of selection of samples from each cluster,
                                     top n percentage of highest density samples.
                                     default: 0.1 (10 percentage of samples with highest density)
                                     THIS option is the first priority.
        sample_select_num_cluster: int, number of nearest samples of cluster,
                                   default: 1000 (samples)
        min_sample_select_num_cluster: int, minimum number of selected samples, default: 30
                                       if total_n_sample_of_cluster * select_ratio is less than this value,
                                       select "min_sample_select_num_cluster" of samples.
        ratop_of_density_in_closeness_distance: float, percentage of sample density when determining
                                                closeness distances to individual cluster centers.
        seq_frame_id_to_traj_frame_id: (sequential_frame_id: (traj_id, frame_id_in_traj)
                                     To convert data point's id (sequential_frame_id) to its original position.
                                     (its trajectory id and frame id in there)
    """
    _, ax = plt.subplots()

    # NOTE: ===========================================================================
    # Step 3. Visualization of Free Energy Landscape of configurations upon TICA spaces
    # 3-0. K-means clustering
    # =================================================================================
    kmeans_estimator = dt_kmeans(n_clusters=best_k, progress=tqdm)  # deeptime Kmeans
    clustering = kmeans_estimator.fit(
        transformed_geometric_data[:, 0:max_tic_dim_for_cluster]  # type: ignore
    ).fetch_model()
    # individual clusters' positions in data
    clustering_points = kmeans_estimator.fit_transform(
        transformed_geometric_data[:, :max_tic_dim_for_cluster]  # type: ignore
    )
    cluster_points_indices = [
        np.where(clustering_points == cid)[0] for cid in range(best_k)
    ]
    if clustering.cluster_centers is None:  # type: ignore
        raise RuntimeError("Error: deeptime KMeans clustering failed.")

    # NOTE: ===========================================================================
    #  3-1. Free Energy Landscape with centers
    # =================================================================================
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
    # cluster center points
    ax.plot(
        *clustering.cluster_centers.T,  # type: ignore
        "X",
        color="black",
        markersize=10,
        markeredgewidth=3,
    )
    # labeling cluster center points
    for i in range(len(center_label)):
        ax.text(
            clustering.cluster_centers.T[0][i],  # type: ignore
            clustering.cluster_centers.T[1][i] + 0.1,  # type: ignore
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

    # NOTE: ===========================================================================
    # 3-2. Evaludation of optimal Bandwidth
    # =================================================================================
    bandwidths = np.linspace(0.01, 1.0, n_eval_bandwidth_kernel)
    param_grid = {"bandwidth": bandwidths}
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=123)
    kernel_scores = []
    kernel_scores_std = []
    best_kernel_bandwidth_score = -np.inf
    best_kernel_model = None
    kde_score_decrease_count = 0
    prev_kde_score = None
    kde_bandwidth_stop_count = 10
    for bw in tqdm(
        bandwidths, desc="Evaluation of optimal bandwith for Kernel Density"
    ):
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        fold_scores = []

        for train_idx, test_idx in cv.split(
            transformed_geometric_data[:, :max_tic_dim_for_cluster]  # type: ignore
        ):  # type: ignore
            kde_fold = clone(kde)
            kde_fold.fit(  # type: ignore
                transformed_geometric_data[:, :max_tic_dim_for_cluster][train_idx]  # type: ignore
            )
            score = kde_fold.score(  # type: ignore
                transformed_geometric_data[:, :max_tic_dim_for_cluster][test_idx]  # type: ignore
            )
            fold_scores.append(score)
        mean_score = np.mean(fold_scores)
        kernel_scores.append(mean_score)
        kernel_scores.append(np.std(fold_scores))
        if prev_kde_score is None:
            prev_kde_score = mean_score
        if prev_kde_score > mean_score:
            kde_score_decrease_count += 1
        if kde_score_decrease_count > kde_bandwidth_stop_count:
            break

        prev_kde_score = mean_score
        if mean_score > best_kernel_bandwidth_score:
            best_kernel_bandwidth_score = mean_score
            best_kernel_model = clone(kde)

    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bandwidths[: len(kernel_scores)], kernel_scores, marker="o")
    ax.set_title("Kernel Scores")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("KDE score")
    plt.savefig(f"kernel_density_{best_feature}.png")

    best_bandwidth = best_kernel_model.bandwidth  # type: ignore
    print(f"Optimal bandwidth : {best_bandwidth}")

    # NOTE: ===============================================================================
    # 3-3. Set optimal Kernel density model and collect near center samples of each cluster
    # =====================================================================================
    opt_kde = KernelDensity(bandwidth=best_bandwidth, kernel="gaussian")
    opt_kde.fit(transformed_geometric_data[:, :max_tic_dim_for_cluster])  # type: ignore
    # KernelDensity.score_samples : log-density of data points
    # np.exp(..) : convert log-density to non-log probability
    # density : 1D array of estimated density (probability) of data upon 2D feature space.
    density = np.exp(
        opt_kde.score_samples(transformed_geometric_data[:, :max_tic_dim_for_cluster])  # type: ignore
    )
    cluster_density_sum = [
        np.sum(density[cluster_points_indices[cid]]) for cid in range(best_k)
    ]
    cluster_density_avg = [
        np.sum(density[cluster_points_indices[cid]])
        / cluster_points_indices[cid].shape[0]
        for cid in range(best_k)
    ]
    cluster_density_avg = np.array(cluster_density_avg)
    densest_cluster_id = int(np.argmax(cluster_density_avg))

    cluster_density_log = open("clusters_avg_density.csv", "w")
    cluster_density_log.write("cluster_id,avg_density,total_density,num_samples\n")
    print("Sample density of clusters")
    for cid in range(best_k):
        print(
            f"Cluster-{cid + 1} : {cluster_density_avg[cid]:.3f} "
            f"({cluster_density_sum[cid]:.3f}/{cluster_points_indices[cid].shape[0]})"
        )
        data_line = f"{cid + 1},{cluster_density_avg[cid]:.3f},"
        data_line += f"{cluster_density_sum[cid]:.3f},"
        data_line += f"{cluster_points_indices[cid].shape[0]}\n"

        cluster_density_log.write(data_line)
    cluster_density_log.close()

    # calculated Eucledian distances between samples and cluster centers
    distances = cdist(
        transformed_geometric_data[:, :max_tic_dim_for_cluster],  # type: ignore
        clustering.cluster_centers,  # type: ignore
    )
    # NOTE: ===========================================================================
    # 3-4. Collect cluster's near center samples
    # =================================================================================
    selected_points = []
    selected_indice_sets = []
    all_indices_sets_sort_by_closeness = []
    closeness_score_sets = []
    # HACK:
    #   - needs to be refactored for determining best n_select
    #       n_select_weight = 0.1 -> as not hard-coded.
    #   - when I refactor the code, consider variables following,
    #       sample_select_ratio_cluster, min_sampl_select_num_cluster
    n_select_weight = 0.1
    for cid in range(best_k):
        # NOTE: closeness_score
        # The closeness of a sample with respect to the cluster center is determined
        # not only by density of a sample but also by its distance to the center.
        # Density is 90% and distance 10%.

        # NOTE: ONLY position indices of this cluster points
        cls_pnt_ids = cluster_points_indices[cid]
        # NOTE: ONLY geometric data for this cluster points
        cls_geometric_data = transformed_geometric_data[cls_pnt_ids]
        # NOTE: ONLY eucledian distance of this cluster points to this cluster center point.
        dist_to_this_cls_center = distances[:, cid][cls_pnt_ids]

        center_closeness_score = (
            ratio_of_density_in_closeness_distance * density[cls_pnt_ids]
            - (1.0 - ratio_of_density_in_closeness_distance) * dist_to_this_cls_center
        )
        closeness_score_sets.append(center_closeness_score)
        n_select = 0

        # HACK:
        #   - if sample_select_num_cluster (default: 1000) is larger than 10 % of number of points
        #       in this cluster,
        #   - just collect 10 (default) % of samples according to the closeness score
        # => This part requires REFACTORING...
        # ======================================================================================

        # if (
        #     min_sample_select_num_cluster
        #     > len(cls_pnt_ids) * sample_select_ratio_cluster
        # ):
        #     # n_select_sample < 30
        #     n_select = min_sample_select_num_cluster
        # elif (
        #     sample_select_num_cluster * n_select_weight
        #     > int(len(cls_pnt_ids) * sample_select_ratio_cluster)
        #     >= min_sample_select_num_cluster
        # ):
        #     # 30 <= n_select_sample < 1000 * 0.1
        #     n_select = int(len(cls_pnt_ids) * sample_select_ratio_cluster)
        # elif (
        #     sample_select_num_cluster
        #     > int(len(cls_pnt_ids) * sample_select_ratio_cluster)
        #     >= sample_select_num_cluster * n_select_weight
        # ):
        #     # 1000 * 0.1 < n_select_sample < 1000
        #     n_select = int(sample_select_num_cluster * n_select_weight)
        # else:
        #     # n_select_sample >= 1000
        #     n_select = sample_select_num_cluster

        if sample_select_num_cluster != -1:
            n_select = sample_select_num_cluster
        else:
            n_select = int(len(cls_pnt_ids)) * sample_select_ratio_cluster

        # HACK: =================================================================================
        # NOTE: where_is_max_n: this is not original indcies.
        # The indices of top "n_select" of closensess scores IN THE ARRAY of "cls_pnt_ids"
        where_is_max_n = np.argsort(center_closeness_score)[-n_select:]
        # original data indices in this cluster
        top_org_indices_of_this_cluster = cls_pnt_ids[where_is_max_n]
        # store indices for later print out selected_indices
        selected_indice_sets.append(top_org_indices_of_this_cluster)
        # for later visualizing selected points upon free energy landscapes
        selected_points.append(cls_geometric_data[where_is_max_n])
        # NOTE:
        #  - for storing all data points's indices in this cluster according to data's closeness values
        #  - all_indices_sets_sort_by_closeness[i]
        #    : indices of data point sort by closeness with ascending order
        all_indices_sets_sort_by_closeness.append(
            cls_pnt_ids[np.argsort(center_closeness_score)]
        )
    # NOTE:
    # - Write out the indices of selected sample points into data files.
    # - Sort clusters according to their average density "in descending order."
    #   from large density cluster -> small density cluster
    for rank_idx, org_idx in enumerate(cluster_density_avg.argsort()[::-1]):
        # NOTE: distance data, closeness data
        #       density is common for all clusters
        dist_to_this_cls_cent = distances[:, org_idx]
        # NOTE: for SELECTED data points
        indices = selected_indice_sets[org_idx]
        selected_points_outfile = open(
            f"rank_{rank_idx + 1:02d}_cluster_id_{org_idx + 1:02d}_"
            + f"avg_density_{cluster_density_avg[org_idx]:.5f}_sub_{indices.shape[0]:d}_samples.csv",
            "w",
        )
        selected_points_outfile.write(
            "#data_id,traj_id,frame_id_in_traj,distance,density,closeness\n"
        )
        # Reverse order: large density point -> small density point
        for id_ in indices[::-1]:
            traj_id, frame_id_in_traj = seq_frame_id_to_traj_frame_id[id_]
            dist = dist_to_this_cls_cent[id_]
            rho = density[id_]
            closeness = (
                ratio_of_density_in_closeness_distance * rho
                - (1.0 - ratio_of_density_in_closeness_distance) * dist
            )
            # HACK:
            # - id_ : sequential id of a frame in merged list of frames
            # - traj_id : original trajectory id (1,2,...) -> I checked it when storing traj_id
            # - frame_id_in_traj : frame's id in its trajectory
            selected_points_outfile.write(
                f"{id_ + 1},{traj_id},{frame_id_in_traj + 1},{dist},{rho},{closeness}\n"
            )
        selected_points_outfile.close()

        # NOTE: for all data points
        # IMPORTANT -> for surveying optimal samples size for weighted average of free energies.
        all_indices = all_indices_sets_sort_by_closeness[org_idx]
        all_points_outfile = open(
            f"rank_{rank_idx + 1:02d}_cluster_id_{org_idx + 1:02d}_"
            + f"avg_density_{cluster_density_avg[org_idx]:.5f}_all_samples.csv",
            "w",
        )
        all_points_outfile.write(
            "#data_id,traj_id,frame_id_in_traj,distance,density,closeness\n"
        )
        # Reverse order: large density point -> small density point
        for id_ in all_indices[::-1]:
            traj_id, frame_id_in_traj = seq_frame_id_to_traj_frame_id[id_]
            dist = dist_to_this_cls_cent[id_]
            rho = density[id_]
            closeness = (
                ratio_of_density_in_closeness_distance * rho
                - (1.0 - ratio_of_density_in_closeness_distance) * dist
            )
            all_points_outfile.write(
                f"{id_ + 1},{traj_id},{frame_id_in_traj + 1},{dist},{rho},{closeness}\n"
            )
        all_points_outfile.close()

    # NOTE: ===========================================================================
    # 3-5. Free Energy Landscape with centers and their nearest neighbors
    # =================================================================================
    _, ax = plt.subplots()
    pyemma.plots.plot_free_energy(
        transformed_geometric_data[:, 0],  # type: ignore
        transformed_geometric_data[:, 1],  # type: ignore
        ax=ax,
    )
    for i, point in enumerate(selected_points):
        ax.plot(
            *point[:, :max_tic_dim_for_cluster].T,
            "X",
            markersize=10,
            # color=f"C{i}",
            color="k",
            alpha=0.3,
        )
    ax.plot(*clustering.cluster_centers.T, "y.", markersize=10, markeredgewidth=3)  # type: ignore
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(f"free_energy_landscape_{best_feature}_with_cluster_samples.png")

    # NOTE: ===========================================================================
    # 3-6. Free Energy Landscape with samples of most dense area.
    # =================================================================================
    # free energy density
    _, ax = plt.subplots()
    pyemma.plots.plot_free_energy(
        transformed_geometric_data[:, 0],  # type: ignore
        transformed_geometric_data[:, 1],  # type: ignore
        ax=ax,
    )
    ax.plot(
        # *transformed_geometric_data[densest_area_indices][:, 0:max_tic_dim_for_cluster].T,
        *selected_points[densest_cluster_id][:, :max_tic_dim_for_cluster].T,
        "X",
        markersize=10,
        color="k",
        alpha=0.3,
    )
    ax.plot(*clustering.cluster_centers.T, "rx", markersize=10, markeredgewidth=3)  # type: ignore
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(f"free_energy_landscape_{best_feature}_of_densest_area.png")


def float_range(start, end):
    def check_float(value):
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating-point number.")
        if not start <= value <= end:
            raise argparse.ArgumentTypeError(f"Value must be in range[{start},{end}.]")
        return value

    return check_float


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
        default="amber.prmtop",
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
        "-stride",
        "--frame-stride",
        dest="stride",
        type=int,
        default=1,
        help="Loading stide of frames per a given trajectory file. default: 1."
        + " CAUTION: stride MUST NOT be larger than number of frames in a trajectory file",
    )
    parser.add_argument(
        "-ncrd_per_traj",
        "--num-crd-per-each-traj",
        dest="num_crd_per_traj",
        type=int,
        default=-1,
        help="Numbner of coordinates to be selected in each trajectory. -1 -> select all",
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
        "-sample-ratio",
        dest="ratio_sample",
        type=float_range(0.0, 1.0),
        default=0.1,
        help="Percentage of samples selected from each cluster,"
        + " top N percentage of high density samples. default: 10 percentage",
    )
    parser.add_argument(
        "-sample-num",
        dest="num_sample",
        type=int,
        default=-1,
        help="Number of samples selected from each cluster. default:1000 (samples/cluster)",
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
    pl_complex = PL_Com(infile=input_pdb)

    print(f"2. Extracting Geometric Features from Initial Structure: {args.inpdb}")
    # NOTE: Get geometry features
    features_geometric_data = extract_geometric_features(
        complex_obj=pl_complex, feature_type=args.md_features
    )

    print("3. Collecting Coordinates from Selected Trajectories")
    # NOTE: Collect coordinate files
    # coord_list : absoulte paths of coordinate files
    coord_list, crd_count_per_traj = collect_crds_of_trajectories(
        first_traj=args.start_traj,
        last_traj=args.end_traj,
        num_crd_per_traj=args.num_crd_per_traj,
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
    #   crd_data:
    #       - length = number of crd files
    #       - shape[0] : number of frames of each crd file
    crd_data = load_crd_data(
        featurizers=featurizers,
        feature_type=args.md_features,
        crds=coord_list,
        n_atoms=num_atoms,
        stride=args.stride,
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

    # NOTE: collect number of frames of each crds in trajectories for "BEST feature type"
    #   data_index_to_crd_id_traj_id = {sequential_id: (traj_id, traj_frame_id)}
    #           - sequential_id: frame id when merging all the crds' frames sequentially
    #           - traj_id: trajectory id (start_traj ~ end_traj)
    #           - traj_frame_id: frame id in this trajectory
    data_index_to_crd_id_traj_id = {}
    accum_crd_num = 0
    seq_frame_id = 0

    for traj_id, crd_num_of_traj in crd_count_per_traj.items():
        frame_id_in_this_traj = 0
        for crd_id in range(accum_crd_num, accum_crd_num + crd_num_of_traj):
            # NOTE:
            # only consider "best_feature_name"'s feature data
            for frame_id in range(crd_data[best_feature_name][crd_id].shape[0]):
                data_index_to_crd_id_traj_id[seq_frame_id] = (
                    traj_id,
                    frame_id_in_this_traj,
                )
                seq_frame_id += 1
                frame_id_in_this_traj += 1
        accum_crd_num += crd_num_of_traj

    print("8. Evaluation of K-Means Clustering for given Geometric Features")
    # NOTE: KMeans clustering and its quality assessment
    best_num_cluster = clustering_kmeans_and_plot(
        transformed_geometric_data=transformed_geometric_data,
        best_feature=best_feature_name,
        max_cluster=100,
        max_tic_dim_for_cluster=2,
        show_plots=args.show_plots,
    )

    print("9. Plot Free Energy Landscape and Extracting Cluster-Center Near Samples")
    # NOTE: Free energy landscape with & without sample_select_num_cluster
    free_energy_landscape(
        transformed_geometric_data=transformed_geometric_data,
        best_feature=best_feature_name,
        best_k=best_num_cluster,
        max_tic_dim_for_cluster=2,
        n_cv=5,
        n_eval_bandwidth_kernel=100,
        sample_select_ratio_cluster=args.ratio_sample,
        sample_select_num_cluster=args.num_sample,
        ratio_of_density_in_closeness_distance=0.9,
        seq_frame_id_to_traj_frame_id=data_index_to_crd_id_traj_id,
        show_plots=args.show_plots,
    )

    os.chdir(cwd)


if __name__ == "__main__":
    main()
