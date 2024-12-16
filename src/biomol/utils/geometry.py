import numpy as np


def calc_optimal_rotation_matrix_translation_vector(
    *, P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes  the optimal rotation matrix and translation vector to align two sets of points (P -> Q).

    Args:
    P: N X M arrays of points that works as reference points
    Q: N X M arrays of points that will be transformed

    Returns:
    R: rotation matrix
    t: translational vector
    """
    assert (
        P.shape == Q.shape
    ), "Matrix dimensions of two input matrix should be matched."

    R: np.ndarray  # noqa: N806
    t: np.ndarray

    # Centroid
    cent_P = np.mean(P, axis=0, keepdims=True)  # noqa: N806
    cent_Q = np.mean(Q, axis=0, keepdims=True)  # noqa: N806

    # Optimal translation
    t = cent_Q - cent_P

    # Center the points
    p = P - cent_P
    q = Q - cent_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)  # noqa: N806

    # Single Value Decomposition of covariance matrix H
    U, S, Vt = np.linalg.svd(H)  # noqa: N806

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] += -1.0

    # Optimal rotaion
    R = np.dot(Vt.T, U.T)  # noqa: N806
    # # RMSD
    # rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, t


def transform_by_kabsch(*, P: np.ndarray, Q: np.ndarray) -> np.ndarray:  # noqa: N803
    """
    Transform points Q to align to points, P.

    Args:
    P: N X M arrays of points that works as reference points
    Q: N X M arrays of points that will be transformed

    Returns:
    transformed: transformed points
    """
    R, t = calc_optimal_rotation_matrix_translation_vector(P=P, Q=Q)  # noqa: N806
    # Transformed coordinates
    transformed = np.matmul(R, (Q - np.mean(Q, axis=0, keepdims=True).T)) + t

    return transformed
