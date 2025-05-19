"""
RANSAC iterations applied to matches between descriptors to find the transformation that aligns best the keypoints.
"""

import logging

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from functools import reduce
from scipy.spatial.distance import cdist
from shot_fpfh.core import RigidTransform, solver_point_to_point

# setting a seed
rng = np.random.default_rng(seed=72)



def kabsch_alg(P, Q):
    """
    Given two sets of points P and Q (Nx3), computes the optimal rotation and translation
    using the Kabsch algorithm to align P to Q.
    """
    assert P.shape == Q.shape

    # 1. Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # 2. Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # 3. Compute covariance matrix
    H = P_centered.T @ Q_centered

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 5. Ensure right-handed system (det(R) == 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6. Translation
    t = centroid_Q - R @ centroid_P

    return R, t

def ransac_on_matches(
    scan_descriptors_indices: np.ndarray[np.int32],
    ref_descriptors_indices: np.ndarray[np.int32],
    scan_keypoints: npt.NDArray[np.float64],
    ref_keypoints: npt.NDArray[np.float64],
    n_draws: int = 10000,
    draw_size: int = 4,
    distance_threshold: float = 1,
    verbose: bool = False,
    disable_progress_bar: bool = False,
) -> tuple[float, RigidTransform]:
    """
    Matching strategy that establishes point-to-point correspondences between descriptors and performs RANSAC-type
    iterations to find the best rigid transformation between the two point clouds based on random picks of the matches.
    Only works if the biggest cluster of consistent matches (matches that can all be laid on top of one another by a
    common rigid transform) contains good matches.

    Returns:
        Rotation and translation of the rigid transform to perform on the point cloud.
    """
    best_n_inliers: int | None = None
    best_transform: RigidTransform | None = None

    for _ in (
        pbar := tqdm(
            range(n_draws),
            desc="RANSAC",
            total=n_draws,
            disable=disable_progress_bar,
            delay=0.5,
        )
    ):
        try:
            draw = rng.choice(
                scan_descriptors_indices.shape[0],
                draw_size,
                replace=False,
                shuffle=False,
            )
            transformation = solver_point_to_point(
                scan_keypoints[scan_descriptors_indices[draw]],
                ref_keypoints[ref_descriptors_indices[draw]],
            )
            n_inliers = (
                np.linalg.norm(
                    transformation[scan_keypoints[scan_descriptors_indices]]
                    - ref_keypoints[ref_descriptors_indices],
                    axis=1,
                )
                <= distance_threshold
            ).sum()
            if best_n_inliers is None or n_inliers > best_n_inliers:
                if verbose:
                    logging.info(
                        f"Updating best n_inliers from {best_n_inliers} to {n_inliers}"
                    )
                best_n_inliers = n_inliers
                best_transform = transformation
            pbar.set_description(f"RANSAC - current best n_inliers: {best_n_inliers}")
        except KeyboardInterrupt:
            logging.info("RANSAC interrupted by user.")
            break

    best_transform.normalize_rotation()

    return best_n_inliers / scan_descriptors_indices.shape[0], best_transform


def ransac_on_n_best_matches(scan_pc, ref_pc, scan_descriptors, ref_descriptors , correspondences, threshold, iters=10, m=3):
    """
    Runs RANSAC. Takes m points from scan and from correspondences
    finds some point in ref to match with the one chosen.
    If the distances between the points in the scan pointcloud
    approximately matches the ones matchen to them, a transformation is found
    using Kabsch.

    Args:
        scan_pc: pointcloud to be aligned.
        ref_pc: pointcloud aligned with.
        m: number of point found from scan_pc
        correspondeces: index from scan_pc with its closest matches found
            in getnbestcorrespondeces.
        threshold: how close points have to be in order to classify them as inliers.
    """
    k = 0
    best_inliers = 0
    scan_nonzero = [scan_pc[i] for i in range(len(scan_pc)) if np.any(scan_descriptors[i])]
    ref_nonzero = [ref_pc[i] for i in range(len(ref_pc)) if np.any(ref_descriptors[i])]
    partial_scan = np.array([scan_nonzero[correspondences[i][0]] for i in range(len(correspondences))])
    partial_index_ref = reduce(lambda x, y: x + y,
                               [correspondences[i][1] for i in range(len(correspondences))])
    partial_ref = np.array([ref_nonzero[i] for i in partial_index_ref])
    while k < iters:
        idxs = np.random.choice(len(partial_scan), size=m, replace=False)
        scan_smpl = [list(partial_scan[idx]) for idx in idxs]
        #print(correspondences)
        #print([correspondences[i][0] for i in idxs])
        #print([correspondences[i][1] for i in idxs])
        ind = [int(np.random.choice(correspondences[idx][1])) for idx in idxs]
        ref_smpl = [list(ref_nonzero[i]) for i in ind]
        # print("scan"+ str(scan_smpl))
        # print("ref" +str(ref_smpl))
        if valid_matches(scan_smpl, ref_smpl, 0.25):
            #R_est, t_est = kabsch_alg(np.array(scan_smpl), np.array(ref_smpl))
            transformation = solver_point_to_point(
                np.array(scan_smpl),
                np.array(ref_smpl),
            )
            #trans_scan = (R_est @ partial_scan.T).T + t_est
            k = k + 1

            dists = np.array([np.linalg.norm(
                transformation[partial_scan[i]] - partial_ref[np.argmin([
                    np.min(np.linalg.norm(transformation[partial_scan[i]] - ref_nonzero[c], axis=0))
                    for c in correspondences[i][1]
                ])], axis=0) for i in range(len(partial_scan))])

            inliers = np.sum(dists < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                print(best_inliers)
                best_transformation = transformation
                print(transformation)
    best_inliers_ratio = best_inliers / partial_scan.shape[0]
    best_transformation.normalize_rotation()

    return best_inliers_ratio, best_transformation

def valid_matches(scan_samples,ref_samples,threshold) -> bool:
    scan_distance_matrix = cdist(
            scan_samples,
            scan_samples,
            "euclidean"
            )
    ref_distance_matrix = cdist(
            ref_samples,
            ref_samples,
            "euclidean"
            )
    return np.sum(np.absolute(np.subtract(scan_distance_matrix,ref_distance_matrix))) < threshold
