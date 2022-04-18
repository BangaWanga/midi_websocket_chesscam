from typing import Optional

import cv2
import numpy as np
from functools import partial


def standardize_position(frame: np.ndarray, debug: str = '') -> Optional[np.ndarray]:
    """Transform the input frame in a way that maps the board fields to standard locations

    This tries to detect the positioning markers in the frame. In case the expected pattern of markers is not found,
    returns `None`. If the positioning markers are detected, returns a frame in which the board fields appear in
    the always same location, in the always same orientation.

    Args:
        frame: The input image
        debug: (optional) A string denoting which debug modes should be activated (separated by '+').
            Supported modes are: 'histogram' and 'contours' (so e.g. 'histogram+contours' would also be accepted).
            Default: ''. Note that some debug modes might return early and thus might prevent others to be applied.

    Returns:
        The transformed image in case of successful position detection or `None` in case of no success.
    """
    debug_modes = debug.split('+')

    # Image preprocessing
    # -------------------
    # Convert to grayscale
    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur to remove noise
    proc_frame = cv2.GaussianBlur(proc_frame, ksize=(5, 5), sigmaX=0)
    preproc_frame = proc_frame.copy()  # save for later reference in debugging

    # Binarization (we want an image that only contains full black or white)
    # adaptive binarization thresholding, using pixel neighborhood for threshold calculation
    proc_frame = binarize_adaptive(proc_frame)
    bin_frame = proc_frame.copy()  # save for later reference in debugging

    # ---- Find the position markers
    # find contours of black shapes in the preprocessed image
    # invert image (findContours() returns contours of white objects on black background)
    proc_frame = 255 - proc_frame
    # find the hierarchy of nested contours
    contours, hierarchy = cv2.findContours(proc_frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # analyze the found contours and try to identify the position markers
    # 1. the position markers have exactly 2 child contours on 2 levels with no siblings
    matches_marker_hierarchy = [has_marker_hierarchy(h, hierarchy=hierarchy[0]) for h in hierarchy[0]]
    marker_indices = np.where(matches_marker_hierarchy)[0]
    if not debug and len(marker_indices) < 4:
        return None

    # 2. the marker contours should be approximate quadrilaterals
    approxes = [cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.05, True) for i in marker_indices]
    corner_nums = np.array(list(map(len, approxes)))
    corner_number_match_indices = np.where(corner_nums == 4)[0]
    marker_indices = marker_indices[corner_number_match_indices]
    if not debug and len(marker_indices) < 4:
        return None

    # 3. the quadrilateral defined by position markers has to be convex from any perspective
    marker_centroids = np.array([np.mean(app.squeeze(), axis=0).astype(np.int32) for app in approxes])
    convex_hull_indices = cv2.convexHull(marker_centroids, clockwise=True, returnPoints=False).squeeze()
    if not debug and len(convex_hull_indices) != len(marker_indices):
        return None

    print(f'{len(marker_indices)} position marker candidates found')

    # if the number of found position markers is not right, we can abort
    if not debug and len(marker_indices) != 4:
        return None

    marker_contours = [contours[i] for i in marker_indices]
    other_contours = [contours[i] for i in range(len(contours)) if i not in marker_indices]

    # ---- Transform the frame into standard form
    # find the top left marker due to its smaller inner square
    # Note: It would be more stable against extreme perspective distortions to do this step after a first transformation
    # without sorting, for fairer size comparison (close vs. far objects).
    # Maybe implement by rotation after transformation.
    innermost_contours = [contours[hierarchy[0][hierarchy[0][i][2]][2]] for i in marker_indices]
    innermost_peris = [cv2.arcLength(c, True) for c in innermost_contours]
    topleft_index = np.argmin(innermost_peris)
    hull_topleft = np.where(convex_hull_indices == topleft_index)[0][0]

    # Note: Instead of the centroids, it should be more stable to use e.g. the innermost vertices of each marker
    # find the centroid of all markers, as a reference for vertex distance measurement
    board_centroid = np.mean(marker_centroids, axis=0)
    inner_vertices = np.array([vertices[np.argmin(distance(vertices, board_centroid))] for vertices in approxes])
    # inner_hull = cv2.convexHull(inner_vertices, clockwise=True, returnPoints=True).squeeze()

    # source_points = np.array([marker_centroids[convex_hull_indices[i % 4]]
    #                           for i in range(hull_topleft, hull_topleft + 4)]).astype(np.float32)
    # source_points = inner_hull.astype(np.float32)
    source_points = np.array([inner_vertices[convex_hull_indices[i % 4]]
                              for i in range(hull_topleft, hull_topleft + 4)]).astype(np.float32)
    target_points = np.array([
        [0, 0],
        [0, 500],
        [500, 500],
        [500, 0]
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(source_points, target_points)
    proc_frame = cv2.warpPerspective(frame, transform_matrix, (500, 500))

    # drawing for optional debugging
    draw_frame = frame
    if 'histogram' in debug_modes:
        draw_frame = cv2.cvtColor(preproc_frame, cv2.COLOR_GRAY2BGR)
        proc_frame = draw_histogram(preproc_frame, draw_frame)

    if 'binarization' in debug_modes:
        draw_frame = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)
        proc_frame = draw_frame

    if 'contours' in debug_modes:
        # draw the detected contours on the frame for inspection
        proc_frame = cv2.drawContours(draw_frame, other_contours, -1, (0, 255, 0), 2)
        proc_frame = cv2.drawContours(proc_frame, marker_contours, -1, (255, 0, 0), 2)

    return proc_frame


def weighted_var(xs, weights):
    mean = np.average(xs, weights=weights)
    return np.average((xs - mean) * (xs - mean), weights=weights)


def binarize_otsu_own(frame):
    """Our own implementation of Otsu's method for global binarization"""
    # Analyze histogram to find a good binarization threshold
    hist = cv2.calcHist(images=[proc_frame], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

    hist_x = np.arange(hist.shape[0])
    var_losses = np.array([weighted_var(hist_x[:t], hist[:t, 0] + 1) + weighted_var(hist_x[t + 1:], hist[t + 1:, 0] + 1)
                           for t in range(1, hist_x.shape[0] - 1)])
    thresh_idx = np.argmin(var_losses)
    thresh, proc_frame = cv2.threshold(frame, thresh_idx, maxval=255, type=cv2.THRESH_BINARY)

    return thresh, proc_frame


def binarize_otsu(frame):
    """Otsu's method for global binarization as implemented in cv2"""
    thresh, proc_frame = cv2.threshold(frame, 0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh, proc_frame


def binarize_adaptive(frame):
    """Adaptive binarization thresholding, using pixel neighborhood for threshold calculation"""
    min_dim = np.min(frame.shape)
    block_size = min_dim // 3
    block_size += block_size % 2 + 1
    proc_frame = cv2.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY, blockSize=block_size, C=10)

    return proc_frame


def draw_histogram(frame, draw_frame):
    """Show histogram for debugging"""
    hist = cv2.calcHist(images=[frame], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

    # convert histogram values to x-y-coordinates
    hist_x = np.arange(256).astype(hist.dtype)
    hist_coords = np.vstack((hist_x, hist[:, 0])).T

    # adjust coordinate ranges to image size for nice plotting
    hist_coords[:, 0] = hist_coords[:, 0] * frame.shape[0] / np.max(hist_coords[:, 0])
    hist_coords[:, 1] = hist_coords[:, 1] * 0.8 * frame.shape[1] / np.max(hist_coords[:, 1]) \
                        + 0.1 * frame.shape[1]  # offset at the bottom
    hist_coords[:, 1] = frame.shape[1] - hist_coords[:, 1]  # flip y-axis (y=0 is at top of image)

    # convert coordinates to int32 as expected by cv2.polylines()
    hist_coords = hist_coords.astype(np.int32)

    # draw histogram curve on top of image
    # proc_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    proc_frame = cv2.polylines(draw_frame, [hist_coords], False, (0, 255, 0), thickness=3)

    return proc_frame


def get_contour_child_num(hierarchy_element, hierarchy):
    if hierarchy_element[2] == -1:  # no child remaining
        return 0
    else:
        # get all children siblings
        children = []
        current_child = hierarchy[hierarchy_element[2]]
        children.append(current_child)

        while current_child[0] > -1:
            current_child = hierarchy[current_child[0]]
            children.append(current_child)

        # recurse over children
        child_nums = list(map(partial(get_contour_child_num, hierarchy=hierarchy), children))
        return len(children) + sum(child_nums)


def has_marker_hierarchy(hierarchy_element, hierarchy, level=0):
    """The requirements are:

    - there should be exactly 2 levels of children
    - each child must not have any siblings
    """
    if level > 2:
        return False

    # this condition only applies to children (level > 0): there must be no siblings
    if level > 0 and hierarchy_element[0] > -1:
        return False

    if hierarchy_element[2] == -1:
        if level == 2:
            return True
        else:
            return False
    else:
        return has_marker_hierarchy(hierarchy[hierarchy_element[2]], hierarchy, level=level + 1)


def distance(points, other):
    diffs = points - other
    return np.sqrt(np.sum(diffs * diffs, axis=-1))


if __name__ == '__main__':
    # input_img = cv2.imread('tests/test_image_processing/resources/synthetisch/no_board.jpg')
    # input_img = cv2.imread('tests/test_image_processing/resources/synthetisch/board.png')
    # input_img = cv2.imread('tests/test_image_processing/resources/synthetisch/small_board.png')
    # input_img = cv2.imread('tests/test_image_processing/resources/synthetisch/rotated_board.png')
    # input_img = cv2.imread('tests/test_image_processing/resources/fotos/valid_rotated2.jpg')
    # input_img = cv2.imread('tests/test_image_processing/resources/fotos/valid_half_dark.jpg')
    input_img = cv2.imread('tests/test_image_processing/resources/fotos/valid_dark_corner.jpg')
    # input_img = cv2.imread('tests/test_image_processing/resources/fotos/valid_normal.jpg')
    # stand_img = standardize_position(input_img, debug='contours+binarization')
    stand_img = standardize_position(input_img, debug='')

    # resize to fixed height
    scale_fac = 800 / input_img.shape[0]
    input_img = cv2.resize(input_img, (int(scale_fac * input_img.shape[1]), int(scale_fac * input_img.shape[0])))
    cv2.imshow('Input image', input_img)

    if stand_img is not None:
        # resize to fixed height
        scale_fac = 800 / stand_img.shape[0]
        stand_img = cv2.resize(stand_img, (int(scale_fac * stand_img.shape[1]), int(scale_fac * stand_img.shape[0])))
        cv2.imshow('Processed image', stand_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
