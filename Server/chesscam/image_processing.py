from typing import Optional, Tuple

import cv2
import numpy as np
from functools import partial


def standardize_position(frame: np.ndarray, debug: str = '') \
        -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Transform the input frame in a way that maps the board fields to standard locations

    This tries to detect the positioning markers in the frame. In case the expected pattern of markers is not found,
    returns `None`. If the positioning markers are detected, returns a frame in which the board fields appear in
    the always same location, in the always same orientation.

    Args:
        frame: The input image
        debug: (optional) A string denoting which debug modes should be activated (separated by '+').
            Supported modes are: 'histogram', 'binarization', 'contours' and 'print' (so e.g. 'histogram+contours'
            would also be accepted). Default: ''.

    Returns:
        proc_frame: The transformed image in case of successful position detection or `None` in case of no success.
        source_coords: The alignment coordinates in the original image or `None` in case of no success.
        target_coords: The alignment coordinates in the processed image or `None` in case of no success.
    """
    debug_modes = debug.split('+')

    # Image preprocessing
    # -------------------
    # Convert to grayscale
    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to remove noise
    # proc_frame = cv2.GaussianBlur(proc_frame, ksize=(3, 3), sigmaX=0)
    # Alternative: Erode and dilate to remove noise
    # proc_frame = 255 - proc_frame
    # kernel = np.ones((2, 2), np.uint8)
    # proc_frame = cv2.erode(proc_frame, kernel, iterations=2)
    # proc_frame = cv2.dilate(proc_frame, kernel, iterations=2)
    # proc_frame = 255 - proc_frame
    # Alternative 2: Non-local means de-noising (runs rather slowly)
    # proc_frame = cv2.fastNlMeansDenoising(proc_frame)
    # Alternative 3: Median blur
    # proc_frame = cv2.medianBlur(proc_frame, ksize=3)
    # Alternative 4: Bilateral filter
    proc_frame = cv2.bilateralFilter(proc_frame, 5, 25, 20)

    if debug:
        preproc_frame = proc_frame.copy()  # save for later reference in debugging

    # ---- Binarization (we want an image that only contains full black or white)
    # adaptive binarization thresholding, using pixel neighborhood for threshold calculation
    proc_frame = binarize_adaptive(proc_frame)

    # proc_frame = 255 - proc_frame
    # kernel = np.ones((2, 2), np.uint8)
    # proc_frame = cv2.erode(proc_frame, kernel, iterations=1)
    # proc_frame = cv2.dilate(proc_frame, kernel, iterations=1)
    # proc_frame = 255 - proc_frame

    if debug:
        bin_frame = proc_frame.copy()  # save for later reference in debugging

    # ---- Find the position markers
    markers_found = True
    # find contours of black shapes in the preprocessed image
    # first, invert image (findContours() returns contours of white objects on black background)
    proc_frame = 255 - proc_frame
    # find the hierarchy of nested contours
    contours, hierarchy = cv2.findContours(proc_frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # harmonize the output format - in case of no contours, `hierarchy` is `None`. For further processing,
    # we make sure it is an empty list
    if hierarchy is None:
        hierarchy = []
    else:
        # if there are contours found, the hierarchy list is nested in an outside list, that we squeeze away here
        hierarchy = hierarchy[0]

    # analyze the found contours and try to identify the position markers
    # 1. the position markers have exactly 2 child contours on 2 levels with no siblings
    matches_marker_hierarchy = [has_marker_hierarchy(h, hierarchy=hierarchy) for h in hierarchy]
    marker_indices = np.where(matches_marker_hierarchy)[0]
    if len(marker_indices) < 4:
        markers_found = False
        if 'print' in debug_modes:
            print('Too few marker hierarchies found')
        if not debug:
            return None, None, None

    # 2. the marker contours should be approximate quadrilaterals
    approxes = [cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.05, True) for i in marker_indices]
    corner_nums = np.array(list(map(len, approxes)))
    corner_number_match_indices = np.where(corner_nums == 4)[0]  # just keep the shapes with four vertices
    # apply the four-vertex-mask to the marker indices and the approximate polygons
    marker_indices = marker_indices[corner_number_match_indices]
    approxes = [approxes[i] for i in corner_number_match_indices]
    if len(marker_indices) < 4:
        markers_found = False
        if 'print' in debug_modes:
            print('Number of vertices not satisfied')
        if not debug:
            return None, None, None

    # 3. the shape defined by all position markers has to be convex from any perspective
    marker_centroids = np.array([np.mean(app.squeeze(), axis=0).astype(np.int32) for app in approxes])
    if len(marker_centroids) > 0:
        convex_hull_indices = cv2.convexHull(marker_centroids, clockwise=True, returnPoints=False).reshape((-1))
    else:
        convex_hull_indices = np.array([])

    if len(convex_hull_indices) != len(marker_indices):
        markers_found = False
        if 'print' in debug_modes:
            print('Shape is not convex.')
        if not debug:
            return None, None, None

    if 'print' in debug_modes:
        print(f'{len(marker_indices)} position marker candidates found', flush=True)

    # if the number of found position markers is not right, we can abort
    if len(marker_indices) != 4:
        markers_found = False
        if not debug:
            return None, None, None

    marker_contours = [contours[i] for i in marker_indices]
    other_contours = [contours[i] for i in range(len(contours)) if i not in marker_indices]

    # ---- Transform the frame into standard form
    source_coords, target_coords = None, None
    if markers_found:
        # find the top left marker due to its smaller inner square
        # Note: It would be more stable against extreme perspective distortions to do this step
        # after a first transformation
        # without sorting, for fairer size comparison (close vs. far objects).
        # Maybe implement by rotation after transformation.
        innermost_contours = [contours[hierarchy[hierarchy[i][2]][2]] for i in marker_indices]
        innermost_peris = [cv2.arcLength(c, True) for c in innermost_contours]
        topleft_index = np.argmin(innermost_peris)
        try:
            hull_topleft = np.argwhere(convex_hull_indices == topleft_index).squeeze()

        except:
            pass
        print(marker_centroids, convex_hull_indices, topleft_index, hull_topleft)

        # Note: Instead of the centroids, it should be more stable to use e.g. the outermost vertices of each marker
        # find the centroid of all markers, as a reference for vertex distance measurement
        board_centroid = np.mean(marker_centroids, axis=0)
        outer_vertices = np.array([vertices[np.argmax(distance(vertices, board_centroid))] for vertices in approxes])
        # inner_hull = cv2.convexHull(outer_vertices, clockwise=True, returnPoints=True).squeeze()

        # source_coords = np.array([marker_centroids[convex_hull_indices[i % 4]]
        #                           for i in range(hull_topleft, hull_topleft + 4)]).astype(np.float32)
        # source_coords = inner_hull.astype(np.float32)
        source_coords = np.array([outer_vertices[convex_hull_indices[i % 4]]
                                  for i in range(hull_topleft, hull_topleft + 4)]).astype(np.float32)
        target_coords = get_target_coords()
        proc_frame = transform_quadrilateral(frame, source_coords, target_coords)

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

    return proc_frame, source_coords, target_coords


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
    # min_dim = np.min(frame.shape)
    # block_size = min_dim // 10
    # block_size += block_size % 2 + 1
    block_size = 51
    proc_frame = cv2.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=cv2.THRESH_BINARY, blockSize=block_size, C=block_size / 2. - 3)

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


def get_target_coords(target_img_w_h=(500, 500), padding: int = 5):
    w, h = target_img_w_h

    target_coords = np.array([
        [padding, padding],
        [padding, h - padding],
        [w - padding, h - padding],
        [w - padding, padding]
    ], dtype=np.float32)

    return target_coords


def get_board_parameters(target_img_w_h=(500, 500), padding: int = 5) -> Tuple[np.ndarray, int, int]:
    """Compute the offset of the board and single field dimensions

    Args:
        target_img_w_h:
        padding:

    Returns:
        origin: 2-array of integer pixel coordinates of upper left board corner in target image
        field_width: The width of a single chess field in pixels
        field_height: The height of a single chess field in pixels
    """
    w, h = target_img_w_h

    return None


def transform_quadrilateral(frame: np.ndarray, source_coords: np.ndarray, target_coords: np.ndarray,
                            target_img_w_h=(500, 500)) -> np.ndarray:
    transform_matrix = cv2.getPerspectiveTransform(source_coords, target_coords)
    proc_frame = cv2.warpPerspective(frame, transform_matrix, target_img_w_h)

    return proc_frame


if __name__ == '__main__':
    test_mode = 'from_stream'

    if test_mode == 'from_file':
        input_img = cv2.imread('tests/test_image_processing/resources/fotos/valid_dark_corner.jpg')
        stand_img, _, _ = standardize_position(input_img, debug='')

        # resize to fixed height
        scale_fac = 800 / input_img.shape[0]
        input_img = cv2.resize(input_img, (int(scale_fac * input_img.shape[1]), int(scale_fac * input_img.shape[0])))
        cv2.imshow('Input image', input_img)

        if stand_img is not None:
            cv2.imshow('Processed image', stand_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif test_mode == 'from_stream':
        capture = cv2.VideoCapture(0)  # internal cam
        # capture = cv2.VideoCapture('http://192.168.2.117:8080/video')  # phone cam

        while True:
            grabbed, frame = capture.read()
            if cv2.waitKey(1) == ord("q"):
                break

            if not grabbed:
                continue

            # downscale if the image is too big for efficient processing
            max_dim = np.max(frame.shape[:2])
            if max_dim > 800:
                scale_factor = 800 / max_dim
                frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

            proc_frame, _, _ = standardize_position(frame, debug='')
            # proc_frame, _, _ = standardize_position(frame, debug='histogram')

            if proc_frame is not None:
                cv2.imshow('Processed', proc_frame)
            else:
                cv2.imshow('Processed', frame)

            # time.sleep(0.1)

        capture.release()
        cv2.destroyAllWindows()

