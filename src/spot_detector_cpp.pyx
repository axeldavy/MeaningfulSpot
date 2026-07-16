#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: embedsignature=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: always_allow_keywords=False
#cython: profile=True
#cython: infer_types=False
#cython: initializedcheck=False

"""
This file defines the code that links the C++ implementation
to Python.
"""

from libc.stdint cimport uint8_t, uint16_t, int32_t, int64_t
from libcpp.vector cimport vector
from spot_detector_cpp_interface cimport MaxPool2DExportDispatch, FindSpotReturnBoxes,\
    FindSpotReturnPixelLists, SSpotBox, SPixelCoord,\
    RemoveSmallSets, FindSpotReturnBoxesMaxtree, FindSpotReturnPixelListsMaxtree,\
    RemoveSmallLevelSetsMaxtree, SLevelStat, ComputeDeltaStatistics

import numpy as np

cdef list _vector_index_to_list(vector[size_t]& vec, size_t width):
    """
    Convert a vector of pixel indices (flattened array) to
    a Python list of (y, x) coordinates
    """
    cdef list result = []
    cdef size_t idx, x, y
    for idx in vec:
        y = idx // width
        x = idx % width
        result.append((y, x))
    return result

cdef list _max_pool_filter_8b(image, int32_t kernel_height, int32_t kernel_width):
    """MaxPool effect visualization (8b)"""
    cdef uint8_t[:, ::1] image_view = image
    cdef vector[size_t] result_vector
    cdef bint success
    with nogil:
        success = MaxPool2DExportDispatch(
            result_vector,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            kernel_height,
            kernel_width
        )
    if not success:
        raise ValueError(f"Unsupported kernel value: {kernel_height}, {kernel_width}")

    return _vector_index_to_list(result_vector, image_view.shape[1])

cdef list _max_pool_filter_16b(image, int32_t kernel_height, int32_t kernel_width):
    """MaxPool effect visualization (16b)"""
    cdef uint16_t[:, ::1] image_view = image
    cdef vector[size_t] result_vector
    cdef bint success
    with nogil:
        success = MaxPool2DExportDispatch(
            result_vector,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            kernel_height,
            kernel_width
        )
    if not success:
        raise ValueError(f"Unsupported kernel value: {kernel_height}, {kernel_width}")

    return _vector_index_to_list(result_vector, image_view.shape[1])

cdef list _max_pool_filter_float(image, int32_t kernel_height, int32_t kernel_width):
    """MaxPool effect visualization (float)"""
    cdef float[:, ::1] image_view = image
    cdef vector[size_t] result_vector
    cdef bint success
    with nogil:
        success = MaxPool2DExportDispatch(
            result_vector,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            kernel_height,
            kernel_width
        )
    if not success:
        raise ValueError(f"Unsupported kernel value: {kernel_height}, {kernel_width}")

    return _vector_index_to_list(result_vector, image_view.shape[1])

def max_pool_filter(image, int32_t kernel_height, int32_t kernel_width):
    """Perform a maxpool operation on the target image
    
    Inputs:
        image: image on which to perform the maxpool operation
        kernel_height: height of the maxpool kernel
        kernel_width: width of the maxpool kernel

    Outputs:
        a list of (y, x) positions for the maxes

    Note:
        supported image formats are 8b, 16b and float32. Other formats
            are converted to float32.
    """
    if image.dtype == np.uint8:
        return _max_pool_filter_8b(image, kernel_height, kernel_width)
    elif image.dtype == np.uint16:
        return _max_pool_filter_16b(image, kernel_height, kernel_width)
    else:
        return _max_pool_filter_float(np.asarray(image, dtype=np.float32), kernel_height, kernel_width)

cdef void _detect_spots_as_boxes_internal_8b(vector[SSpotBox]& boxes, image, int max_elems, float noise_level, float epsilon):
    """Specialized detect_spots_as_boxes for 8b inputs"""
    cdef uint8_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnBoxes(
            boxes,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

cdef void _detect_spots_as_boxes_internal_16b(vector[SSpotBox]& boxes, image, int max_elems, float noise_level, float epsilon):
    """Specialized detect_spots_as_boxes for 16b inputs"""
    cdef uint16_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnBoxes(
            boxes,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

cdef void _detect_spots_as_boxes_internal_float(vector[SSpotBox]& boxes, image, int max_elems, float noise_level, float epsilon):
    """Specialized detect_spots_as_boxes for float inputs"""
    cdef float[:, ::1] image_view = image
    with nogil:
        FindSpotReturnBoxes(
            boxes,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

def detect_spots_as_boxes(image, max_elems=15, noise_level=4., epsilon=1.):
    """
    Detect spots (local level set search implementation).

    The output corresponds to a list of detected boxes.

    Inputs:
        image: Input image. Supported formats are 8b, 16b and float32.
            Other formats are converted to float32.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.
        noise_level: Used in the method threshold. Assumed image noise level.
        epsilon: Used in the method threshold. Target theorical number of
            false detections on a random image of the same size.

    Outputs:
        A list of tuples (one tuple per detected box). The tuples contain
            (y1, x1, y2, x2, size, score), where size if the number of pixels
            in the detected level set, and score is the log(NFA) value that was
            used in the threshold (lower means more contrasted spot)
    """
    cdef vector[SSpotBox] boxes

    # Dispatch depending on input type
    if image.dtype == np.uint8:
        _detect_spots_as_boxes_internal_8b(boxes, image, max_elems, noise_level, epsilon)
    elif image.dtype == np.uint16:
        _detect_spots_as_boxes_internal_16b(boxes, image, max_elems, noise_level, epsilon)
    else:
        _detect_spots_as_boxes_internal_float(boxes, np.asarray(image, dtype=np.float32), max_elems, noise_level, epsilon)

    # Convert the CPP structure to a Python list
    result = []
    cdef SSpotBox box 
    for box in boxes:
        result.append((
            box.m_Y,
            box.m_X,
            box.m_Y + box.m_Height,
            box.m_X + box.m_Width,
            box.m_NumPixels,
            box.m_Score
        ))
    return result

cdef void _detect_spots_as_boxes_maxtree_internal_8b(vector[SSpotBox]& boxes, image, int max_elems, float noise_level, float epsilon):
    """Specialized detect_spots_as_boxes_maxtree for 8b inputs"""
    cdef uint8_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnBoxesMaxtree[uint8_t](
            boxes,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

cdef void _detect_spots_as_boxes_maxtree_internal_16b(vector[SSpotBox]& boxes, image, int max_elems, float noise_level, float epsilon):
    """Specialized detect_spots_as_boxes_maxtree for 16b inputs"""
    cdef uint16_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnBoxesMaxtree[uint16_t](
            boxes,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

def detect_spots_as_boxes_maxtree(image, max_elems=15, noise_level=4., epsilon=1.):
    """
    Detect spots (Max-Tree implementation).

    The output corresponds to a list of detected boxes.

    Inputs:
        image: Input image. Supported formats are unsigned 8b, 16b.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.
        noise_level: Used in the method threshold. Assumed image noise level.
        epsilon: Used in the method threshold. Target theorical number of
            false detections on a random image of the same size.

    Outputs:
        A list of tuples (one tuple per detected box). The tuples contain
            (y1, x1, y2, x2, size, score), where size if the number of pixels
            in the detected level set, and score is the log(NFA) value that was
            used in the threshold (lower means more contrasted spot)
    """
    cdef vector[SSpotBox] boxes

    # Dispatch depending on input type
    if image.dtype == np.uint8:
        _detect_spots_as_boxes_maxtree_internal_8b(boxes, image, max_elems, noise_level, epsilon)
    elif image.dtype == np.uint16:
        _detect_spots_as_boxes_maxtree_internal_16b(boxes, image, max_elems, noise_level, epsilon)
    else:
        raise TypeError("Maxtree does not support float (pylene requires unsigned integer <= 16 bits)")

    # Convert the CPP structure to a Python list
    result = []
    cdef SSpotBox box 
    for box in boxes:
        result.append((
            box.m_Y,
            box.m_X,
            box.m_Y + box.m_Height,
            box.m_X + box.m_Width,
            box.m_NumPixels,
            box.m_Score
        ))
    return result

cdef void _detect_spots_as_mask_internal_8b(vector[SPixelCoord]& pixels,
                                            vector[size_t] &ll_starts,
                                            vector[double] &log_nfa,
                                            image,
                                            int max_elems,
                                            float noise_level,
                                            float epsilon):
    """Specialized detect_spots_mask for 8b inputs"""
    cdef uint8_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnPixelLists(
            pixels,
            ll_starts,
            log_nfa,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

cdef void _detect_spots_as_mask_internal_16b(vector[SPixelCoord]& pixels,
                                             vector[size_t] &ll_starts,
                                             vector[double] &log_nfa,
                                             image,
                                             int max_elems,
                                             float noise_level,
                                             float epsilon):
    """Specialized detect_spots_mask for 16b inputs"""
    cdef uint16_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnPixelLists(
            pixels,
            ll_starts,
            log_nfa,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

cdef void _detect_spots_as_mask_internal_float(vector[SPixelCoord]& pixels,
                                               vector[size_t] &ll_starts,
                                               vector[double] &log_nfa,
                                               image,
                                               int max_elems,
                                               float noise_level,
                                               float epsilon):
    """Specialized detect_spots_mask for float inputs"""
    cdef float[:, ::1] image_view = image
    with nogil:
        FindSpotReturnPixelLists(
            pixels,
            ll_starts,
            log_nfa,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )
    

def detect_spots_as_mask(image, max_elems=15, noise_level=4., epsilon=1.):
    """
    Detect spots (local level set search implementation).

    The output corresponds to a detected mask. All pixels part of a
    detection are colored in the mask. The main use of this method
    is for educational purposes, as it enables to visualize the boundaries
    of the detections. 

    Inputs:
        image: Input image. Supported formats are 8b, 16b and float32.
            Other formats are converted to float32.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.
        noise_level: Used in the method threshold. Assumed image noise level.
        epsilon: Used in the method threshold. Target theorical number of
            false detections on a random image of the same size.

    Outputs:
        A list of tuples (one tuple per detected box). The tuples contain
            (y1, x1, y2, x2, size, score), where size if the number of pixels
            in the detected level set, and score is the log(NFA) value that was
            used in the threshold (lower means more contrasted spot)
    """
    cdef vector[SPixelCoord] pixels
    cdef vector[size_t] ll_starts
    cdef vector[double] log_nfa

    # Dispatch depending on the image type
    if image.dtype == np.uint8:
        _detect_spots_as_mask_internal_8b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    elif image.dtype == np.uint16:
        _detect_spots_as_mask_internal_16b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    else:
        _detect_spots_as_mask_internal_float(pixels, ll_starts, log_nfa, np.asarray(image, dtype=np.float32), max_elems, noise_level, epsilon)

    # Color each pixel part of a detection on the output mask
    result = np.zeros(image.shape, dtype=np.uint8)
    cdef uint8_t[:,::1] result_view = result
    cdef SPixelCoord coord 
    for coord in pixels:
        result_view[coord.m_Y, coord.m_X] = 255
    return result

def detect_spots_detailed(image, max_elems=15, noise_level=4., epsilon=1.):
    """
    Detect spots (local level set search implementation).

    The output corresponds to a list of pixels for each detection.

    Inputs:
        image: Input image. Supported formats are 8b, 16b and float32.
            Other formats are converted to float32.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.
        noise_level: Used in the method threshold. Assumed image noise level.
        epsilon: Used in the method threshold. Target theorical number of
            false detections on a random image of the same size.

    Outputs:
        A list of lists (one list per detected box). The list contains
            (y, x, score), where y, x is a pixel part of the detection
            and score is the log(NFA) value that was used in the
            threshold (lower means more contrasted spot). The same score
            is contained for all pixels of a given detection.
    """
    cdef vector[SPixelCoord] pixels
    cdef vector[size_t] ll_starts
    cdef vector[double] log_nfa

    # Dispatch depending on the image type
    if image.dtype == np.uint8:
        _detect_spots_as_mask_internal_8b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    elif image.dtype == np.uint16:
        _detect_spots_as_mask_internal_16b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    else:
        _detect_spots_as_mask_internal_float(pixels, ll_starts, log_nfa, np.asarray(image, dtype=np.float32), max_elems, noise_level, epsilon)
    assert pixels.size() == log_nfa.size()

    # Convert the detections to lists of pixels
    cdef list results = []
    cdef list result
    cdef size_t i, start, stop
    for i in range(ll_starts.size()):
        start = ll_starts[i]
        if i + 1 < ll_starts.size():
            stop = ll_starts[i + 1]
        else:
            stop = pixels.size()
        result = []
        for j in range(start, stop):
            result.append((
                pixels[j].m_Y,
                pixels[j].m_X,
                log_nfa[i]
            ))
        results.append(result)
    return results

cdef void _detect_spots_as_mask_maxtree_internal_8b(
    vector[SPixelCoord]& pixels,
    vector[size_t] &ll_starts,
    vector[double] &log_nfa,
    image,
    int max_elems,
    float noise_level,
    float epsilon
):
    """Specialized implementation of detect_spots_as_mask_maxtree (8b)"""
    cdef uint8_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnPixelListsMaxtree[uint8_t](
            pixels,
            ll_starts,
            log_nfa,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

cdef void _detect_spots_as_mask_maxtree_internal_16b(
    vector[SPixelCoord]& pixels,
    vector[size_t] &ll_starts,
    vector[double] &log_nfa,
    image,
    int max_elems,
    float noise_level,
    float epsilon
):
    """Specialized implementation of detect_spots_as_mask_maxtree (16b)"""
    cdef uint16_t[:, ::1] image_view = image
    with nogil:
        FindSpotReturnPixelListsMaxtree[uint16_t](
            pixels,
            ll_starts,
            log_nfa,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems,
            epsilon / (image_view.shape[0] * image_view.shape[1]),
            noise_level
        )

def detect_spots_as_mask_maxtree(image, max_elems=15, noise_level=4., epsilon=1.):
    """
    Detect spots (Max-Tree implementation).

    The output corresponds to a detected mask. All pixels part of a
    detection are colored in the mask. The main use of this method
    is for educational purposes, as it enables to visualize the boundaries
    of the detections. 

    Inputs:
        image: Input image. Supported formats are 8b and 16b.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.
        noise_level: Used in the method threshold. Assumed image noise level.
        epsilon: Used in the method threshold. Target theorical number of
            false detections on a random image of the same size.

    Outputs:
        A list of tuples (one tuple per detected box). The tuples contain
            (y1, x1, y2, x2, size, score), where size if the number of pixels
            in the detected level set, and score is the log(NFA) value that was
            used in the threshold (lower means more contrasted spot)
    """
    cdef vector[SPixelCoord] pixels
    cdef vector[size_t] ll_starts
    cdef vector[double] log_nfa

    # Dispatch depending on the image type
    if image.dtype == np.uint8:
        _detect_spots_as_mask_maxtree_internal_8b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    elif image.dtype == np.uint16:
        _detect_spots_as_mask_maxtree_internal_16b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    else:
        raise TypeError("Maxtree does not support float (pylene requires unsigned integer <=16 bits)")

    # Color for each detection its pixels in the mask
    result = np.zeros(image.shape, dtype=np.uint8)
    cdef uint8_t[:,::1] result_view = result
    cdef SPixelCoord coord 
    for coord in pixels:
        result_view[coord.m_Y, coord.m_X] = 255
    return result

def detect_spots_detailed_maxtree(image, max_elems=15, noise_level=4., epsilon=1.):
    """
    Detect spots (Max-Tree implementation).

    The output corresponds to a list of pixels for each detection.

    Inputs:
        image: Input image. Supported formats are 8b and 16b.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.
        noise_level: Used in the method threshold. Assumed image noise level.
        epsilon: Used in the method threshold. Target theorical number of
            false detections on a random image of the same size.

    Outputs:
        A list of lists (one list per detected box). The list contains
            (y, x, score), where y, x is a pixel part of the detection
            and score is the log(NFA) value that was used in the
            threshold (lower means more contrasted spot). The same score
            is contained for all pixels of a given detection.
    """
    cdef vector[SPixelCoord] pixels
    cdef vector[size_t] ll_starts
    cdef vector[double] log_nfa

    # Dispatch depending on image type
    if image.dtype == np.uint8:
        _detect_spots_as_mask_maxtree_internal_8b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    elif image.dtype == np.uint16:
        _detect_spots_as_mask_maxtree_internal_16b(pixels, ll_starts, log_nfa, image, max_elems, noise_level, epsilon)
    else:
        raise TypeError("Maxtree does not support float (pylene requires unsigned integer <=16 bits)")
    assert pixels.size() == log_nfa.size()

    # Convert the detections to lists of pixels
    cdef list results = []
    cdef list result
    cdef size_t i, start, stop
    for i in range(ll_starts.size()):
        start = ll_starts[i]
        if i + 1 < ll_starts.size():
            stop = ll_starts[i + 1]
        else:
            stop = pixels.size()
        result = []
        for j in range(start, stop):
            result.append((
                pixels[j].m_Y,
                pixels[j].m_X,
                log_nfa[i]
            ))
        results.append(result)
    return results

cdef object _remove_small_sets_8b(image, int max_elems):
    """Specialized implementation of remove_small_sets (8b)"""
    result = np.copy(image) # result must be initialized with a copy of the input image
    cdef uint8_t[:, ::1] image_view = image
    cdef uint8_t[:, ::1] result_view = result
    with nogil:
        RemoveSmallSets(
            &result_view[0, 0],
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems
        )
    return result

cdef object _remove_small_sets_16b(image, int max_elems):
    """Specialized implementation of remove_small_sets (16b)"""
    result = np.copy(image) # result must be initialized with a copy of the input image
    cdef uint16_t[:, ::1] image_view = image
    cdef uint16_t[:, ::1] result_view = result
    with nogil:
        RemoveSmallSets(
            &result_view[0, 0],
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems
        )
    return result

cdef object _remove_small_sets_float(image, int max_elems):
    """Specialized implementation of remove_small_sets (float)"""
    result = np.copy(image) # result must be initialized with a copy of the input image
    cdef float[:, ::1] image_view = image
    cdef float[:, ::1] result_view = result
    with nogil:
        RemoveSmallSets(
            &result_view[0, 0],
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems
        )
    return result

def remove_small_sets(image, max_elems=15):
    """
    Remove small level sets from the image (Local level set search implementation).

    It does flatten pixels that are part of small sets.

    Inputs:
        image: Input image. Supported formats are 8b, 16b and float32.
            Other formats are converted to float32.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.

    Outputs:
        An image where all found level sets matching the size criteria are
            replaced with flat regions of the lowest value in the
            corresponding level set.
    """
    if image.dtype == np.uint8:
        return _remove_small_sets_8b(image, max_elems)
    elif image.dtype == np.uint16:
        return _remove_small_sets_16b(image, max_elems)
    else:
        return _remove_small_sets_float(np.asarray(image, dtype=np.float32), max_elems)

cdef object _remove_small_sets_maxtree_8b(image, int max_elems):
    """Specialized implementation of remove_small_sets_maxtree (8b)"""
    cdef uint8_t[:, ::1] image_view = image
    result = np.empty_like(image)
    cdef uint8_t[:, ::1] result_view = result
    with nogil:
        RemoveSmallLevelSetsMaxtree[uint8_t](
            &result_view[0, 0],
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems
        )
    return result

cdef object _remove_small_sets_maxtree_16b(image, int max_elems):
    """Specialized implementation of remove_small_sets_maxtree (16b)"""
    cdef uint16_t[:, ::1] image_view = image
    result = np.empty_like(image)
    cdef uint16_t[:, ::1] result_view = result
    with nogil:
        RemoveSmallLevelSetsMaxtree[uint16_t](
            &result_view[0, 0],
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1],
            max_elems
        )
    return result

def remove_small_sets_maxtree(image, max_elems=15):
    """
    Remove small level sets from the image (Max-Tree implementation).

    It does flatten pixels that are part of small sets.

    Inputs:
        image: Input image. Supported formats are 8b and 16b.
        max_elems: The maximum number of pixels allowed in a level set
            during the search.

    Outputs:
        An image where all found level sets matching the size criteria are
            replaced with flat regions of the lowest value in the
            corresponding level set.
    """
    if image.dtype == np.uint8:
        return _remove_small_sets_maxtree_8b(image, max_elems)
    elif image.dtype == np.uint16:
        return _remove_small_sets_maxtree_16b(image, max_elems)
    else:
        raise NotImplementedError(f"Unsupported dtype: {image.dtype}")

cdef object _get_level_stats_8b(image):
    """Specialized implementation of get_level_stats (8b)"""
    cdef uint8_t[:, ::1] image_view = image
    cdef vector[SLevelStat[uint8_t]] level_stats
    with nogil:
        ComputeDeltaStatistics(
            level_stats,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1]
        )
    result_x = np.empty(shape=(level_stats.size(),), dtype=np.float32)
    result_y = np.empty(shape=(level_stats.size(),), dtype=np.float32)
    cdef float[::1] result_x_view = result_x
    cdef float[::1] result_y_view = result_y
    cdef SLevelStat[uint8_t] level_stat
    cdef int64_t i
    for i in range(level_stats.size()):
        level_stat = level_stats[i]
        delta_level = level_stat.m_Level - level_stat.m_ParentLevel
        num_pixels = level_stat.m_NumPixels
        result_x_view[i] = num_pixels
        result_y_view[i] = delta_level
    return result_x, result_y

cdef object _get_level_stats_16b(image):
    """Specialized implementation of get_level_stats (16b)"""
    cdef uint16_t[:, ::1] image_view = image
    cdef vector[SLevelStat[uint16_t]] level_stats
    with nogil:
        ComputeDeltaStatistics(
            level_stats,
            &image_view[0, 0],
            image_view.shape[0],
            image_view.shape[1]
        )
    result_x = np.empty(shape=(level_stats.size(),), dtype=np.float32)
    result_y = np.empty(shape=(level_stats.size(),), dtype=np.float32)
    cdef float[::1] result_x_view = result_x
    cdef float[::1] result_y_view = result_y
    cdef SLevelStat[uint16_t] level_stat
    cdef int64_t i
    for i in range(level_stats.size()):
        level_stat = level_stats[i]
        delta_level = level_stat.m_Level - level_stat.m_ParentLevel
        num_pixels = level_stat.m_NumPixels
        result_x_view[i] = num_pixels
        result_y_view[i] = delta_level
    return result_x, result_y

def get_level_stats(image):
    """
    Get statistics about the level sets of an image.

    The goal of this function is to visualize the distribution
    of deltas between level sets for a given image.

    Inputs:
        image: Input image. Supported formats are 8b, 16b.

    Outputs:
        A tuple containing:
            A list of number of pixels (size of a level set)
            A list of intensity deltas with their parent level set 
    """
    if image.dtype == np.uint8:
        return _get_level_stats_8b(image)
    elif image.dtype == np.uint16:
        return _get_level_stats_16b(image)
    else:
        raise NotImplementedError(f"Unsupported dtype: {image.dtype}")