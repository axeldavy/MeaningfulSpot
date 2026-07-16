// This file implements the header for the spot detector using
// a Max-Tree.

#pragma once

#include <cstdint>
#include <vector>
#include "spot_detector.hpp"

/// <summary>
/// Find spots (bright regions) in an image using a maxtree-based approach.
/// This variant builds a complete max-tree of the image, then traverses it
/// to find meaningful level sets, providing a comparison against the
/// maxpool + local flooding approach.
///
/// The function builds a max-tree representing all connected components
/// at all intensity levels, then evaluates each node for meaningfulness
/// using the NFA criterion.
///
/// </summary>
/// <typeparam name="T">Type of pixel data (uint8_t, uint16_t, or float)</typeparam>
/// <param name="o_Boxes">
/// Output vector that will contain detected spot bounding boxes with scores
/// </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_MaxElements">
/// Maximum number of pixels allowed in detected spots (inclusive)
/// </param>
/// <param name="p_EpsilonThreshold">
/// NFA threshold for detection (typical value: 1.0)
/// </param>
/// <param name="p_NoiseLevel">
/// Expected standard deviation of noise in the image
/// </param>
/// <note> Uses 4-connectivity for the max-tree construction </note>
template <typename T>
void FindSpotReturnBoxesMaxtree(
    std::vector<SSpotBox>& o_Boxes,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

/// <summary>
/// Find spots (bright regions) in an image using a maxtree-based approach
/// and return pixel lists for each detection.
///
/// This variant builds a complete max-tree of the image, then traverses it
/// to find meaningful level sets.
///
/// </summary>
/// <typeparam name="T">Type of pixel data (uint8_t, uint16_t, or float)</typeparam>
/// <param name="o_Pixels">
/// Output vector containing all pixels from all detections
/// </param>
/// <param name="o_DetectionPixelStartIdx">
/// Output vector containing start indices in o_Pixels for each detection
/// </param>
/// <param name="o_DetectionLogNFA">
/// Output vector containing log(NFA) scores for each detection
/// </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_MaxElements">
/// Maximum number of pixels allowed in detected spots (inclusive)
/// </param>
/// <param name="p_EpsilonThreshold">
/// NFA threshold for detection (typical value: 1.0)
/// </param>
/// <param name="p_NoiseLevel">
/// Expected standard deviation of noise in the image
/// </param>
/// <note> Uses 4-connectivity for the max-tree construction </note>
template <typename T>
void FindSpotReturnPixelListsMaxtree(
    std::vector<SPixelCoord>& o_Pixels,
    std::vector<std::size_t>& o_DetectionPixelStartIdx,
    std::vector<double>& o_DetectionLogNFA,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

/// <summary>
/// Remove small level sets using a maxtree-based approach.
/// This function builds a max-tree and removes all nodes (connected components)
/// that have subtree size <= p_MaxElements by setting their pixels to the
/// parent level value.
///
/// This is more efficient than the flooding-based approach when processing
/// the entire image, as it builds the tree once and processes all small
/// regions in a single pass.
///
/// </summary>
/// <typeparam name="T">Type of pixel data (uint8_t or uint16_t)</typeparam>
/// <param name="o_Dst"> Output 2D image (C contiguous) </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_MaxElements">
/// Maximum number of elements for level sets to be removed (inclusive)
/// </param>
/// <note> Uses 4-connectivity for the max-tree construction </note>
/// <note> o_Dst and p_Src should not overlap (o_Dst will be overwritten) </note>
template <typename T>
void RemoveSmallLevelSetsMaxtree(
    T* o_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);

template <typename T>
struct SLevelStat
{
    T m_Level;
    T m_ParentLevel;
    std::size_t m_NumPixels;
};

template <typename T>
void ComputeDeltaStatistics(
    std::vector<SLevelStat<T>>& r_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width
);