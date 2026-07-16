// This file implements the header of the spot detector
// using a local level set search.

#pragma once

#include <cstdint>
#include <vector>

struct SSpotBox
{
    std::size_t m_Y;
    std::size_t m_X;
    std::size_t m_Height;
    std::size_t m_Width;
    std::size_t m_NumPixels;
    double m_Score;
};

struct SPixelCoord
{
    std::size_t m_Y;
    std::size_t m_X;
};

/// <summary>
/// Find spots (bright regions) in an image by detecting meaningful level sets
/// at local maxima positions and return bounding boxes.
///
/// The function performs a 5x5 max pooling to find local maxima, then computes
/// level sets at each maximum and evaluates their meaningfulness using the
/// Number of False Alarms (NFA) criterion.
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
/// <note> Uses 4-connectivity (cross '+' pattern) for level sets </note>
template <typename T>
void FindSpotReturnBoxes(
    std::vector<SSpotBox>& o_Boxes,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

/// <summary>
/// Find spots (bright regions) in an image by detecting meaningful level sets
/// at local maxima positions and return pixel lists for each detection.
///
/// The function performs a 5x5 max pooling to find local maxima, then computes
/// level sets at each maximum and evaluates their meaningfulness using the
/// Number of False Alarms (NFA) criterion.
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
/// <note> Uses 4-connectivity (cross '+' pattern) for level sets </note>
/// <note>
/// For detection i, pixels are stored in o_Pixels from index
/// o_DetectionPixelStartIdx[i] to o_DetectionPixelStartIdx[i+1]-1
/// </note>
template <typename T>
void FindSpotReturnPixelLists(
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
/// Remove level sets of size <= p_MaxElements.
///
/// The function performs a 5x5 max pooling to find local maxima, then
/// removes all level sets at each maximum that have size <= p_MaxElements
/// by setting their pixels to the next level value.
///
/// </summary>
/// <typeparam name="T">Type of pixel data (uint8_t, uint16_t, or float)</typeparam>
/// <param name="o_Dst">
/// Output 2D image (C contiguous), will be initialized with a copy of p_Src
/// </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_MaxElements">
/// Maximum size of level sets to remove (inclusive)
/// </param>
/// <note> Uses 4-connectivity (cross '+' pattern) for level sets </note>
/// <note> p_Dst must be pre-allocated to p_Height * p_Width * sizeof(T) </note>
/// <note> p_Dst and p_Src must not overlap </note>
template <typename T>
void RemoveSmallSets(
    T* o_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);

struct SDebugCounter
{
    std::size_t m_NumMerge;
    std::size_t m_NumLevel;
    std::size_t m_NumRetained;
    void reset()
    {
        m_NumMerge = 0;
        m_NumLevel = 0;
        m_NumRetained = 0;
    }
};

inline SDebugCounter& getSDebugCounter()
{
    static SDebugCounter l_Counter{};
    return l_Counter;
};