// This file implements the header for the level line search
// used for the implementation that doesn't rely on a maxtree.

#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

template<typename T>
struct SPixelData
{
    std::size_t m_Y;
    std::size_t m_X;
    T m_V;
};


template<bool MAX, typename T>
struct SPixelState
{
    // Indicates whether the pixel has already been part of a detected level set
    bool m_PartOfDetection = false;
    // Level sets of value <= (MAX = True) or >= (MAX = False)
    // are considered already explored and should not be processed again
    T m_MinLevelSetValueExplored = MAX ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();

    bool operator ==(const SPixelState<MAX, T>& other) const
    {
        return m_PartOfDetection == other.m_PartOfDetection &&
               m_MinLevelSetValueExplored == other.m_MinLevelSetValueExplored;
    }
};

template<typename T>
struct SNeigborPixelData
{
    std::size_t m_Y;
    std::size_t m_X;
    T m_V;
    bool m_Merged;
};

template<typename T>
struct SLevelSetInfo
{
    /// <summary> stop index (inclusive) in the SPixelData array </summary>
    std::size_t m_StopIdx;
    /// Level set value (min or max of the items)
    T m_Level;
};


/// <summary>
/// Compute, starting from a given pixel, all level sets
/// of size <= p_MaxElements containing the pixel.
///
/// A level set l is a set of neigboring pixels such that
/// all values are >= l (or <= respectively).
///
/// The function is expected to be called with a pixel that is part
/// of its extremum level set (a local maxima for >= and vice versa).
/// This is not mandatory however.
///
/// </summary>
/// <typeparam name="MAX">True for '>= l', False for '<= l' </typeparam>
/// <typeparam name="T">Type of data</typeparam>
/// <param name="o_LevelSetInfo">
/// Output array that will contain the delimitation of the level
/// sets in the o_PixelData array and the level 'l'.
/// </param>
/// <param name="o_PixelData">
/// Output array that will contain pixel position and values.
/// Only the section covered by o_LevelSetInfo needs to be valid.
/// In particular o_PixelData may contain more than p_MaxElements pixels.
/// </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_StartY"> Y position of the starting pixel </param>
/// <param name="p_StartX"> X position of the starting pixel </param>
/// <param name="p_MaxElements">
/// Maximum number of elements allowed in the level sets (inclusive).
/// </param>
/// <note> The neighboring pattern is the cross '+' (4 neighbors per pixel) </note>
template <bool MAX, typename T>
T LocalLevelSets(
    std::vector<SLevelSetInfo<T>>& o_LevelSetInfo,
    std::vector<SPixelData<T>>& o_PixelData,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_StartY,
    std::size_t p_StartX,
    std::size_t p_MaxElements,
    const T p_StopValue,
    const SPixelState<MAX, T>* p_PixelState
);

// Compilation toggle for deduplication of pixels in level sets
#ifndef SPOT_DETECTOR_DEDUPLICATION
#define SPOT_DETECTOR_DEDUPLICATION 1
#endif


/// <summary>
/// Compute, starting from a given pixel, all level sets
/// of size <= p_MaxElements containing the pixel, and select
/// the first one passing the NFA threshold (epsilon). If one
/// such level passes, p_OnFound is called.
///
/// A level set l is a set of neigboring pixels such that
/// all values are >= l (or <= respectively).
///
/// The function is expected to be called with a pixel that is part
/// of its extremum level set (a local maxima for >= and vice versa).
/// This is not mandatory however.
///
/// </summary>
/// <typeparam name="MAX">True for '>= l', False for '<= l' </typeparam>
/// <typeparam name="T">Type of data</typeparam>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_StartY"> Y position of the starting pixel </param>
/// <param name="p_StartX"> X position of the starting pixel </param>
/// <param name="p_MaxElements">
/// Maximum number of elements allowed in the level sets (inclusive).
/// </param>
/// <param name="p_LogEpsilonThreshold"> log of the NFA threshold </param>
/// <param name="p_NoiseLevel"> estimated std of the level set differences </param>
/// <param name="r_PixelState">
/// Per-pixel state array for deduplication (C contiguous).
/// Must be of size p_Height * p_Width. Unused if
/// SPOT_DETECTOR_DEDUPLICATION == 0.
/// </param>
/// <note> The neighboring pattern is the cross '+' (4 neighbors per pixel) </note>
template <bool MAX, typename T>
void FindMeaningFulLevelSets(
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_StartY,
    std::size_t p_StartX,
    std::size_t p_MaxElements,
    double p_LogEpsilonThreshold,
    double p_NoiseLevel,
    SPixelState<MAX, T>* r_PixelState, // Unused if SPOT_DETECTOR_DEDUPLICATION == 0
    std::function<void(const SLevelSetInfo<T>&,
                       const std::vector<SPixelData<T>>&,
                       size_t, size_t,
                       double)> p_OnFound
);

/// <summary>
/// Remove all level sets containing the starting pixel that are
/// <= p_MaxElements in size by setting those pixels to a background value.
///
/// The function computes all level sets containing the starting pixel
/// and sets all pixels in level sets of size <= p_MaxElements to the
/// next level value (effectively removing small regions).
///
/// </summary>
/// <typeparam name="MAX">True for '>= l', False for '<= l' </typeparam>
/// <typeparam name="T">Type of data</typeparam>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_StartY"> Y position of the starting pixel </param>
/// <param name="p_StartX"> X position of the starting pixel </param>
/// <param name="p_Dst"> Output 2D image (C contiguous), initialized as copy of p_Src </param>
/// <param name="p_MaxElements">
/// Maximum number of elements for level sets to be removed (inclusive).
/// </param>
/// <note> The neighboring pattern is the cross '+' (4 neighbors per pixel) </note>
/// <note> p_Dst and p_Src should not overlap </note>
template <bool MAX, typename T>
void RemoveSmallLevelSets(
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_StartY,
    std::size_t p_StartX,
    T* p_Dst,
    std::size_t p_MaxElements,
    SPixelState<MAX, T>* r_PixelState // Unused if SPOT_DETECTOR_DEDUPLICATION == 0
);