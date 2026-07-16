// This file implements the level set search used for the
// implementation that doesn't rely on a Max-Tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>

#include "bitarray.hpp"
#include "meaningful_ll.hpp"
#include "spot_detector.hpp"

// <summary> Class that ensures all pixels set in the l_Seen array are released</summary>
template <typename T>
class ScratchMemsetGuard
{
private:
    std::size_t m_Width;
    std::vector<SPixelData<T>>& m_PixelData;
    std::vector<SNeigborPixelData<T>>& m_NeigborsData;
    AlignedBitArray2D& m_Seen;

public:
    ScratchMemsetGuard(
        std::size_t p_Width,
        std::vector<SPixelData<T>>& p_PixelData,
        std::vector<SNeigborPixelData<T>>& p_NeigborsData,
        AlignedBitArray2D& p_Seen
    ) : m_Width(p_Width)
      , m_PixelData(p_PixelData)
      , m_NeigborsData(p_NeigborsData)
      , m_Seen(p_Seen) {}

    ~ScratchMemsetGuard()
    {
        for (auto& l_PixelDatum: m_PixelData)
        {
            // We use resetBlock instead of reset to avoid bitshift logic
            // basically this we are doing a cheap memset of the whole array
            m_Seen.resetBlock(l_PixelDatum.m_X, l_PixelDatum.m_Y);
        }
        for (auto& l_PixelDatum: m_NeigborsData)
        {
            // Note: we ignore the m_Merged flag since it is ok to overcommit
            m_Seen.resetBlock(l_PixelDatum.m_X, l_PixelDatum.m_Y);
        }
        // Note: it would possibly be more efficient to do a min/max
        // and do a rectangular memset. But you cannot guarantee linear
        // complexity that way.
    }
};


/// <summary>
/// Compute, starting from a given pixel, all level set
/// of size <= p_MaxElements containing the pixel.
///
/// A level set l is a set of neigboring pixels such that
/// all values are >= l (or <= respectively).
///
/// The function must be called with a pixel that is part
/// of its extremum level set (a local maxima for >= and vice versa).
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
/// However in that case, these additional pixels are a subset
/// (allowed to be complete) of the next level set (of intensity
/// equal to the return value of this function)
/// </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_StartY"> Y position of the starting pixel </param>
/// <param name="p_StartX"> X position of the starting pixel </param>
/// <param name="p_MaxElements">
/// Maximum number of elements allowed in the level sets (inclusive).
/// Note SPOT_DETECTOR_DEDUPLICATION relies on all calls to this function
/// being made with p_MaxElements equal to the same value.
/// </param>
/// <param name="p_StopValue"> Level set value that the algorithm
/// does not need to go beyond (inclusive)
/// </param>
/// <note> The neighboring pattern is the cross '+' (4 neighbors per pixel) </note>
/// <return> The intensity of the next level set </return>
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
    const T p_StopValue
)
{
    auto& l_DebugCounters = getSDebugCounter();
    //std::cout << " " << p_StartY << " " << p_StartX << " " << p_MaxElements << " " << p_StopValue << "\n";

    // Initialize output arrays
    o_LevelSetInfo.clear();
    o_PixelData.clear();

    // No level set if outside the image
    if (p_StartY >= p_Height || p_StartX >= p_Width)
    {
        if constexpr(MAX)
        {
            return std::numeric_limits<T>::max();
        }
        else
        {
            return std::numeric_limits<T>::min();
        }
    }

    // Array that will contain the neighbors of the current shape
    thread_local std::vector<SNeigborPixelData<T>> l_Neighbors;
    // Array of the size of the image that will contain pixels already
    // visited this call
    thread_local AlignedBitArray2D l_Seen;

    l_Neighbors.clear();
    l_Seen.resize(p_Width, p_Height);

    // Guard to reset l_Seen to false without a memset.
    ScratchMemsetGuard l_SeenGuard {
        p_Width,
        o_PixelData,
        l_Neighbors,
        l_Seen
    };

    // - Definition of the main algorithm steps with lambdas (will get inlined) - 

    // Adding a neighbor to the list
    auto l_AddNeighbor = [&](std::size_t p_NeighborY, std::size_t p_NeighborX)
    {
        l_Neighbors.emplace_back(
            p_NeighborY, p_NeighborX, p_Src[p_Width * p_NeighborY + p_NeighborX], false
        );
        //std::cout << " >" << p_NeighborY << " " << p_NeighborX << " " << l_Neighbors.size() << "\n";
        //if (l_Seen.get(p_NeighborX, p_NeighborY))
        //    throw std::runtime_error("");
        l_Seen.set(p_NeighborX, p_NeighborY);
    };

    // Move a neighbor from the neighbor list to the o_PixelData output,
    // Add the new neighbors to the list
    auto l_Promote = [&](std::size_t p_NeighborIdx)
    {
        const std::size_t l_VisitY = l_Neighbors[p_NeighborIdx].m_Y;
        const std::size_t l_VisitX = l_Neighbors[p_NeighborIdx].m_X;
        const T l_Value = l_Neighbors[p_NeighborIdx].m_V;

        l_DebugCounters.m_NumMerge += 1;
        // Append to the level set pixel array
        o_PixelData.emplace_back(
            l_VisitY,
            l_VisitX,
            l_Value
        );
        // Remove from the neighbors list
        l_Neighbors[p_NeighborIdx].m_Merged = true;
        //printf("%03ld %03ld %03ld %03ld %ld\n", l_VisitY, l_VisitY, p_StartY, p_StartX, p_NeighborIdx);
        //std::cout << l_VisitY << " " << l_VisitX << " " << p_StartY << " " << p_StartX << " " << p_NeighborIdx << "\n";

        // Add unvisited neighbors
        // Left: x-1, y
        if (l_VisitX > 0) [[likely]]
        {
            const std::size_t l_NeighborX = l_VisitX - 1;
            if (!l_Seen.get(l_NeighborX, l_VisitY))
            {
                l_AddNeighbor(l_VisitY, l_NeighborX);
            }
        }
        
        // Right: x+1, y
        if (l_VisitX < p_Width - 1) [[likely]]
        {
            const std::size_t l_NeighborX = l_VisitX + 1;
            if (!l_Seen.get(l_NeighborX, l_VisitY))
            {
                l_AddNeighbor(l_VisitY, l_NeighborX);
            }
        }
        
        // Top: x, y-1
        if (l_VisitY > 0) [[likely]]    
        {
            const std::size_t l_NeighborY = l_VisitY - 1;
            if (!l_Seen.get(l_VisitX, l_NeighborY))
            {
                l_AddNeighbor(l_NeighborY, l_VisitX);
            }
        }
        
        // Bottom: x, y+1
        if (l_VisitY < p_Height - 1) [[likely]]
        {
            const std::size_t l_NeighborY = l_VisitY + 1;
            if (!l_Seen.get(l_VisitX, l_NeighborY))
            {
                l_AddNeighbor(l_NeighborY, l_VisitX);
            }
        }
    };

    // Visit a neighbor and consider whether to add it to the level set
    auto l_Visit = [&](std::size_t p_NeighborIdx, T p_Level)
    {
        if (l_Neighbors[p_NeighborIdx].m_Merged)
        {
            return;
        }
        if constexpr (MAX)
        {
            if (l_Neighbors[p_NeighborIdx].m_V >= p_Level)
            {
                l_Promote(p_NeighborIdx);
            }
        }
        else
        {
            if (l_Neighbors[p_NeighborIdx].m_V <= p_Level)
            {
                l_Promote(p_NeighborIdx);
            }
        }
    };

    // Find Max/Min level among valid neighbors
    auto l_FindNeighborExtremum = [&]()
    {
        // Find next level
        T l_Level;

        if constexpr (MAX)
        {
            l_Level = std::numeric_limits<T>::min();
            for (auto& l_Neighbor : l_Neighbors)
            {
                if (l_Level < l_Neighbor.m_V && !l_Neighbor.m_Merged)
                {
                    l_Level = l_Neighbor.m_V;
                }
            }
        }
        else
        {
            l_Level = std::numeric_limits<T>::max();
            for (auto& l_Neighbor : l_Neighbors)
            {
                if (l_Level > l_Neighbor.m_V && !l_Neighbor.m_Merged)
                {
                    l_Level = l_Neighbor.m_V;
                }
            }
        }
        return l_Level;
    };

    // Compact the neighbors array
    auto l_CompactNeighbors = [&]()
    {
        std::size_t l_LookIdx = 0;

        // Skip positions with no change
        for (; l_LookIdx < l_Neighbors.size(); ++l_LookIdx)
        {
            if (l_Neighbors[l_LookIdx].m_Merged)
            {
                ++l_LookIdx;
                break;
            }
        }

        // Next available slot
        std::size_t l_MergeIdx = l_LookIdx;

        // Overwrite over merged elements
        for (; l_LookIdx < l_Neighbors.size(); ++l_LookIdx)
        {
            if (l_Neighbors[l_LookIdx].m_Merged)
            {
                continue;
            }
            l_Neighbors[l_MergeIdx] = l_Neighbors[l_LookIdx];
            ++l_MergeIdx;
        }

        // Clamp to merged region
        l_Neighbors.resize(l_MergeIdx);
    };

    // - Core algorithm - 

    // Initialize search with the starting point
    T l_CurLevel = p_Src[p_Width * p_StartY + p_StartX];

    // Can occur if the local maxima is on a flat region
    if constexpr (MAX)
    {
        if (l_CurLevel <= p_StopValue)
        {
            return l_CurLevel;
        }
    } else {
        if (l_CurLevel >= p_StopValue)
        {
            return l_CurLevel;
        }
    }

    l_Neighbors.emplace_back(
        p_StartY, p_StartX, l_CurLevel, false
    );
    l_Seen.set(p_StartX, p_StartY);

    // Build level sets up to the target number of elements
    while (o_PixelData.size() < p_MaxElements)
    {
        // Visit all neighbors until convergence
        // Note l_Neighbors.size() is updated frequently
        // as neighbors are pushed and visited items are not removed.
        for (std::size_t l_VisitIdx = 0;
             l_VisitIdx < l_Neighbors.size() && o_PixelData.size() <= p_MaxElements;
             ++l_VisitIdx)
        {
            l_Visit(l_VisitIdx, l_CurLevel);
        }
        //std::cout << "-- " << l_CurLevel << "\n";
        // Note: the "o_PixelData.size() <= p_MaxElements;" test above in the for loop
        // is NOT '<' because we need to detect when the level overflows (too many pixels
        // in the level means the level is rejected). 

        // Level overflow check
        if (o_PixelData.size() > p_MaxElements)
        {
            // The current level set is too large and will not
            // be recorded. We stop here. Thus the intensity of the
            // next level set is precisely l_CurLevel.
            return l_CurLevel;
        }

        // Flush information about the treated level
        o_LevelSetInfo.emplace_back(
            o_PixelData.size()-1,
            l_CurLevel
        );

        // Compact the neighbor list to remove merged items
        //l_CompactNeighbors();

        // Find next level
        T l_NextLevel = l_FindNeighborExtremum();

        if (o_PixelData.size() == p_MaxElements)
        {
            // We have filled exactly the allowed number of pixels.
            // We stop here. The intensity of the next level set is
            // precisely l_NextLevel.
            return l_NextLevel;
        }

        // Stopping condition on the level set value
        if constexpr (MAX)
        {
            if (l_NextLevel <= p_StopValue)
            {
                return l_NextLevel;
            }
        } else {
            if (l_NextLevel >= p_StopValue)
            {
                return l_NextLevel;
            }
        }

        // Unchanged extremum check (For safety).
        // Could occur if all neighbors are merged (small image ?) or flat zero image
        if constexpr (MAX)
        {
            if (l_NextLevel == std::numeric_limits<T>::min())
            {
                // Note: similar to p_StopValue, this also results
                // in preventing level sets with the minimum/maximum
                // representable value, which is acceptable in our case
                // as it is always, when present, the root level set,
                // and thus we cannot compute a delta for it.
                return l_NextLevel;
            }
        }
        else
        {
            if (l_NextLevel == std::numeric_limits<T>::max())
            {
                return l_NextLevel;
            }
        }

        // Update the target level
        if constexpr (MAX)
        {
            assert (l_CurLevel > l_NextLevel);
        }
        else
        {
            assert (l_CurLevel < l_NextLevel);
        }
        l_CurLevel = l_NextLevel;
    }

    return l_FindNeighborExtremum(); 
}

/// <summary>
/// Approximation of the number of shape
/// combinaisons for p_NumPoints contiguous points
/// <summary>
/// <param name="p_NumPoints"> Number of contiguous points </param>
//static double inline NumberOfPolyominoes(int p_NumPoints)
//{
//    // known approximation https://www.ipol.im/pub/art/2021/342/
//    constexpr double c_Alpha = 0.316915;
//    constexpr double c_Beta = 4.062570;
//    return c_Alpha * std::pow(c_Beta, p_NumPoints)/p_NumPoints;
//}

/// <summary>
/// Logarithm of the approximation of the number of shape
/// combinaisons for p_NumPoints contiguous points
/// <summary>
/// <param name="p_NumPoints"> Number of contiguous points </param>
static double inline LogNumberOfPolyominoes(int p_NumPoints)
{
    // known approximation https://www.ipol.im/pub/art/2021/342/
    constexpr double c_Alpha = 0.316915;
    constexpr double c_Beta = 4.062570;
    // pow(x, y) is exp(y * log(x))
    const double l_NumPoints = static_cast<double>(p_NumPoints);
    return std::log(c_Alpha) + l_NumPoints * std::log(c_Beta) - std::log(l_NumPoints);
}

/// <summary>
/// Precomputed lookup table for LogNumberOfPolyominoes for p_NumPoints in [1, 128]
/// </summary>
constexpr std::size_t LOG_POLYOMINOES_TABLE_SIZE = 128;

static std::array<double, LOG_POLYOMINOES_TABLE_SIZE> ComputeLogPolyominoesTable()
{
    std::array<double, LOG_POLYOMINOES_TABLE_SIZE> l_Table{};

    for (std::size_t l_I = 1; l_I <= LOG_POLYOMINOES_TABLE_SIZE; ++l_I)
    {
        l_Table[l_I - 1] = LogNumberOfPolyominoes(static_cast<int>(l_I));
    }

    return l_Table;
}

static const std::array<double, LOG_POLYOMINOES_TABLE_SIZE> g_LogPolyominoesTable = ComputeLogPolyominoesTable();

/// <summary>
/// Logarithm of the approximation of the number of shape
/// combinaisons for p_NumPoints contiguous points
/// <summary>
/// <param name="p_NumPoints"> Number of contiguous points </param>
static double inline GetLogNumberOfPolyominoes(int p_NumPoints)
{
    if (p_NumPoints <= 0) [[unlikely]]
    {
        return 0.0;
    }
    else if (static_cast<std::size_t>(p_NumPoints) <= LOG_POLYOMINOES_TABLE_SIZE)
    {
        // We use a lookup table for small sizes for speed
        return g_LogPolyominoesTable[static_cast<std::size_t>(p_NumPoints) - 1];
    }
    else
    {
        return LogNumberOfPolyominoes(p_NumPoints);
    }
}

/// <summary> Compute the log Number of False Alarms score </summary>
/// <param name="p_NumPoints"> Number of points in the level set</param>
/// <param name="p_LevelDelta"> Intensity delta with next level set </param>
/// <param name="p_NoiseLevel"> Expected standard deviation of one level set to the next </param>
/// <note> This is actually not the log NFA, but the log NFA + log(M) where M is the number of pixels
/// to avoid computing log(M) multiple times. </note>
static double inline ComputeLogNFA(
    int p_NumPoints,
    double p_LevelDelta,
    double p_NoiseLevel
)
// For a gaussian model of noise
//{
//    return LogNumberOfPolyominoes(p_NumPoints)
//        + p_NumPoints * std::log(
//            std::erfc(p_LevelDelta / (std::sqrt(2.) * p_NoiseLevel))
//        );
//}
{
    return GetLogNumberOfPolyominoes(p_NumPoints)
        - (p_NumPoints / p_NoiseLevel) * p_LevelDelta;
}

/// <summary>
/// Get if we should skip processing this pixel based on deduplication state
/// </summary>
template <bool MAX, typename T>
static bool inline shouldSkipPixel(
    const SPixelState<MAX, T>* p_PixelState,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_StartY,
    std::size_t p_StartX)
{
#if SPOT_DETECTOR_DEDUPLICATION
    // Check if start pixel is already part of a detection
    const std::size_t l_StartIdx = p_StartY * p_Width + p_StartX;
    const SPixelState<MAX, T> l_StartState = p_PixelState[l_StartIdx];

    if (l_StartState.m_PartOfDetection)
    {
        // Already part of a detection, skip entirely
        //std::cout << "Pixel (" << p_StartY << ", " << p_StartX
        //          << ") already part of a detection, skipping.\n";
        return true;
    }
#endif
    return false;
}

/// <summary>
/// Get the stop level for the current pixel based on deduplication state
/// </summary>
template <bool MAX, typename T>
static T inline getStopLevel(
    const SPixelState<MAX, T>* p_PixelState,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_StartY,
    std::size_t p_StartX)
{
#if SPOT_DETECTOR_DEDUPLICATION
    // Check if start pixel is already part of a detection
    const std::size_t l_StartIdx = p_StartY * p_Width + p_StartX;
    const SPixelState<MAX, T> l_StartState = p_PixelState[l_StartIdx];

    // Basically using the inclusion property of level set, if a level set
    // was explored by another pixel, and that level set includes our seed,
    // we will get the same level set for that size, thus we don't need
    // to retest it. The same reasoning applies to level sets that didn't
    // fit p_MaxElements, which why we use the value as limit, rather
    // than clamping p_MaxElements.
    const T m_StopLevel = l_StartState.m_MinLevelSetValueExplored;
#else
    const SPixelState<MAX, T> l_StartState{};
    const T m_StopLevel = l_StartState.m_MinLevelSetValueExplored;
#endif
    return m_StopLevel;
}

/// <summary>
/// Update the pixel state after processing a set of level sets
/// </summary>
template <bool MAX, typename T>
static void inline updatePixelState(
    [[maybe_unused]] const std::vector<SLevelSetInfo<T>>& p_LevelSetInfo,
    [[maybe_unused]] const std::vector<SPixelData<T>>& p_PixelData,
    [[maybe_unused]] SPixelState<MAX, T>* r_PixelState,
    [[maybe_unused]] std::size_t p_Width,
    [[maybe_unused]] const T p_NextLevelT,
    [[maybe_unused]] int64_t p_LevelDetectionFound
)
{
#if SPOT_DETECTOR_DEDUPLICATION
    // Update pixel state for all explored pixels
    if (p_LevelDetectionFound >= 0)
    {
        // Find the level set that was detected
        // Mark all pixels in the detected level set as part of a detection
        for (std::size_t l_I = 0; l_I < p_LevelSetInfo[p_LevelDetectionFound].m_StopIdx + 1; ++l_I)
        {
            const std::size_t l_Idx = p_PixelData[l_I].m_Y * p_Width + p_PixelData[l_I].m_X;
            r_PixelState[l_Idx].m_PartOfDetection = true;
            // We don't need to update m_MinLevelSetValueExplored since the above will
            // skip processing completly
            //std::cout << "Pixel (" << p_PixelData[l_I].m_Y << ", " << p_PixelData[l_I].m_X
            //                  << ") set to state " << PIXEL_STATE_DETECTION << "\n";
        }
        // Remaining level sets (larger than the detected one) are not marked as they
        // could contain smaller meaningful level sets.
    }
    std::size_t l_LevelStartIdx = 0;
    if (!p_LevelSetInfo.empty())
    {
        if (p_LevelDetectionFound >= 0)
        {
            // Start after the detected level set
            l_LevelStartIdx = p_LevelSetInfo[p_LevelDetectionFound].m_StopIdx + 1;
        }
        // Update each pixel with the minimum level set size it belongs to
        for (int64_t l_LevelIdx = p_LevelDetectionFound + 1; l_LevelIdx < (int64_t)p_LevelSetInfo.size(); ++l_LevelIdx)
        {
            const T l_LevelValue = p_LevelSetInfo[l_LevelIdx].m_Level;
            std::size_t l_LevelSetStop = p_LevelSetInfo[l_LevelIdx].m_StopIdx;
            
            // Update all pixels in this level set
            for (std::size_t l_I = l_LevelStartIdx; l_I <= l_LevelSetStop; ++l_I)
            {
                const std::size_t l_Idx = p_PixelData[l_I].m_Y * p_Width + p_PixelData[l_I].m_X;
                
                // Store minimum: smaller level sets have priority
                if constexpr (MAX)
                {
                    if (l_LevelValue > r_PixelState[l_Idx].m_MinLevelSetValueExplored)
                    {
                        r_PixelState[l_Idx].m_MinLevelSetValueExplored = l_LevelValue;
                        //std::cout << "Pixel (" << p_PixelData[l_I].m_Y << ", " << p_PixelData[l_I].m_X
                        //          << ") set to state " << l_LevelSetSize << "\n";
                    }
                } else {
                    if (l_LevelValue < r_PixelState[l_Idx].m_MinLevelSetValueExplored)
                    {
                        r_PixelState[l_Idx].m_MinLevelSetValueExplored = l_LevelValue;
                        //std::cout << "Pixel (" << p_PixelData[l_I].m_Y << ", " << p_PixelData[l_I].m_X
                        //          << ") set to state " << l_LevelSetSize << "\n";
                    }
                }
            }
            l_LevelStartIdx = l_LevelSetStop + 1;
        }
    }
    // Update the state of the other pixels part of the extremum level line
    for (std::size_t l_I = l_LevelStartIdx; l_I < p_PixelData.size(); ++l_I)
    {
        const std::size_t l_Idx = p_PixelData[l_I].m_Y * p_Width + p_PixelData[l_I].m_X;

        if constexpr (MAX)
        {
            if (p_NextLevelT > r_PixelState[l_Idx].m_MinLevelSetValueExplored)
            {
                r_PixelState[l_Idx].m_MinLevelSetValueExplored = p_NextLevelT;
            }
        } else {
            if (p_NextLevelT < r_PixelState[l_Idx].m_MinLevelSetValueExplored)
            {
                r_PixelState[l_Idx].m_MinLevelSetValueExplored = p_NextLevelT;
            }
        }
    }
#endif
}

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
)
{
    auto& l_DebugCounters = getSDebugCounter();

    thread_local std::vector<SLevelSetInfo<T>> l_LevelSetInfo;
    thread_local std::vector<SPixelData<T>> l_PixelData;

    if (shouldSkipPixel<MAX, T>(r_PixelState, p_Height, p_Width, p_StartY, p_StartX))
    {
        return;
    }

    T l_NextLevelT = LocalLevelSets<MAX, T>(
        l_LevelSetInfo,
        l_PixelData,
        p_Src,
        p_Height,
        p_Width,
        p_StartY,
        p_StartX,
        p_MaxElements,
        getStopLevel<MAX, T>(r_PixelState, p_Height, p_Width, p_StartY, p_StartX)
    );
    double l_NextLevelD = static_cast<double>(l_NextLevelT);

    l_DebugCounters.m_NumLevel += l_LevelSetInfo.size();
    if (l_LevelSetInfo.size() > 0)
    {
        l_DebugCounters.m_NumRetained += l_LevelSetInfo.back().m_StopIdx+1;
    }

    int64_t l_LevelDetectionFound = -1;
    (void)l_LevelDetectionFound;

    // We return the largest meaningful level set
    for (int64_t l_LevelIdx = (int64_t)(l_LevelSetInfo.size())-1; l_LevelIdx >= 0; --l_LevelIdx)
    {
        double l_LevelD = static_cast<double>(l_LevelSetInfo[l_LevelIdx].m_Level);
        std::size_t l_NumPoints = l_LevelSetInfo[l_LevelIdx].m_StopIdx + 1;

        // l_Delta = abs(l_LevelD-l_NextLevelD)
        double l_Delta;
        if constexpr (MAX)
        {
            l_Delta = l_LevelD - l_NextLevelD;
        }
        else
        {
            l_Delta = l_NextLevelD - l_LevelD;
        }
        assert (l_Delta >= 0.);
        if (l_Delta < 0)
            throw std::runtime_error("Assert False");

        // Stop at the largest shape that passes the threshold
        double l_LogNFA = ComputeLogNFA((int)l_NumPoints, l_Delta, p_NoiseLevel);
        if (l_LogNFA < p_LogEpsilonThreshold)
        {
            //std::cerr << l_NumPoints<< " " << l_Delta << " " << p_NoiseLevel << " " << l_LogNFA << " " << p_LogEpsilonThreshold << "\n";
            p_OnFound(
                l_LevelSetInfo[l_LevelIdx],
                l_PixelData,
                p_StartY,
                p_StartX,
                l_LogNFA
            );
            l_LevelDetectionFound = l_LevelIdx;
            break;
        }

        l_NextLevelD = l_LevelD;
    }

    updatePixelState<MAX, T>(
        l_LevelSetInfo,
        l_PixelData,
        r_PixelState,
        p_Width,
        l_NextLevelT,
        l_LevelDetectionFound
    );
}


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
){
    thread_local std::vector<SLevelSetInfo<T>> l_LevelSetInfo;
    thread_local std::vector<SPixelData<T>> l_PixelData;

    if (shouldSkipPixel<MAX, T>(r_PixelState, p_Height, p_Width, p_StartY, p_StartX))
    {
        return;
    }

    // Compute all level sets up to p_MaxElements
    T l_NextLevel = LocalLevelSets<MAX, T>(
        l_LevelSetInfo,
        l_PixelData,
        p_Src,
        p_Height,
        p_Width,
        p_StartY,
        p_StartX,
        p_MaxElements,
        getStopLevel<MAX, T>(r_PixelState, p_Height, p_Width, p_StartY, p_StartX)
    );

    // If no level sets found or starting pixel is outside bounds, return
    if (l_LevelSetInfo.empty())
    {
        return;
    }

    // Find the largest level set that is <= p_MaxElements
    std::size_t l_MaxValidIdx = l_LevelSetInfo.size() - 1;
    std::size_t l_NumPixelsToRemove = l_LevelSetInfo[l_MaxValidIdx].m_StopIdx + 1;

    assert (l_NumPixelsToRemove <= p_MaxElements);

    // Set all pixels in the level sets to the next level value
    for (std::size_t i = 0; i < l_NumPixelsToRemove; ++i)
    {
        const std::size_t l_Y = l_PixelData[i].m_Y;
        const std::size_t l_X = l_PixelData[i].m_X;
        p_Dst[l_Y * p_Width + l_X] = l_NextLevel;
    }

    updatePixelState<MAX, T>(
        l_LevelSetInfo,
        l_PixelData,
        r_PixelState,
        p_Width,
        l_NextLevel,
        l_MaxValidIdx // The whole level set was treated (== 'detected')
    );
}


template
uint8_t LocalLevelSets<true, uint8_t>(
    std::vector<SLevelSetInfo<uint8_t>>&,
    std::vector<SPixelData<uint8_t>>&,
    const uint8_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    const uint8_t
);

template
uint8_t LocalLevelSets<false, uint8_t>(
    std::vector<SLevelSetInfo<uint8_t>>&,
    std::vector<SPixelData<uint8_t>>&,
    const uint8_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    const uint8_t
);

template
uint16_t LocalLevelSets<true, uint16_t>(
    std::vector<SLevelSetInfo<uint16_t>>&,
    std::vector<SPixelData<uint16_t>>&,
    const uint16_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    const uint16_t
);

template
uint16_t LocalLevelSets<false, uint16_t>(
    std::vector<SLevelSetInfo<uint16_t>>&,
    std::vector<SPixelData<uint16_t>>&,
    const uint16_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    const uint16_t
);

template
float LocalLevelSets<true, float>(
    std::vector<SLevelSetInfo<float>>&,
    std::vector<SPixelData<float>>&,
    const float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    const float
);

template
float LocalLevelSets<false, float>(
    std::vector<SLevelSetInfo<float>>&,
    std::vector<SPixelData<float>>&,
    const float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    const float
);



template
void FindMeaningFulLevelSets<true, uint8_t>(
    const uint8_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    SPixelState<true, uint8_t>*,
    std::function<void(const SLevelSetInfo<uint8_t>&,
                       const std::vector<SPixelData<uint8_t>>&,
                       size_t, size_t,
                       double)>
);

template
void FindMeaningFulLevelSets<false, uint8_t>(
    const uint8_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    SPixelState<false, uint8_t>*,
    std::function<void(const SLevelSetInfo<uint8_t>&,
                       const std::vector<SPixelData<uint8_t>>&,
                       size_t, size_t,
                       double)>
);

template
void FindMeaningFulLevelSets<true, uint16_t>(
    const uint16_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    SPixelState<true, uint16_t>*,
    std::function<void(const SLevelSetInfo<uint16_t>&,
                       const std::vector<SPixelData<uint16_t>>&,
                       size_t, size_t,
                       double)>
);

template
void FindMeaningFulLevelSets<false, uint16_t>(
    const uint16_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    SPixelState<false, uint16_t>*,
    std::function<void(const SLevelSetInfo<uint16_t>&,
                       const std::vector<SPixelData<uint16_t>>&,
                       size_t, size_t,
                       double)>
);

template
void FindMeaningFulLevelSets<true, float>(
    const float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    SPixelState<true, float>*,
    std::function<void(const SLevelSetInfo<float>&,
                       const std::vector<SPixelData<float>>&,
                       size_t, size_t,
                       double)>
);

template
void FindMeaningFulLevelSets<false, float>(
    const float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    SPixelState<false, float>*,
    std::function<void(const SLevelSetInfo<float>&,
                       const std::vector<SPixelData<float>>&,
                       size_t, size_t,
                       double)>
);

template
void RemoveSmallLevelSets<true, uint8_t>(
    const uint8_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    uint8_t*,
    std::size_t,
    SPixelState<true, uint8_t>*
);

template
void RemoveSmallLevelSets<false, uint8_t>(
    const uint8_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    uint8_t*,
    std::size_t,
    SPixelState<false, uint8_t>*
);

template
void RemoveSmallLevelSets<true, uint16_t>(
    const uint16_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    uint16_t*,
    std::size_t,
    SPixelState<true, uint16_t>*
);

template
void RemoveSmallLevelSets<false, uint16_t>(
    const uint16_t*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    uint16_t*,
    std::size_t,
    SPixelState<false, uint16_t>*
);

template
void RemoveSmallLevelSets<true, float>(
    const float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    float*,
    std::size_t,
    SPixelState<true, float>*
);

template
void RemoveSmallLevelSets<false, float>(
    const float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    float*,
    std::size_t,
    SPixelState<false, float>*
);