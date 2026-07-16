// This file implements the spot detector using
// a Max-Tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <iostream>

#include "spot_detector_maxtree.hpp"
#include "maxpool.hpp"

// Pylene includes
#include <mln/core/image/ndimage.hpp>
#include <mln/core/neighborhood/c4.hpp>
#include <mln/morpho/maxtree.hpp>

// Helper structures for accumulating node properties
template <typename T>
struct NodeInfo
{
    std::size_t m_Size = 0;            // Number of pixels in this node
    std::size_t m_SubTreeSize = 0;     // Number of pixels in this node and the children nodes (recursively)
    std::size_t m_MinY = SIZE_MAX;
    std::size_t m_MinX = SIZE_MAX;
    std::size_t m_MaxY = 0;
    std::size_t m_MaxX = 0;
    T m_Level;                         // Intensity level of this node
    T m_NextLevel;                     // Intensity of parent (next level)
    bool m_HasLocalMax = false;        // Whether this node contains a local maximum
    std::vector<int64_t> m_Children;   // Direct children of this node
};

/// <summary>
/// Approximation of the number of shape
/// combinaisons for p_NumPoints contiguous points
/// <summary>
/// <param name="p_NumPoints"> Number of contiguous points </param>
static double inline NumberOfPolyominoes(int p_NumPoints)
{
    // known approximation https://www.ipol.im/pub/art/2021/342/
    constexpr double c_Alpha = 0.316915;
    constexpr double c_Beta = 4.062570;
    return c_Alpha * std::pow(c_Beta, p_NumPoints)/p_NumPoints;
}

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
/// Core maxtree-based detection implementation
/// </summary>
template <typename T>
void FindSpotMaxtreeCore(
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel,
    std::function<void(int, const std::vector<NodeInfo<T>>&, const mln::image2d<int>& p_NodeMap, double)> p_OnDetection
)
{
    if (p_Height == 0 || p_Width == 0)
        return;

    // Thread-local allocations for efficiency
    thread_local std::vector<uint8_t> l_LocalMaxMask;
    thread_local std::vector<NodeInfo<T>> l_NodeInfo;
    thread_local std::vector<std::size_t> l_SubtreeSize;
    thread_local std::vector<uint8_t> l_NodeUsed;
    thread_local std::vector<int64_t> l_Stack;
    
    // Compute local maxima mask using MaxPool2D<5,5>
    const std::size_t l_TotalPixels = p_Height * p_Width;
    l_LocalMaxMask.resize(l_TotalPixels);
    std::memset(l_LocalMaxMask.data(), 0, l_TotalPixels);
    
    std::function<void(const T*, std::size_t, std::size_t, std::size_t, std::size_t)> l_MarkLocalMax = 
        [&](const T* p_SrcBis, std::size_t p_HeightBis, std::size_t p_WidthBis,
            std::size_t p_Y, std::size_t p_X)
    {
        l_LocalMaxMask[p_Y * p_Width + p_X] = 1;
    };
    
    MaxPool2D<5, 5>(p_Src, p_Height, p_Width, l_MarkLocalMax);

    // Convert raw pointer to mln::ndbuffer_image without copying
    int l_Sizes[] = {static_cast<int>(p_Width), static_cast<int>(p_Height)};
    std::ptrdiff_t l_ByteStrides[] = {
        static_cast<std::ptrdiff_t>(sizeof(T)),
        static_cast<std::ptrdiff_t>(p_Width * sizeof(T))
    };
    auto l_Img = mln::image2d<T>::from_buffer(
        const_cast<T*>(p_Src),
        l_Sizes,
        l_ByteStrides,
        false
    );

    // Build maxtree with 4-connectivity
    auto [l_Tree, l_NodeMap] = mln::morpho::maxtree(l_Img, mln::c4);
    
    const std::size_t l_NumNodes = l_Tree.parent.size();
    if (l_NumNodes == 0)
        return;

    // Resize and initialize thread-local structures
    l_NodeInfo.resize(l_NumNodes);
    l_SubtreeSize.resize(l_NumNodes);
    l_NodeUsed.resize(l_NumNodes);
    
    // Clear node info
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        l_NodeInfo[l_NodeId].m_Size = 0;
        l_NodeInfo[l_NodeId].m_SubTreeSize = 0;
        l_NodeInfo[l_NodeId].m_MinY = SIZE_MAX;
        l_NodeInfo[l_NodeId].m_MinX = SIZE_MAX;
        l_NodeInfo[l_NodeId].m_MaxY = 0;
        l_NodeInfo[l_NodeId].m_MaxX = 0;
        l_NodeInfo[l_NodeId].m_HasLocalMax = false;
        l_NodeInfo[l_NodeId].m_Children.clear();
    }

    // Build children lists
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        l_NodeInfo[l_NodeId].m_Level = l_Tree.values[l_NodeId];
        
        int64_t l_ParentId = l_Tree.parent[l_NodeId];
        if (l_ParentId >= 0)
        {
            l_NodeInfo[l_NodeId].m_NextLevel = l_Tree.values[l_ParentId];
            // Build children list at parent
            l_NodeInfo[l_ParentId].m_Children.push_back(l_NodeId);
        }
        else
        {
            l_NodeInfo[l_NodeId].m_NextLevel = std::numeric_limits<T>::min();
        }
    }

    // First pass: accumulate pixels per node (without storing indices)
    for (std::size_t l_Y = 0; l_Y < p_Height; ++l_Y)
    {
        for (std::size_t l_X = 0; l_X < p_Width; ++l_X)
        {
            mln::point2d pt{static_cast<int>(l_X), static_cast<int>(l_Y)};
            int64_t l_NodeId = l_NodeMap(pt);
            
            if (l_NodeId >= 0 && l_NodeId < static_cast<int64_t>(l_NumNodes))
            {
                NodeInfo<T>& info = l_NodeInfo[l_NodeId];
                
                info.m_Size++;
                info.m_MinY = std::min(info.m_MinY, l_Y);
                info.m_MinX = std::min(info.m_MinX, l_X);
                info.m_MaxY = std::max(info.m_MaxY, l_Y);
                info.m_MaxX = std::max(info.m_MaxX, l_X);
                
                // Check if this pixel is a local maximum
                if (l_LocalMaxMask[l_Y * p_Width + l_X])
                {
                    info.m_HasLocalMax = true;
                }
            }
        }
    }

    // Propagate bounds and local max flag from children to parents
    for (int64_t l_NodeId = l_NumNodes - 1; l_NodeId >= 0; --l_NodeId)
    {
        int64_t l_ParentId = l_Tree.parent[l_NodeId];
        if (l_ParentId >= 0)
        {
            l_NodeInfo[l_ParentId].m_MinY = std::min(l_NodeInfo[l_ParentId].m_MinY, l_NodeInfo[l_NodeId].m_MinY);
            l_NodeInfo[l_ParentId].m_MinX = std::min(l_NodeInfo[l_ParentId].m_MinX, l_NodeInfo[l_NodeId].m_MinX);
            l_NodeInfo[l_ParentId].m_MaxY = std::max(l_NodeInfo[l_ParentId].m_MaxY, l_NodeInfo[l_NodeId].m_MaxY);
            l_NodeInfo[l_ParentId].m_MaxX = std::max(l_NodeInfo[l_ParentId].m_MaxX, l_NodeInfo[l_NodeId].m_MaxX);
            
            // Propagate local max flag
            if (l_NodeInfo[l_NodeId].m_HasLocalMax)
            {
                l_NodeInfo[l_ParentId].m_HasLocalMax = true;
            }
        }
    }

    // Second pass: propagate sizes to parents (accumulate subtree sizes)
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        l_NodeInfo[l_NodeId].m_SubTreeSize = l_NodeInfo[l_NodeId].m_Size;
    }
    
    for (int64_t l_NodeId = l_NumNodes - 1; l_NodeId >= 0; --l_NodeId)
    {
        int64_t l_ParentId = l_Tree.parent[l_NodeId];
        if (l_ParentId >= 0)
        {
            l_NodeInfo[l_ParentId].m_SubTreeSize += l_NodeInfo[l_NodeId].m_SubTreeSize;
        }
    }

    double l_LogEpsilonThreshold = std::log(p_EpsilonThreshold);
    std::memset(l_NodeUsed.data(), 0, l_NumNodes * sizeof(uint8_t));

    // Third pass: evaluate each node for meaningfulness
    // Process from root to leaves (forward order) to find the LARGEST meaningful level set
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        if (l_NodeUsed[l_NodeId])
            continue;
            
        std::size_t l_NodeSize = l_NodeInfo[l_NodeId].m_SubTreeSize;
        
        if (l_NodeSize > p_MaxElements || l_NodeSize == 0)
            continue;

        // Skip if no local maximum in subtree
        if (!l_NodeInfo[l_NodeId].m_HasLocalMax)
            continue;

        // Compute level delta
        T level = l_NodeInfo[l_NodeId].m_Level;
        T next_level = l_NodeInfo[l_NodeId].m_NextLevel;
        double l_Delta = static_cast<double>(level) - static_cast<double>(next_level);
        
        if (l_Delta <= 0)
            l_Delta = 0.0;

        // Compute NFA
        double l_LogNFA = ComputeLogNFA(static_cast<int64_t>(l_NodeSize), l_Delta, p_NoiseLevel);
        
        if (l_LogNFA < l_LogEpsilonThreshold)
        {
            // Mark all nodes in subtree as used using children list
            l_Stack.clear();
            l_Stack.push_back(l_NodeId);
            
            while (!l_Stack.empty())
            {
                int64_t curr = l_Stack.back();
                l_Stack.pop_back();
                
                if (l_NodeUsed[curr])
                    continue;
                    
                l_NodeUsed[curr] = true;
                
                // Add children using pre-built list
                for (int64_t child : l_NodeInfo[curr].m_Children)
                {
                    if (!l_NodeUsed[child])
                    {
                        l_Stack.push_back(child);
                    }
                }
            }
            
            // Call the detection callback
            p_OnDetection(l_NodeId, l_NodeInfo, l_NodeMap, l_LogNFA);
        }
    }
}

template <typename T>
void FindSpotReturnBoxesMaxtree(
    std::vector<SSpotBox>& o_Boxes,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
)
{
    o_Boxes.clear();
    
    auto l_OnDetection = [&](int64_t p_NodeIdx,
                             const std::vector<NodeInfo<T>>& p_NodeInfo,
                             const mln::image2d<int>& p_NodeMap,
                             double p_LogNFA)
    {
        // Bounding box already computed during tree traversal
        std::size_t minY = p_NodeInfo[p_NodeIdx].m_MinY;
        std::size_t minX = p_NodeInfo[p_NodeIdx].m_MinX;
        std::size_t maxY = p_NodeInfo[p_NodeIdx].m_MaxY;
        std::size_t maxX = p_NodeInfo[p_NodeIdx].m_MaxX;
        
        if (minY != SIZE_MAX && minX != SIZE_MAX)
        {
            o_Boxes.emplace_back(
                minY,
                minX,
                maxY - minY + 1,
                maxX - minX + 1,
                p_NodeInfo[p_NodeIdx].m_SubTreeSize,
                p_LogNFA
            );
        }
    };
    
    FindSpotMaxtreeCore<T>(p_Src, p_Height, p_Width, p_MaxElements, 
                           p_EpsilonThreshold, p_NoiseLevel, l_OnDetection);
}

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
)
{
    o_Pixels.clear();
    o_DetectionPixelStartIdx.clear();
    o_DetectionLogNFA.clear();
    
    // Need to store l_NodeMap for pixel reconstruction
    thread_local std::vector<int64_t> l_Stack;
    thread_local std::vector<bool> l_InSubtree;
    
    if (p_Height == 0 || p_Width == 0)
        return;
    
    auto l_OnDetection = [&](int64_t p_NodeIdx,
                             const std::vector<NodeInfo<T>>& p_NodeInfo,
                             const mln::image2d<int>& p_NodeMap,
                             double p_LogNFA)
    {
        std::size_t l_NumNodes = p_NodeInfo.size();
        o_DetectionPixelStartIdx.push_back(o_Pixels.size());
        o_DetectionLogNFA.push_back(p_LogNFA);
        
        // Mark all nodes in the subtree
        l_InSubtree.clear();
        l_InSubtree.resize(l_NumNodes, false);
        
        l_Stack.clear();
        l_Stack.push_back(p_NodeIdx);
        
        while (!l_Stack.empty())
        {
            int64_t curr = l_Stack.back();
            l_Stack.pop_back();
            
            if (l_InSubtree[curr])
                continue;
                
            l_InSubtree[curr] = true;
            
            // Add children
            for (int64_t child : p_NodeInfo[curr].m_Children)
            {
                l_Stack.push_back(child);
            }
        }
        
        // Collect all pixels from the subtree by scanning within bounding box
        std::size_t minY = p_NodeInfo[p_NodeIdx].m_MinY;
        std::size_t minX = p_NodeInfo[p_NodeIdx].m_MinX;
        std::size_t maxY = p_NodeInfo[p_NodeIdx].m_MaxY;
        std::size_t maxX = p_NodeInfo[p_NodeIdx].m_MaxX;
        
        for (std::size_t l_Y = minY; l_Y <= maxY; ++l_Y)
        {
            for (std::size_t l_X = minX; l_X <= maxX; ++l_X)
            {
                mln::point2d pt{static_cast<int>(l_X), static_cast<int>(l_Y)};
                int l_NodeId = p_NodeMap(pt);
                
                // Check if this pixel's node is in the subtree
                if (l_NodeId >= 0 && l_NodeId < static_cast<int>(l_NumNodes) && l_InSubtree[l_NodeId])
                {
                    o_Pixels.emplace_back(l_Y, l_X);
                }
            }
        }
    };
    
    FindSpotMaxtreeCore<T>(p_Src, p_Height, p_Width, p_MaxElements,
                           p_EpsilonThreshold, p_NoiseLevel, l_OnDetection);
}

template <typename T>
void RemoveSmallLevelSetsMaxtree(
    T* o_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
)
{
    if (p_Height == 0 || p_Width == 0)
        return;

    // Initialize output as copy of input
    std::memcpy(o_Dst, p_Src, p_Height * p_Width * sizeof(T));

    // Build maxtree
    int l_Sizes[] = {static_cast<int>(p_Width), static_cast<int>(p_Height)};
    std::ptrdiff_t l_ByteStrides[] = {
        static_cast<std::ptrdiff_t>(sizeof(T)),
        static_cast<std::ptrdiff_t>(p_Width * sizeof(T))
    };
    auto l_Img = mln::image2d<T>::from_buffer(
        const_cast<T*>(p_Src),
        l_Sizes,
        l_ByteStrides,
        false
    );

    auto [l_Tree, l_NodeMap] = mln::morpho::maxtree(l_Img, mln::c4);
    
    const std::size_t l_NumNodes = l_Tree.parent.size();
    if (l_NumNodes == 0)
        return;

    // Thread-local allocations for efficiency
    thread_local std::vector<std::size_t> l_SubtreeSize;
    thread_local std::vector<T> l_ReplacementValue;
    
    l_SubtreeSize.resize(l_NumNodes);
    l_ReplacementValue.resize(l_NumNodes);
    
    // Initialize subtree sizes
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        l_SubtreeSize[l_NodeId] = 0;
        l_ReplacementValue[l_NodeId] = l_Tree.values[l_NodeId]; // Default: keep original value
    }
    
    // Count pixels per node
    for (std::size_t l_Y = 0; l_Y < p_Height; ++l_Y)
    {
        for (std::size_t l_X = 0; l_X < p_Width; ++l_X)
        {
            mln::point2d pt{static_cast<int>(l_X), static_cast<int>(l_Y)};
            int64_t l_NodeId = l_NodeMap(pt);
            
            if (l_NodeId >= 0 && l_NodeId < static_cast<int64_t>(l_NumNodes))
            {
                l_SubtreeSize[l_NodeId]++;
            }
        }
    }
    
    // Propagate subtree sizes from children to parents (bottom-up)
    for (int64_t l_NodeId = l_NumNodes - 1; l_NodeId >= 0; --l_NodeId)
    {
        int64_t l_ParentId = l_Tree.parent[l_NodeId];
        if (l_ParentId >= 0)
        {
            l_SubtreeSize[l_ParentId] += l_SubtreeSize[l_NodeId];
        }
    }
    
    // Compute replacement values (top-down)
    // Process from root to leaves so parent values are already computed
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        if (l_SubtreeSize[l_NodeId] <= p_MaxElements && l_SubtreeSize[l_NodeId] > 0)
        {
            // This node should be removed
            int64_t l_ParentId = l_Tree.parent[l_NodeId];
            
            if (l_ParentId >= 0)
            {
                // Use parent's replacement value (already computed since we go top-down)
                l_ReplacementValue[l_NodeId] = l_ReplacementValue[l_ParentId];
            }
            else
            {
                // Root node with small size: use min value
                l_ReplacementValue[l_NodeId] = std::numeric_limits<T>::min();
            }
        }
        // else: keep l_Tree.values[l_NodeId] (already set as default)
    }
    
    // Apply replacements to output image in a single pass
    for (std::size_t l_Y = 0; l_Y < p_Height; ++l_Y)
    {
        for (std::size_t l_X = 0; l_X < p_Width; ++l_X)
        {
            mln::point2d pt{static_cast<int>(l_X), static_cast<int>(l_Y)};
            int64_t l_NodeId = l_NodeMap(pt);
            
            if (l_NodeId >= 0 && l_NodeId < static_cast<int64_t>(l_NumNodes))
            {
                o_Dst[l_Y * p_Width + l_X] = l_ReplacementValue[l_NodeId];
            }
        }
    }
}

template <typename T>
void ComputeDeltaStatistics(
    std::vector<SLevelStat<T>>& r_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width
)
{
    if (p_Height == 0 || p_Width == 0)
        return;

    // Build maxtree
    int l_Sizes[] = {static_cast<int>(p_Width), static_cast<int>(p_Height)};
    std::ptrdiff_t l_ByteStrides[] = {
        static_cast<std::ptrdiff_t>(sizeof(T)),
        static_cast<std::ptrdiff_t>(p_Width * sizeof(T))
    };
    auto l_Img = mln::image2d<T>::from_buffer(
        const_cast<T*>(p_Src),
        l_Sizes,
        l_ByteStrides,
        false
    );

    auto [l_Tree, l_NodeMap] = mln::morpho::maxtree(l_Img, mln::c4);
    
    const std::size_t l_NumNodes = l_Tree.parent.size();
    if (l_NumNodes == 0)
        return;

    // Thread-local allocations for efficiency
    thread_local std::vector<std::size_t> l_SubtreeSize;
    
    l_SubtreeSize.resize(l_NumNodes);

    // Initialize subtree sizes
    for (std::size_t l_NodeId = 0; l_NodeId < l_NumNodes; ++l_NodeId)
    {
        l_SubtreeSize[l_NodeId] = 0;
    }

    // Count pixels per node
    for (std::size_t l_Y = 0; l_Y < p_Height; ++l_Y)
    {
        for (std::size_t l_X = 0; l_X < p_Width; ++l_X)
        {
            mln::point2d pt{static_cast<int>(l_X), static_cast<int>(l_Y)};
            int64_t l_NodeId = l_NodeMap(pt);
            
            if (l_NodeId >= 0 && l_NodeId < static_cast<int64_t>(l_NumNodes))
            {
                l_SubtreeSize[l_NodeId]++;
            }
        }
    }
    
    // Propagate subtree sizes from children to parents (bottom-up)
    for (int64_t l_NodeId = l_NumNodes - 1; l_NodeId >= 0; --l_NodeId)
    {
        int64_t l_ParentId = l_Tree.parent[l_NodeId];
        if (l_ParentId >= 0)
        {
            l_SubtreeSize[l_ParentId] += l_SubtreeSize[l_NodeId];
        }
    }
    
    // Output statistics, skipping roots
    for (int64_t l_NodeId = l_NumNodes - 1; l_NodeId >= 0; --l_NodeId)
    {
        int64_t l_ParentId = l_Tree.parent[l_NodeId];
        if (l_ParentId >= 0)
        {
            T l_Level = l_Tree.values[l_NodeId];
            T l_NextLevel = l_Tree.values[l_ParentId];
            r_Dst.emplace_back(l_Level, l_NextLevel, l_SubtreeSize[l_NodeId]);
        }
    }
}

// Explicit template instantiations
template
void FindSpotReturnBoxesMaxtree<uint8_t>(
    std::vector<SSpotBox>& o_Boxes,
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void FindSpotReturnBoxesMaxtree<uint16_t>(
    std::vector<SSpotBox>& o_Boxes,
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

// Note: float not supported by pylene maxtree (requires unsigned integer ≤16 bits)

template
void FindSpotReturnPixelListsMaxtree<uint8_t>(
    std::vector<SPixelCoord>& o_Pixels,
    std::vector<std::size_t>& o_DetectionPixelStartIdx,
    std::vector<double>& o_DetectionLogNFA,
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void FindSpotReturnPixelListsMaxtree<uint16_t>(
    std::vector<SPixelCoord>& o_Pixels,
    std::vector<std::size_t>& o_DetectionPixelStartIdx,
    std::vector<double>& o_DetectionLogNFA,
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void RemoveSmallLevelSetsMaxtree<uint8_t>(
    uint8_t* o_Dst,
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);

template
void RemoveSmallLevelSetsMaxtree<uint16_t>(
    uint16_t* o_Dst,
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);

template
void ComputeDeltaStatistics<uint8_t>(
    std::vector<SLevelStat<uint8_t>>& r_Dst,
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width
);

template
void ComputeDeltaStatistics<uint16_t>(
    std::vector<SLevelStat<uint16_t>>& r_Dst,
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width
);

// Note: float not supported by pylene maxtree (requires unsigned integer ≤16 bits)