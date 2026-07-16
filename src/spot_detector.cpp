// This file implements the spot detector using
// a local level set search.

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

#include "maxpool.hpp"
#include "meaningful_ll.hpp"
#include "spot_detector.hpp"

#include <iostream>

template <typename T>
void FindSpotReturnBoxes(
    std::vector<SSpotBox>& o_Boxes,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
)
{
    thread_local std::vector<SPixelState<true, T>> l_PixelState;
    l_PixelState.clear();
    l_PixelState.resize(p_Height * p_Width);

    auto& l_DebugCounters = getSDebugCounter();
    l_DebugCounters.reset();

    double l_LogEpsilonThreshold = std::log(p_EpsilonThreshold);
    auto l_FillBoxes = [&](
        const SLevelSetInfo<T>& p_LLInfo,
        const std::vector<SPixelData<T>>& p_PixelData,
        size_t p_StartY,
        size_t p_StartX,
        double p_LogNFA
    )
    {
        std::size_t l_MinY = p_Height;
        std::size_t l_MinX = p_Width;
        std::size_t l_MaxY = 0;
        std::size_t l_MaxX = 0;
        for (std::size_t l_Index = 0; l_Index <= p_LLInfo.m_StopIdx; ++l_Index)
        {
            l_MinY = std::min(l_MinY, p_PixelData[l_Index].m_Y);
            l_MinX = std::min(l_MinX, p_PixelData[l_Index].m_X);
            l_MaxY = std::max(l_MaxY, p_PixelData[l_Index].m_Y);
            l_MaxX = std::max(l_MaxX, p_PixelData[l_Index].m_X);
        }
        o_Boxes.emplace_back(
            l_MinY,
            l_MinX,
            l_MaxY-l_MinY+1,
            l_MaxX-l_MinX+1,
            p_LLInfo.m_StopIdx+1,
            p_LogNFA
        );
    };

    std::function<void(const T*, std::size_t, std::size_t, std::size_t, std::size_t)>
        l_OnLocaLMax = [&](
            const T* p_SrcBis,
            std::size_t p_HeightBis,
            std::size_t p_WidthBis,
            std::size_t p_StartY,
            std::size_t p_StartX
    )
    {
        FindMeaningFulLevelSets<true, T>(
            p_SrcBis,
            p_HeightBis,
            p_WidthBis,
            p_StartY,
            p_StartX,
            p_MaxElements,
            l_LogEpsilonThreshold,
            p_NoiseLevel,
            l_PixelState.data(),
            l_FillBoxes
        );
    };

    MaxPool2D<5, 5>(p_Src, p_Height, p_Width, l_OnLocaLMax);

    //std::cerr << "Merged: " << l_DebugCounters.m_NumMerge
    //          << " Levels: " << l_DebugCounters.m_NumLevel
    //          << " Actual merged: " << l_DebugCounters.m_NumRetained
    //          << "\n";
}

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
)
{
    thread_local std::vector<SPixelState<true, T>> l_PixelState;
    l_PixelState.clear();
    l_PixelState.resize(p_Height * p_Width);

    double l_LogEpsilonThreshold = std::log(p_EpsilonThreshold);
    auto l_FillPixels = [&](
        const SLevelSetInfo<T>& p_LLInfo,
        const std::vector<SPixelData<T>>& p_PixelData,
        size_t p_StartY,
        size_t p_StartX,
        double p_LogNFA
    )
    {
        o_DetectionPixelStartIdx.push_back(o_Pixels.size());
        o_DetectionLogNFA.push_back(p_LogNFA);
        for (std::size_t l_Index = 0; l_Index <= p_LLInfo.m_StopIdx; ++l_Index)
        {
            o_Pixels.emplace_back(
                p_PixelData[l_Index].m_Y,
                p_PixelData[l_Index].m_X
            );
        }
    };

    std::function<void(const T*, std::size_t, std::size_t, std::size_t, std::size_t)>
        l_OnLocaLMax = [&](
            const T* p_SrcBis,
            std::size_t p_HeightBis,
            std::size_t p_WidthBis,
            std::size_t p_StartY,
            std::size_t p_StartX
    )
    {
        FindMeaningFulLevelSets<true, T>(
            p_SrcBis,
            p_HeightBis,
            p_WidthBis,
            p_StartY,
            p_StartX,
            p_MaxElements,
            l_LogEpsilonThreshold,
            p_NoiseLevel,
            l_PixelState.data(),
            l_FillPixels
        );
    };

    MaxPool2D<5, 5>(p_Src, p_Height, p_Width, l_OnLocaLMax);
}

template <typename T>
void RemoveSmallSets(
    T* o_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
)
{
    thread_local std::vector<SPixelState<true, T>> l_PixelState;
    l_PixelState.clear();
    l_PixelState.resize(p_Height * p_Width);

    // Initialize output with copy of input
    std::memcpy(o_Dst, p_Src, p_Height * p_Width * sizeof(T));

    std::function<void(const T*, std::size_t, std::size_t, std::size_t, std::size_t)>
        l_OnLocalMax = [&](
            const T* p_SrcBis,
            std::size_t p_HeightBis,
            std::size_t p_WidthBis,
            std::size_t p_StartY,
            std::size_t p_StartX
    )
    {
        RemoveSmallLevelSets<true, T>(
            p_SrcBis,
            p_HeightBis,
            p_WidthBis,
            p_StartY,
            p_StartX,
            o_Dst,
            p_MaxElements,
            l_PixelState.data()
        );
    };

    // NOTE: for exact removal of small sets, we would need to use a +-shaped pooling here instead (3x3 cross)
    MaxPool2D<5, 5>(p_Src, p_Height, p_Width, l_OnLocalMax);
}

template
void FindSpotReturnBoxes<uint8_t>(
    std::vector<SSpotBox>& o_Boxes,
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void FindSpotReturnBoxes<uint16_t>(
    std::vector<SSpotBox>& o_Boxes,
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void FindSpotReturnBoxes<float>(
    std::vector<SSpotBox>& o_Boxes,
    const float* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void FindSpotReturnPixelLists<uint8_t>(
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
void FindSpotReturnPixelLists<uint16_t>(
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
void FindSpotReturnPixelLists<float>(
    std::vector<SPixelCoord>& o_Pixels,
    std::vector<std::size_t>& o_DetectionPixelStartIdx,
    std::vector<double>& o_DetectionLogNFA,
    const float* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements,
    double p_EpsilonThreshold,
    double p_NoiseLevel
);

template
void RemoveSmallSets<uint8_t>(
    uint8_t* o_Dst,
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);

template
void RemoveSmallSets<uint16_t>(
    uint16_t* o_Dst,
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);

template
void RemoveSmallSets<float>(
    float* o_Dst,
    const float* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::size_t p_MaxElements
);