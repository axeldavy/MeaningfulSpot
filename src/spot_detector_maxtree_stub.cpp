// This file defines a fallback to compile against when
// the Max-Tree implementation is not available. It does
// fall back all calls to the non-Max-Tree implementation.

#include "spot_detector.hpp"
#include "spot_detector_maxtree.hpp"


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
    
    FindSpotReturnBoxes<T>(o_Boxes, p_Src, p_Height, p_Width, p_MaxElements, 
                           p_EpsilonThreshold, p_NoiseLevel);
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
    FindSpotReturnPixelLists<T>(
        o_Pixels, o_DetectionPixelStartIdx, o_DetectionLogNFA,
        p_Src, p_Height, p_Width, p_MaxElements,
        p_EpsilonThreshold, p_NoiseLevel);
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
    RemoveSmallSets(
        o_Dst, p_Src, p_Height, p_Width, p_MaxElements
    );
}

template <typename T>
void ComputeDeltaStatistics(
    std::vector<SLevelStat<T>>& r_Dst,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width
)
{}

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

