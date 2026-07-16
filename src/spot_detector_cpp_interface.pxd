# This file defines the interface of the C++ methods
# we want to use from Cython.

from libcpp.vector cimport vector

cdef extern from "maxpool.hpp" nogil:
    void MaxPool2DExport[KH, KW, T](
        vector[size_t] &,
        const T*,
        size_t,
        size_t
    )

# Cython doesn't seem to like size_t in templates
cdef extern from * nogil:
    """
template <typename T>
inline bool MaxPool2DExportDispatch(
    std::vector<size_t>& p_Out,
    const T* p_Src,
    size_t p_Height,
    size_t p_Width,
    size_t p_KernelHeight,
    size_t p_KernelWidth
)
{
    if (p_KernelHeight == 3 && p_KernelWidth == 3)
    {
        MaxPool2DExport<3, 3>(
            p_Out,
            p_Src,
            p_Height,
            p_Width
        );
        return true;
    } else if (p_KernelHeight == 5 && p_KernelWidth == 5)
    {
        MaxPool2DExport<5, 5>(
            p_Out,
            p_Src,
            p_Height,
            p_Width
        );
        return true;
    } else if (p_KernelHeight == 7 && p_KernelWidth == 7)
    {
        MaxPool2DExport<7, 7>(
            p_Out,
            p_Src,
            p_Height,
            p_Width
        );
        return true;
    }

    // fallthrough:
    return false;
}

    """
    bint MaxPool2DExportDispatch[T](
        vector[size_t] &,
        const T*,
        size_t,
        size_t,
        size_t,
        size_t
    )

cdef extern from "spot_detector.hpp" nogil:
    struct SSpotBox:
        size_t m_Y
        size_t m_X
        size_t m_Height
        size_t m_Width
        size_t m_NumPixels
        double m_Score

    struct SPixelCoord:
        size_t m_Y
        size_t m_X

    void FindSpotReturnBoxes[T](
        vector[SSpotBox]& o_Boxes,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width,
        size_t p_MaxElements,
        double p_EpsilonThreshold,
        double p_NoiseLevel
    )

    void FindSpotReturnPixelLists[T](
        vector[SPixelCoord]& o_Pixels,
        vector[size_t]& o_DetectionPixelStartIdx,
        vector[double]& o_DetectionLogNFA,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width,
        size_t p_MaxElements,
        double p_EpsilonThreshold,
        double p_NoiseLevel
    )

    void RemoveSmallSets[T](
        T* o_Dst,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width,
        size_t p_MaxElements
    )

cdef extern from "spot_detector_maxtree.hpp" nogil:
    void FindSpotReturnBoxesMaxtree[T](
        vector[SSpotBox]& o_Boxes,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width,
        size_t p_MaxElements,
        double p_EpsilonThreshold,
        double p_NoiseLevel
    )

    void FindSpotReturnPixelListsMaxtree[T](
        vector[SPixelCoord]& o_Pixels,
        vector[size_t]& o_DetectionPixelStartIdx,
        vector[double]& o_DetectionLogNFA,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width,
        size_t p_MaxElements,
        double p_EpsilonThreshold,
        double p_NoiseLevel
    )

    void RemoveSmallLevelSetsMaxtree[T](
        T* o_Dst,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width,
        size_t p_MaxElements
    )

    cppclass SLevelStat[T]:
        T m_Level
        T m_ParentLevel
        size_t m_NumPixels

    void ComputeDeltaStatistics[T](
        vector[SLevelStat[T]]& r_Dst,
        const T* p_Src,
        size_t p_Height,
        size_t p_Width
    )
