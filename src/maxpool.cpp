// This file implements MaxPool2D to find local intensity maxes.

#include <cstdint>
#include <functional>
#include <xsimd/xsimd.hpp>
#include <vector>

#include "maxpool.hpp"

#include <iostream>

// <summary> Perform KHxKW maxpooling and call the function on maxes </summary>
// <note> p_OnMax: src, height, width, y, x </note>
// <note> Ignores border pixels that do not fit in the kernel </note>
template <std::size_t KH, std::size_t KW, typename T>
void MaxPool2D(
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const T*, size_t, size_t, size_t, size_t)> p_OnMax
)
{
    // Use largest available SIMD
    using batch = xsimd::batch<T>;
    // Use a step size the number of items in the SIMD register
    constexpr std::ptrdiff_t l_Step = batch::size;

    // kernel half size (centers)
    constexpr std::ptrdiff_t KH_2 = KH / 2;
    constexpr std::ptrdiff_t KW_2 = KW / 2;

    // sized arithmetics
    constexpr std::ptrdiff_t KWS = KW;
    constexpr std::ptrdiff_t KHS = KH;

    // Check dimensions
    static_assert(KH > 0);
    static_assert(KW > 0);
    if (p_Height < KH)
        return;
    if (p_Width < KW)
        return;

    // Last indices with valid elements
    const std::ptrdiff_t l_LastValidY = (p_Height - KH) + KH_2;
    const std::ptrdiff_t l_LastValidX = (p_Width - KW) + KW_2;

    for (std::ptrdiff_t l_Y = KH_2; l_Y <= l_LastValidY; ++l_Y)
    {
        // Process a row until the SIMD width doesn't fit anymore
        std::ptrdiff_t l_BatchX = KW_2;
        for(; l_BatchX <= (l_LastValidX+1-l_Step); l_BatchX += l_Step)
        {
            // Perform 1xKW maxes
            // Here we assume KH is small, else using
            // a temporary buffer and two passes should
            // perform faster.
            batch l_RowMaxes[KH];
            for (std::ptrdiff_t l_RowDelta = -KH_2; l_RowDelta < KHS - KH_2; ++l_RowDelta)
            {
                batch l_LocalMax = batch::load_unaligned(
                    &p_Src[p_Width * (l_Y + l_RowDelta) + l_BatchX + (-KW_2)]
                );
                // We start from -KW_2+1 because we already loaded KW_2 above
                for (std::ptrdiff_t l_ColDelta = -KW_2 + 1; l_ColDelta < KWS - KW_2; ++l_ColDelta)
                {
                    batch l_LocalValue = batch::load_unaligned(
                        &p_Src[p_Width * (l_Y + l_RowDelta) + l_BatchX + l_ColDelta]
                    );
                    // Update l_LocalMax
                    l_LocalMax = xsimd::max(l_LocalMax, l_LocalValue);
                }
                l_RowMaxes[l_RowDelta + KH_2] = l_LocalMax;
            }

            // Perform the max of these intermediate values
            batch l_KernelMax = l_RowMaxes[0];
            for (std::ptrdiff_t l_RoxMaxesIdx = 0; l_RoxMaxesIdx < KHS; ++l_RoxMaxesIdx)
            {
                // Update l_KernelMax
                l_KernelMax = xsimd::max(l_KernelMax, l_RowMaxes[l_RoxMaxesIdx]);
            }

            // Check if CenterValue is >= to the max
            batch l_CenterValue = batch::load_unaligned(
                &p_Src[p_Width * l_Y + l_BatchX]
            );
            xsimd::batch_bool l_IsKernelMax = xsimd::ge(l_CenterValue, l_KernelMax);

            // If any is its local max, we call the functional
            if (xsimd::any(l_IsKernelMax))
            {
                bool l_IsKernelMaxArray[l_Step] = {};
                l_IsKernelMax.store_unaligned(l_IsKernelMaxArray);
                for (std::ptrdiff_t l_ItemIdx = 0; l_ItemIdx < l_Step; ++l_ItemIdx)
                {
                    if (l_IsKernelMaxArray[l_ItemIdx])
                    {
                        p_OnMax(p_Src, p_Height, p_Width, l_Y, l_BatchX + l_ItemIdx);
                    }
                }
            }
        }
        // Treat remaining pixels at the end of the row
        for (std::ptrdiff_t l_ItemX = l_BatchX; l_ItemX <= l_LastValidX; ++l_ItemX)
        {
            T l_CenterValue = p_Src[p_Width * l_Y + l_ItemX];
            T l_MaxValue = l_CenterValue;
            for (std::ptrdiff_t l_RowDelta = -KH_2; l_RowDelta < KHS - KH_2; ++l_RowDelta)
            {
                for (std::ptrdiff_t l_ColDelta = -KW_2; l_ColDelta < KWS - KW_2; ++l_ColDelta)
                {
                    T l_LocalValue = p_Src[p_Width * (l_Y + l_RowDelta) + l_ItemX + l_ColDelta];
                    if (l_MaxValue < l_LocalValue)
                    {
                        l_MaxValue = l_LocalValue;
                    }
                }
            }
            if (l_CenterValue >= l_MaxValue)
            {
                p_OnMax(p_Src, p_Height, p_Width, l_Y, l_ItemX);
            }
        }
    }
}

template
void MaxPool2D<3, 3, uint8_t>(
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const uint8_t*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<5, 5, uint8_t>(
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const uint8_t*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<7, 7, uint8_t>(
    const uint8_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const uint8_t*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<3, 3, uint16_t>(
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const uint16_t*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<5, 5, uint16_t>(
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const uint16_t*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<7, 7, uint16_t>(
    const uint16_t* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const uint16_t*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<3, 3, float>(
    const float* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const float*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<5, 5, float>(
    const float* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const float*, size_t, size_t, size_t, size_t)> p_OnMax
);

template
void MaxPool2D<7, 7, float>(
    const float* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const float*, size_t, size_t, size_t, size_t)> p_OnMax
);
