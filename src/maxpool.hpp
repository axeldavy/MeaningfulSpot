// This file implements the header for MaxPool2D which is used
// to find local intensity maxes.

#pragma once

#include <functional>
#include <vector>

// NOTE: to reduce computation on totally flat regions, we could
// only raise onMax when the center pixel is strictly greater
// than at least one of its neighbors.

/// <summary>
/// Perform KH x KW max pooling and call a function on each local maximum.
///
/// For each pixel in the valid region (excluding borders), computes the
/// maximum value in a KH x KW kernel centered on that pixel. If the center
/// pixel value is >= the kernel maximum, calls p_OnMax with that position.
///
/// Uses SIMD optimization for improved performance on large images.
///
/// </summary>
/// <typeparam name="KH">Kernel height (must be odd)</typeparam>
/// <typeparam name="KW">Kernel width (must be odd)</typeparam>
/// <typeparam name="T">Type of pixel data</typeparam>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <param name="p_OnMax">
/// Callback function(src, height, width, y, x) called for each local maximum
/// </param>
/// <note> Ignores border pixels that do not fit in the kernel </note>
/// <note> If multiple pixels have the same maximum value, all are reported </note>
template <std::size_t KH, std::size_t KW, typename T>
void MaxPool2D(
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width,
    std::function<void(const T*, size_t, size_t, size_t, size_t)> p_OnMax
);

/// <summary>
/// Perform KH x KW max pooling and export linear indices of local maxima.
///
/// Wrapper around MaxPool2D that collects the linear indices (y * width + x)
/// of all local maxima into an output vector.
///
/// </summary>
/// <typeparam name="KH">Kernel height (must be odd)</typeparam>
/// <typeparam name="KW">Kernel width (must be odd)</typeparam>
/// <typeparam name="T">Type of pixel data</typeparam>
/// <param name="o_Out">
/// Output vector that will contain linear indices of local maxima
/// </param>
/// <param name="p_Src"> Input 2D image (C contiguous) </param>
/// <param name="p_Height"> Height of the 2D image </param>
/// <param name="p_Width"> Width of the 2D image </param>
/// <note> o_Out is cleared before filling </note>
template <std::size_t KH, std::size_t KW, typename T>
inline void MaxPool2DExport(
    std::vector<std::size_t> &o_Out,
    const T* p_Src,
    std::size_t p_Height,
    std::size_t p_Width)
{
    o_Out.clear();

    auto l_FillOut = [&](const T*, size_t, size_t, size_t x, size_t y)
    {
        o_Out.push_back(y * p_Width + x);
    };

    MaxPool2D<KH, KW, T>(
        p_Src,
        p_Height,
        p_Width,
        l_FillOut
    );
}
