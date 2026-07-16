// This files defines AlignedBitArray2D which is used to track
// seen pixels in the local level set search implementation.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

/// <summary>
/// Optimized bit-packed 2D array aligned to uint64_t boundaries
/// with column padding for efficient vertical neighbor access.
/// 
/// When accessing (x, y-1), (x, y), (x, y+1), the same bit mask
/// can be reused, enabling compiler optimization.
///
/// Stride is rounded up to the next power of 2, allowing y*stride
/// to be computed as y << m_StrideShift instead of multiplication.
/// </summary>
class AlignedBitArray2D
{
private:
    std::vector<uint64_t> m_Data;
    std::size_t m_Width;           // Actual image width
    std::size_t m_Height;          // Actual image height
    std::size_t m_StrideU64;       // Number of uint64_t per row (padded, power of 2)
    unsigned int m_StrideShift;    // log2(m_StrideU64) for bit shifting

    /// <summary>
    /// Round up to the next power of 2
    /// </summary>
    static std::size_t roundUpPowerOf2(std::size_t p_Value)
    {
        if (p_Value <= 1) return 1;
        --p_Value;
        p_Value |= p_Value >> 1;
        p_Value |= p_Value >> 2;
        p_Value |= p_Value >> 4;
        p_Value |= p_Value >> 8;
        p_Value |= p_Value >> 16;
        p_Value |= p_Value >> 32;
        return p_Value + 1;
    }

    /// <summary>
    /// Calculate log2 of a power of 2 value
    /// </summary>
    static unsigned int log2PowerOf2(std::size_t p_Value)
    {
        unsigned int l_Shift = 0;
        while ((1ULL << l_Shift) < p_Value) ++l_Shift;
        return l_Shift;
    }

public:
    AlignedBitArray2D() : m_Width(0), m_Height(0), m_StrideU64(0), m_StrideShift(0) {}

    /// <summary>
    /// Resize the array to accommodate width x height bits
    /// Stride is padded to the next power of 2 uint64_t values
    /// </summary>
    void resize(std::size_t p_Width, std::size_t p_Height)
    {
        m_Width = p_Width;
        m_Height = p_Height;
        
        // Calculate how many uint64_t are needed per row (minimum)
        std::size_t l_MinStrideU64 = (p_Width + 63) / 64;
        
        // Round up to next power of 2
        m_StrideU64 = roundUpPowerOf2(l_MinStrideU64);
        m_StrideShift = log2PowerOf2(m_StrideU64);
        
        // Total size: height * stride
        std::size_t l_TotalSize = m_Height * m_StrideU64;
        
        if (m_Data.size() < l_TotalSize)
        {
            m_Data.resize(l_TotalSize, 0);
        }
    }

    /// <summary>
    /// Set bit at position (x, y) to 1
    /// </summary>
    inline void set(std::size_t p_X, std::size_t p_Y)
    {
        const std::size_t l_BitIdx = p_X & 63; // p_X % 64
        const std::size_t l_U64Idx = (p_Y << m_StrideShift) + (p_X >> 6);
        m_Data[l_U64Idx] |= (1ULL << l_BitIdx);
    }

    /// <summary>
    /// Reset bit at position (x, y) to 0
    /// </summary>
    inline void reset(std::size_t p_X, std::size_t p_Y)
    {
        const std::size_t l_BitIdx = p_X & 63; // p_X % 64
        const std::size_t l_U64Idx = (p_Y << m_StrideShift) + (p_X >> 6);
        m_Data[l_U64Idx] &= ~(1ULL << l_BitIdx);
    }

    /// <summary>
    /// Reset the block of bits covering p_X, p_Y (entire uint64_t)
    /// </summary>
    inline void resetBlock(std::size_t p_X, std::size_t p_Y)
    {
        const std::size_t l_U64Idx = (p_Y << m_StrideShift) + (p_X >> 6);
        m_Data[l_U64Idx] = 0;
    }

    /// <summary>
    /// Get bit at position (x, y)
    /// Returns true if bit is set, false otherwise
    /// </summary>
    inline bool get(std::size_t p_X, std::size_t p_Y) const
    {
        const std::size_t l_BitIdx = p_X & 63; // p_X % 64
        const std::size_t l_U64Idx = (p_Y << m_StrideShift) + (p_X >> 6);
        return (m_Data[l_U64Idx] >> l_BitIdx) & 1ULL;
    }

    /// <summary>
    /// Clear all bits (set to 0) without deallocating
    /// </summary>
    inline void clear()
    {
        std::fill(m_Data.begin(), m_Data.end(), 0);
    }
};