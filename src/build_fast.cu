#include "build.h"
#include "bvh.h"
#include "util.h"
#include "vec_math_helper.h"

#include <cub/device/device_radix_sort.cuh>
#include <sutil/vec_math.h>

/* \brief Interleave the first 10 bits of x every three bits,
	 * ie insert two zeroes between every of the first 10 bits of x
	 * \param x Quantitized position, must be between 0 and 2^10 - 1 = 1023
	 */
	__device__ __forceinline__ uint32_t InterleaveBits32(uint32_t x)
	{
		/* Comments generated with Python from https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit */

		/*
		 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 1111 1111
		 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000  hex: 0x300
		 * Shifted part (<< 16):   0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 0000 0000  hex: 0x3000000
		 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111  hex: 0xff
		 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 1111 1111  hex: 0x30000ff
		 */
		x = (x | (x << 16)) & 0x30000ff;

		/*
		 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 1111 1111
		 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 0000  hex: 0xf0
		 * Shifted part (<< 8):    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 0000 0000 0000  hex: 0xf000
		 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0000 0000 0000 1111  hex: 0x300000f
		 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 1111 0000 0000 1111  hex: 0x300f00f
		 */
		x = (x | (x << 8)) & 0x300f00f;

		/*
		 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 1111 0000 0000 1111
		 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 1100  hex: 0xc00c
		 * Shifted part (<< 4):    0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 1100 0000  hex: 0xc00c0
		 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 0000 0011 0000 0000 0011  hex: 0x3003003
		 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 1100 0011 0000 1100 0011  hex: 0x30c30c3
		 */
		x = (x | (x << 4)) & 0x30c30c3;

		/*
		 * Current Mask:           0000 0000 0000 0000 0000 0000 0000 0000 0000 0011 0000 1100 0011 0000 1100 0011
		 * Which bits to shift:    0000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0000 1000 0010 0000 1000 0010  hex: 0x2082082
		 * Shifted part (<< 2):    0000 0000 0000 0000 0000 0000 0000 0000 0000 1000 0010 0000 1000 0010 0000 1000  hex: 0x8208208
		 * NonShifted Part:        0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 0000 0100 0001 0000 0100 0001  hex: 0x1041041
		 * Bitmask is now :        0000 0000 0000 0000 0000 0000 0000 0000 0000 1001 0010 0100 1001 0010 0100 1001  hex: 0x9249249
		 */
		x = (x | (x << 2)) & 0x9249249;

		return x;
	}

    /* \brief Compute a 32-bit Morton code for the given quantitized 3D point
	 * \param x The quantitized x coordinate
	 * \param y The quantitized y coordinate
	 * \param z The quantitized z coordinate
	 */
	__device__ __forceinline__ uint32_t MortonCode32(uint32_t x, uint32_t y, uint32_t z)
	{
		return InterleaveBits32(x) | InterleaveBits32(y) << 1 | InterleaveBits32(z) << 2;
	}