// Implementation of:
//	W. Purgathofer, R. F. Tobler and M. Geiler.
//	"Forced random dithering: improved threshold matrices for ordered dithering"
//	Image Processing, 1994. Proceedings. ICIP-94., IEEE International Conference,
//	Austin, TX, 1994, pp. 1032-1035 vol.2.
//	doi: 10.1109/ICIP.1994.413512
//
// https://books.google.dk/books?id=ekGjBQAAQBAJ&pg=PA297&lpg=PA297&dq=Forced+random+dithering:+Improved+threshold+matrices+for+ordered+dithering&source=bl&ots=Zmo3T9AA6M&sig=oXmYNokR6yWVXEiMO87gsXS-5u0&hl=en&sa=X&ved=0ahUKEwis86Hl9enRAhWiYJoKHbllAy8Q6AEINjAG#v=onepage&q=Forced%20random%20dithering%3A%20Improved%20threshold%20matrices%20for%20ordered%20dithering&f=false
//
// original author Wojciech Jarosz
// https://bitbucket.org/wkjarosz/hdrview/src/master/
//
// Copyright (c) 2017 Wojciech Jarosz. All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// You are under no obligation whatsoever to provide any bug fixes, patches, or
// upgrades to the features, functionality or performance of the source code
// ("Enhancements") to anyone; however, if you choose to make your Enhancements
// available either publicly, or directly to the authors of this software, without
// imposing a separate written license agreement for such Enhancements, then you
// hereby grant the following license: a non-exclusive, royalty-free perpetual
// license to install, use, modify, prepare derivative works, incorporate into
// other computer software, distribute, and sublicense such enhancements or
// derivative works thereof, in binary and source code form.
//////////////////////////////////////////////////////////////////////////////////////////////////////


#include <string>
#include <sstream>
#include <iostream>

#include <thread>
#include <chrono>

#include <assert.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//=============================================================================

typedef float float32_t;
typedef double float64_t;

static_assert (sizeof(int8_t) == 1, "int8_t");
static_assert (sizeof(int16_t) == 2, "int16_t");
static_assert (sizeof(int32_t) == 4, "int32_t");
static_assert (sizeof(int64_t) == 8, "int64_t");

static_assert (sizeof(uint8_t) == 1,  "uint8_t");
static_assert (sizeof(uint16_t) == 2, "uint16_t");
static_assert (sizeof(uint32_t) == 4, "uint32_t");
static_assert (sizeof(uint64_t) == 8, "uint64_t");

static_assert (sizeof(float32_t) == 4, "float32_t");
static_assert (sizeof(float64_t) == 8, "float64_t");

enum { ALLOC_ALIGNMENT = 16 };

//=============================================================================

struct vec2i
{
	int32_t x, y;
};
bool operator==(const vec2i &a, const vec2i &b)
{
	return a.x == b.x && a.y == b.y;
}

int32_t min( int32_t a, int32_t b )
{
	return (a<b) ? a : b;
}
int32_t max( int32_t a, int32_t b )
{
	return (a>b) ? a : b;
}

int32_t coord2idx( int x, int y, int32_t siz )
{
	assert(x < siz && y < siz );
	return siz * y + x;
}
int32_t coord2idx( vec2i v, int32_t siz )
{
	return coord2idx(v.x,v.y,siz);
}

// ====
float64_t toroidalMinimumDistance( const vec2i &a, const vec2i &b, int32_t siz )
{
	int32_t x0 = min(a.x, b.x);
	int32_t x1 = max(a.x, b.x);
	int32_t y0 = min(a.y, b.y);
	int32_t y1 = max(a.y, b.y);
	float64_t deltaX = (float64_t)min(x1-x0, x0+siz-x1);
	float64_t deltaY = (float64_t)min(y1-y0, y0+siz-y1);
	return sqrt( deltaX*deltaX + deltaY*deltaY );
}


//note: https://nullprogram.com/blog/2018/07/31/
//note: bias: 0.020888578919738908 = minimal theoretic limit
class hash_wellon
{
public:
	uint32_t seed = 1337u;

public:
	uint32_t triple32( uint32_t x )
	{
		x ^= x >> 17;
		x *= 0xed5ad4bbU;
		x ^= x >> 11;
		x *= 0xac4c1b51U;
		x ^= x >> 15;
		x *= 0x31848babU;
		x ^= x >> 14;
		return x;
	}
	uint32_t iter()
	{
		seed = triple32( seed );
		return seed;
	}

//public:
//	typedef uint32_t result_type;
//	static uint32_t min() { return 0; }
//	static uint32_t max() { return UINT32_MAX; }
//	uint32_t operator()()
//	{
//		return iter();
//	}
};

struct parms_thread_t
{
	float64_t *M = {nullptr};
	hash_wellon rnd;
	int32_t threadid = {-1};
	int32_t img_siz = {128};
};

//=============================================================================

// ====
//TODO: test combined normalized and integer force? https://www.solidangle.com/research/dither_abstract.pdf
// exp( -pow(r/s,p); // s=0.5, p=0.5
float64_t force( const float64_t r )
{
	return exp(-sqrt(2.0*r));
	//return exp1024(-sqrt(2.0*r));
}
//} //namespace


//=============================================================================

std::string SecondsToHumanReadable( int im )
{
	std::ostringstream outString;
	int h = im / (60 * 60);
	int m = (im / 60) % 60;
	int s = im % 60;
	if ( h > 0 ) outString << h << "h";
	if ( m > 0 ) outString << m << "m";
	if ( s > 0 ) outString << s << "s";
	return outString.str();
}

//TODO: input maxcount and num-updates
void timestamp( int i, int n, const std::chrono::milliseconds &start_time_ms )
{
	if ( i > 0 && i % 1024 == 0 )
	{
		std::chrono::milliseconds cur_t_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() );
		std::chrono::milliseconds dt_ms = cur_t_ms - start_time_ms;
		int est_total_s = static_cast<int>( (dt_ms.count() / static_cast<float32_t>( i ) * n) / 1000 );
		int est_cur_s = static_cast<int>( est_total_s - dt_ms.count() / 1000 );
		printf( "%d/%d, eta %s (est-total %s)\n", i, n, SecondsToHumanReadable(est_cur_s).c_str(), SecondsToHumanReadable(est_total_s).c_str() );
	}
}

//=============================================================================


//note: fisher_yates
void do_shuffle( vec2i * const data, const int32_t siz, hash_wellon &hash )
{
	for ( int i=siz-1,n=siz; i>0; --i )
	{
		int32_t j = hash.iter() % i;

		vec2i tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}
}

// ====
void gen2D( parms_thread_t *pt )
{
	std::chrono::milliseconds start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() );

	printf( "init 2D (%dx%d)\n", pt->img_siz, pt->img_siz );

	memset( pt->M, 0, pt->img_siz * pt->img_siz * sizeof(float64_t) );

	float64_t *forceField = (float64_t*)_aligned_malloc( pt->img_siz*pt->img_siz*sizeof(float64_t), ALLOC_ALIGNMENT );
	memset( forceField , 0, pt->img_siz*pt->img_siz*sizeof(float64_t) );

	int32_t freeLocations_count = pt->img_siz * pt->img_siz;
	vec2i *freeLocations = (vec2i*)_aligned_malloc( freeLocations_count*sizeof(vec2i), ALLOC_ALIGNMENT );

	//note: initialize free locations
	for ( int y = 0; y<pt->img_siz; ++y ) {
	for ( int x = 0; x<pt->img_siz; ++x ) {
		int idx = coord2idx(x,y, pt->img_siz);
		freeLocations[ idx ] = vec2i{x,y};
	}}

	printf( "calc 2D (%dx%d)\n", pt->img_siz, pt->img_siz);

	for (int32_t ditherValue = 0, ditherValueCount=pt->img_siz*pt->img_siz; ditherValue < ditherValueCount; ++ditherValue)
	{
		do_shuffle( freeLocations, freeLocations_count, pt->rnd );
		
		float64_t minimum = DBL_MAX;
		vec2i minimumLocation = { -1, -1 };

		int halfP = min( freeLocations_count, max(1, (int)sqrt(freeLocations_count*3/4)) );
		for ( int i = 0, n = halfP; i<n; ++i )
		{
			const vec2i &location = freeLocations[i];
			int idx = coord2idx( location, pt->img_siz );
			float64_t ff = forceField[ idx ];
			if ( ff < minimum )
			{
				minimum = ff;
				minimumLocation = location;
			}
		}
		assert( minimumLocation.x >= 0 && minimumLocation.y >= 0 );
		const int32_t minimumLocationIdx = coord2idx(minimumLocation, pt->img_siz );

		vec2i cell;
		for ( cell.y = 0; cell.y < pt->img_siz; ++cell.y ) {
		for ( cell.x = 0; cell.x < pt->img_siz; ++cell.x ) {
			float64_t r = toroidalMinimumDistance(cell, minimumLocation, pt->img_siz);
			int idx = coord2idx( cell, pt->img_siz );
			forceField[ idx ] += force(r);
		}}

		for ( int i=0,n=freeLocations_count; i<n; ++i )
		{
			if ( freeLocations[i] == minimumLocation )
			{
				freeLocations[i] = freeLocations[ --freeLocations_count];
				break; //note: there can be only one
			}
		}

		assert( pt->M[ minimumLocationIdx ] == 0.0 );
		pt->M[ minimumLocationIdx ] = (float64_t)ditherValue;

		if ( pt->threadid == 0 )
			timestamp( ditherValue, ditherValueCount, start_time_ms );
	}

	if ( pt->threadid == 0 )
	{
		std::chrono::milliseconds time_post_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() );
		printf( "calc-time: %lldms\n", time_post_ms.count() - start_time_ms.count() );
	}

	_aligned_free( freeLocations );
	_aligned_free( forceField );
}

//=============================================================================

float32_t signf( const float32_t v )
{
	if (v == 0.0f) return 0.0f;
	return (v > 0.0f) ? 1.0f : -1.0f;
}

//note: see https://github.com/Unity-Technologies/ScriptableRenderPipeline/blob/master/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/FinalPass.shader#L139
float32_t remap_noise_tri_unity( float32_t v )
{
	v = v*2.0f - 1.0f;
	v = signf(v) * (1.0f - sqrtf(1.0f - fabs(v)));
	//return v;
	return v*0.5f + 0.5f; //[0;1]
}

//=============================================================================

void write_to_disk( float64_t const * const frd[4], const int32_t siz )
{
	const int32_t elemcount = siz * siz;

	unsigned char* bytedata = new unsigned char[ 4 * elemcount ];
	unsigned char* bytedata_tpdf = new unsigned char[ 4 * elemcount ];
	float32_t *floatdata = new float32_t[ 4 * elemcount ];

	for ( int i = 0, n = elemcount; i<n; ++i )
	{
		for ( int c=0;c<4;++c)
		{
			float32_t vf = static_cast<float32_t>( frd[c][i] ) / static_cast<float32_t>( elemcount );
			floatdata[4*i+c] = vf;
			bytedata[4*i+c] = static_cast<uint8_t>( vf * 256.0f );
			bytedata_tpdf[4*i+c] = static_cast<uint8_t>( remap_noise_tri_unity( vf ) * 256.0f );
		}
	}

	//note: check for uniformity in uniform distribution
	{
		int32_t histogram_buckets[256] = {0};
		for ( int i = 0, n = elemcount; i<n; ++i )
		{
			uint8_t b = bytedata[4*i+0]; //note: only checks first channel
			histogram_buckets[b] += 1;
		}
		int32_t mn=INT_MAX, mx=INT_MIN;
		for ( int i=0,n=256;i<n;++i)
		{
			int32_t b = histogram_buckets[i];
			mn = min( mn, b );
			mx = max( mx, b );
		}
		printf( "uniform histogram num-buckets: min=%d, max=%d\n", mn, mx );
	}

	char filename[512]; memset(filename, 0, 512);

	sprintf_s(filename, 512, "bluenoise_frd_%dx%d_uni.hdr", siz, siz);
	stbi_write_hdr( filename, siz, siz, 4, floatdata);
	std::cout << "wrote " << filename << std::endl;

	sprintf_s(filename, 512, "bluenoise_frd_%dx%d_uni.bmp", siz, siz);
	stbi_write_bmp(filename, siz, siz, 4, bytedata);
	std::cout << "wrote " << filename << std::endl;

	sprintf_s(filename, 512, "bluenoise_frd_%dx%d_tri.bmp", siz, siz);
	stbi_write_bmp(filename, siz, siz, 4, bytedata_tpdf);
	std::cout << "wrote " << filename << std::endl;

	delete[] bytedata;
	delete[] bytedata_tpdf;
	delete[] floatdata;
}


int main( int argc, char **argv )
{
	int32_t siz = 128;
	if ( argc > 1 )
		siz = atoi( argv[1] );

	float64_t *frd[4];
	frd[0] = (float64_t*)_aligned_malloc( siz*siz*sizeof(float64_t), ALLOC_ALIGNMENT );
	frd[1] = (float64_t*)_aligned_malloc( siz*siz*sizeof(float64_t), ALLOC_ALIGNMENT );
	frd[2] = (float64_t*)_aligned_malloc( siz*siz*sizeof(float64_t), ALLOC_ALIGNMENT );
	frd[3] = (float64_t*)_aligned_malloc( siz*siz*sizeof(float64_t), ALLOC_ALIGNMENT );

	parms_thread_t pt[4];
	for(int i=0;i<4;++i)
	{
		pt[i].threadid = i;
		pt[i].img_siz = siz;
		pt[i].M = frd[i];
	}
	pt[0].rnd.seed = 83u;
	pt[1].rnd.seed = 1901u;
	pt[2].rnd.seed = 8159467u;
	pt[3].rnd.seed = 15481607u;

	std::thread tr( gen2D, &pt[0] );
	std::thread tg( gen2D, &pt[1] );
	std::thread tb( gen2D, &pt[2] );
	std::thread ta( gen2D, &pt[3] );
	tr.join();
	tg.join();
	tb.join();
	ta.join();

	write_to_disk( frd, siz );

	_aligned_free( frd[0] );
	_aligned_free( frd[1] );
	_aligned_free( frd[2] );
	_aligned_free( frd[3] );

	return 0;
}
