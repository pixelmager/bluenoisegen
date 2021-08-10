#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

#include <assert.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//=============================================================================


typedef float float32_t;
typedef double float64_t;

enum { ALLOC_ALIGNMENT = 16 };

//#define SHOW_DBG_HISTOGRAM

//=============================================================================


// https://nullprogram.com/blog/2018/07/31/
// bias: 0.020888578919738908 = minimal theoretic limit
uint32_t triple32(uint32_t x)
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

int32_t min( int32_t a, int32_t b )
{
	return (a<b) ? a : b;
}
int32_t max( int32_t a, int32_t b )
{
	return (a>b) ? a : b;
}
float32_t min( float32_t a, float32_t b )
{
	return (a<b) ? a : b;
}
float32_t max( float32_t a, float32_t b )
{
	return (a>b) ? a : b;
}

float32_t signf( const float32_t v )
{
	if (v == 0.0f) return 0.0f;
	return (v > 0.0f) ? 1.0f : -1.0f;
	
	//f32 ret_opt = (((*reinterpret_cast<u32*>(&v)) & (1<<31) ) == 0) ? 1.0f : -1.0f;
	//assert( ret_opt == (v > 0 ? 1.0f : -1.0f), "" );
	//return ret_opt;		
}
	
float64_t sign( const float64_t v )
{
	if (v == 0.0) return 0.0;
	return (v > 0.0) ? 1.0 : -1.0;
}

//note: see https://github.com/Unity-Technologies/ScriptableRenderPipeline/blob/master/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/FinalPass.shader#L139
float64_t remap_noise_tri_unity( float64_t v )
{
	v = v*2.0-1.0;
	v = sign(v) * (1.0 - sqrt(1.0 - abs(v)));

	//return v;
	return v*0.5 + 0.5; //[0;1]
}


float64_t remap_noise_tri( const float64_t v )
{
	return remap_noise_tri_unity( v );
}

//note: https://nullprogram.com/blog/2018/07/31/
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


void progress( char const * const str, int i, int n, int num_updates = 10, int update_interval_s = 20 )
{
	static std::chrono::milliseconds prev_update_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() );

	std::chrono::milliseconds cur_t_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() );
	std::chrono::milliseconds dt_ms = cur_t_ms - prev_update_ms;

	if ( i % (n/num_updates) == 0 || dt_ms.count()/1000 > update_interval_s )
	{
		printf( "%s progress %d/%d...\n", str, i, n );
		prev_update_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() );
	}
}

//=============================================================================

struct parms_shared_t
{
	int32_t image_siz;
	int32_t pct_random; // [0;100]
	int32_t filter_siz; // uneven, {5, 7, 9..}
};

struct parms_thread_t : public parms_shared_t
{
	float64_t *dst;
	int32_t threadid = {-1} ;
	uint32_t seed = {1337};
};

// ====
// note: see http://caca.zoy.org/study/
//       code http://caca.zoy.org/study/source.html
class BlueNoise
{
public:

	int32_t m_gaussSize { 7 };

private:
	int32_t m_size { 16 };
	int32_t m_size_sq { 16 * 16 };

	bool *m_mat { nullptr };
	bool *m_matClone { nullptr };

	float32_t *m_filter { nullptr };
	float32_t *m_gaussTmp { nullptr };
	float32_t *m_gauss { nullptr };
	
	int32_t m_ofs { 0 };

	int32_t m_rank { 0 };
	int32_t m_rankClone { 0 };
	int32_t m_max_x { INT_MIN };
	int32_t m_max_y { INT_MIN };
	int32_t m_min_x { INT_MAX };
	int32_t m_min_y { INT_MAX };

public:
	// ====
	void reset()
	{
		m_rank = 0;
		m_rankClone = 0;
		m_max_x = INT_MIN;
		m_max_y = INT_MIN;
		m_min_x = INT_MAX;
		m_min_y = INT_MAX;

		//note: empty boolean matrix
		for(int i = 0, n = m_size_sq; i < n; ++i)
			m_mat[i] = false;

		for(int i = 0,n = m_size_sq; i < n; ++i)
		{
			m_gaussTmp[i] = 0.0f;
			m_gauss[i] = 0.0f;
		}

		PreCalcGaussWeights();
	}

private:

	//TODO: swap y, x? ...calls indicate this
	inline bool& mat( int32_t x, int32_t y ) { return m_mat[ m_size*y + x]; }
	inline float32_t& filter( int32_t x, int32_t y ) { return m_filter[ m_size*y + x ]; }
	inline float32_t& gaussTmp( int32_t x, int32_t y ) { return m_gaussTmp[ m_size*y + x ]; }
	inline float32_t& gauss( int32_t x, int32_t y ) { return m_gauss[ m_size*y + x ]; }

public:
	// ====
	int32_t fact( int32_t x )
	{
		return ( x == 0 ) ? 1 : x * fact( x-1 );
	}

	// ====
	int32_t Binomial( int32_t n, int32_t r )
	{
		return fact(n) / ( fact(r) * fact(n-r) );
	}

	// ====
	//note: create a gaussian blur kernel for later
	void PreCalcGaussWeights()
	{
		assert( m_gaussSize % 2 == 1 && "uneven number of weights required" );

		m_filter = reinterpret_cast<float32_t*>( _aligned_malloc( m_gaussSize * sizeof(float32_t), ALLOC_ALIGNMENT) );
		assert( m_filter != nullptr );

		int32_t row = m_gaussSize-1;
		float32_t sum = static_cast<float32_t>( 1 << row );

		//note: binomial coefficients, e.g. http://colalg.math.csusb.edu/~devel/IT/main/m12_binomial/src/s01_pascal.html
		for(int i=0,n=m_gaussSize/2+1; i<n; ++i )
		{
			int32_t bn = Binomial( row, i );
			m_filter[i] = static_cast<float32_t>( bn ) / sum;
			m_filter[m_gaussSize-1 - i] = m_filter[i];
		}
	}

	// ====
	struct ivec2_t
	{
		int32_t x;
		int32_t y;
		ivec2_t() {}
		ivec2_t( const int32_t in_x, const int32_t in_y ) : x(in_x),y(in_y) {}
	};

	// ====
	// note: ...just randomly picks one. Terrible heuristic :)
	int32_t pick_idx_from_best__random(const std::vector<ivec2_t> &best_list, hash_wellon &hash)
	{
		assert( best_list.size() > 0 );
		float64_t rnd = hash.iter();
		return static_cast<int32_t>((best_list.size() - 1) * rnd);
	}


	// ====
	void Min()
	{
		Filter();

		//note: find lowest number in filtered matrix
		float32_t best = FLT_MAX;
		for ( int y = 0, h=m_size; y < h; ++y )
		{
			for ( int x = 0, w=m_size; x < w; ++x )	
			{
				if ( !mat(x,y) )
				{
					float32_t g = gauss(x,y);
					if ( g < best )
					{
						best = g;
						m_min_x = x;
						m_min_y = y;
					}
				}
			}
		}
		mat( m_min_x, m_min_y ) = true;
	}

	// ====
	void Min_refilter(hash_wellon &hash)
	{
		Filter();

		//note: find lowest number in filtered matrix
		int bestcount = 0;
		int falsecount = 0;
		while ( bestcount != 1 )
		{
			float32_t best = FLT_MAX;
			bestcount = 0;
			falsecount = 0;
			float32_t minval = FLT_MAX;
			float32_t maxval = -FLT_MAX;
			for ( int y = 0, h=m_size; y < h; ++y )
			{
				for ( int x = 0, w=m_size; x < w; ++x )	
				{
					if ( mat(x,y) == false )
					{
						falsecount++;
						float32_t g = gauss(x,y);
						if ( g < best )
						{
							best = g;
							bestcount = 1;
							m_min_x = x;
							m_min_y = y;
						}
						else if ( g == best )
						{
							++bestcount;
						}
						minval = min(g,minval);
						maxval = max(g,maxval);
					}
				}
			}

			if ( bestcount > 1 )
			{
				//note: catch entirely grey images
				if ( minval == maxval )
				{
					//note: pick a random of those... same as min_multiple
					//note: slow, but hit seldomly
					//printf( "# bestcount=%d, falsecount=%d\n", bestcount, falsecount );

					ivec2_t *best_list = new ivec2_t[bestcount];
					int32_t bc = 0;
					for (int y = 0, h = m_size; y < h; ++y)
					{
						for (int x = 0, w = m_size; x < w; ++x)
						{
							if (mat(x, y) == false)
							{
								float32_t g = gauss(x, y);
								if (g < best)
								{
									best = g;
									bc=0;
									assert( bc <= bestcount );
									best_list[bc++] = ivec2_t(x, y);
								}
								else if (g == best)
								{
									assert( bc <= bestcount );
									best_list[bc++] = ivec2_t(x, y);
								}
							}
						}
					}
					const int32_t idx = hash.iter() % bc;
					m_min_x = best_list[idx].x;
					m_min_y = best_list[idx].y;

					delete [] best_list;
					break;
				}
				else
				{
					ReFilter();
				}
			}
		} //while

		mat( m_min_x, m_min_y ) = true;
	}

	//void Min_multiple( hash_wellon &hash )
	//{
	//	Filter();
	//
	//	//note: find lowest number in filtered matrix
	//	float32_t best = FLT_MAX;
	//	std::vector<ivec2_t> best_list;
	//	for ( int y = 0, h = m_size; y < h; ++y )
	//	{
	//		for ( int x = 0, w = m_size; x < w; ++x )	
	//		{
	//			if ( mat(x,y) == false )
	//			{
	//				float32_t g = gauss(x,y);
	//				if ( g < best )
	//				{
	//					best = g;
	//					best_list.clear();
	//					best_list.push_back( ivec2_t(x, y) );
	//				}
	//				else if ( g == best )
	//				{
	//					best_list.push_back( ivec2_t(x, y) );
	//				}
	//			}
	//		}
	//	}
	//
	//	assert( best_list.size() > 0 );
	//
	//	//note: multiple candidates, bruteforce pick the best one
	//	if ( best_list.size() > 1 )
	//	{
	//		if ( best_list.size() > max_best_siz_min ) max_best_siz_min = best_list.size(); //DEBUG
	//
	//		const int32_t idx = pick_idx_from_best__random( best_list, hash );
	//		m_min_x = best_list[idx].x;
	//		m_min_y = best_list[idx].y;
	//	}
	//	else
	//	{
	//		m_min_x = best_list[0].x;
	//		m_min_y = best_list[0].y;
	//	}
	//
	//	mat(m_min_x, m_min_y) = true;
	//}


	// ====
	void Max()
	{
		Filter();

		//note: find highest number in filtered matrix
		float32_t best = FLT_MIN;
		for ( int y = 0, h=m_size; y < h; ++y )
		{
			for ( int x = 0, w=m_size; x < w; ++x )	
			{
				if ( mat(x,y) )
				{
					float32_t g = gauss(x,y);
					if ( g > best )
					{
						best = g;
						m_max_x = x;
						m_max_y = y;
					}
				}
			}
		}
		mat( m_max_x, m_max_y ) = false;
	}

	// ====
	void Max_refilter()
	{
		Filter();

		//note: find highest number in filtered matrix
		int bestcount = 0;
		while ( bestcount != 1 )
		{
			float32_t best = FLT_MIN;
			bestcount = 0;
			for ( int y = 0, h=m_size; y < h; ++y )
			{
				for ( int x = 0, w=m_size; x < w; ++x )	
				{
					if ( mat(x,y) == true )
					{
						float32_t g = gauss(x,y);
						if ( g > best )
						{
							bestcount = 1;
							best = g;
							m_max_x = x;
							m_max_y = y;
						}
						else if ( g == best )
							++bestcount;
					}
				}
			}

			if ( bestcount > 1 )
			{
				//TODO: catch entirely grey images?
				ReFilter();
			}
		}

		mat( m_max_x, m_max_y ) = false;
	}

	//void Max_multiple( hash_wellon &hash )
	//{
	//	Filter();
	//
	//	//note: find highest number in filtered matrix
	//	float32_t best = -1;
	//	std::vector<ivec2_t> best_list;
	//	for ( int y = 0, h = m_size; y < h; ++y )
	//	{
	//		for ( int x = 0, w = m_size; x < w; ++x )	
	//		{
	//			if ( mat(x,y) == true )
	//			{
	//				float32_t g = gauss(x,y);
	//				if ( g > best )
	//				{
	//					best = g;
	//					best_list.clear();
	//					best_list.push_back( ivec2_t(x,y) );
	//				}
	//				else if ( g == best )
	//				{
	//					best_list.push_back( ivec2_t(x,y) );
	//				}
	//			}
	//		}
	//	}
	//
	//	assert( best_list.size() > 0 );
	//
	//	//note: multiple candidates, bruteforce pick the best one
	//	//TODO: could do a wider blur (downsampled)
	//	if ( best_list.size() > 1 )
	//	{
	//		if ( best_list.size() > max_best_siz_max ) max_best_siz_max = best_list.size(); //DEBUG
	//
	//		const int32_t idx = pick_idx_from_best__random( best_list, hash );
	//		m_max_x = best_list[idx].x;
	//		m_max_y = best_list[idx].y;
	//
	//	}
	//	else
	//	{
	//		m_max_x = best_list[0].x;
	//		m_max_y	= best_list[0].y;
	//	}
	//
	//
	//	//remove that one
	//	mat( m_max_x, m_max_y ) = false;
	//}



	// ====
	// note: performs a separable gaussian blur
	//
	// TODO: running window filter
	// TODO: downsample and blur
	// TODO: convert blurred values to int?
	void Filter()
	{
		memset( m_gauss, 0, m_size_sq*sizeof(float32_t) );
		memset( m_gaussTmp, 0, m_size_sq*sizeof(float32_t) );


		for ( int y = 0, h=m_size; y < h; ++y )
		{
			for ( int x = 0, w=m_size; x < w; ++x )
			{
				for ( int i = 0, n = m_gaussSize; i < n; ++i )
				{
					int idx = (i + y + m_ofs) % m_size;
					//gaussTmp(y,x) += mat( idx, x ) * m_filter[i];
					gaussTmp(x,y) += mat( x, idx ) * m_filter[i];
				}
			}
		}

		for ( int y = 0, h=m_size; y < h; ++y )	
		{
			for ( int x = 0, w=m_size; x < w; ++x )	
			{
				for ( int i = 0, n = m_gaussSize; i < n; ++i )
				{
					int idx = (x + i + m_ofs) % m_size;
					//gauss(y,x) += gaussTmp( y, idx ) * m_filter[ i ];
					gauss(x,y) += gaussTmp( idx, y ) * m_filter[ i ];
				}
			}
		}
	}

	void ReFilter()
	{
		//#ifndef NDEBUG
		//static bool dumpmips = false;
		//if ( dumpmips )
		//	dumpGauss( "mip0", m_gauss, m_size);
		//#endif //NDEBUG

		memset( m_gaussTmp, 0, m_size_sq*sizeof(float32_t) );

		for ( int y = 0, h=m_size; y < h; ++y )
		{
			for ( int x = 0, w=m_size; x < w; ++x )
			{
				for ( int i = 0, n = m_gaussSize; i < n; ++i )
				{
					int idx = (i + y + m_ofs) % m_size;
					//gaussTmp(y,x) += gauss( idx, x ) * m_filter[i];
					gaussTmp(x,y) += gauss( x, idx ) * m_filter[i];
				}
			}
		}

		memset( m_gauss, 0, m_size_sq*sizeof(float32_t) );

		for ( int y = 0, h=m_size; y < h; ++y )	
		{
			for ( int x = 0, w=m_size; x < w; ++x )	
			{
				for ( int i = 0, n = m_gaussSize; i < n; ++i )
				{
					int idx = (x + i + m_ofs) % m_size;
					//gauss(y,x) += gaussTmp( y, idx ) * m_filter[ i ];
					gauss(x,y) += gaussTmp( idx, y ) * m_filter[ i ];
				}
			}
		}

		//#ifndef NDEBUG
		//if ( dumpmips )
		//	dumpGauss( "mip1", m_gauss, m_size);
		//#endif //NDEBUG
	}

	struct img_t
	{
		int32_t width;
		int32_t height;
		int32_t *data;

		int32_t& operator() ( int x, int y )
		{
			return data[ width * y + x ];
		}
	};

	size_t max_best_siz_min = 0;
	size_t max_best_siz_max = 0;

	//#ifndef NDEBUG
	//static
	//void dumpGauss( const std::string &in_filename_noext, f32 const * const ptr, const i32 siz )
	//{
	//	ImageProxy img = cImage::createbitmap( siz, siz, cImage::eImageBpp::BPP32 );
	//	u32 *imgptr = reinterpret_cast<u32*>( img->GetData() );
	//
	//	f32 max_gauss = 0;
	//	for ( int i=0,n=siz*siz; i<n; ++i )
	//	{
	//		f32 v = ptr[i];
	//		if ( v > max_gauss )
	//			max_gauss = v;
	//	}
	//
	//	//note: splat to channels
	//	for ( int i=0, n = siz*siz; i<n; ++i )
	//	{
	//		f32 v = ptr[i] / max_gauss;
	//		u8 vb = static_cast<u8>( v * 255.0f + 0.5f );
	//		imgptr[i] = color_t( vb, vb, vb, 255 ).data;
	//	}
	//
	//	std::string ext = std::string(".bmp");
	//	std::string fn = in_filename_noext;
	//	ImageIO_freeimage::SaveImage( *img, fn + ext );
	//	std::cerr << "wrote \"" << fn << "\"" << std::endl;
	//}
	//#endif //NDEBUG


	// ====
	void BlueNoiseMat( parms_thread_t &pt, img_t &output )
	{
		reset();

		hash_wellon hash;
		hash.seed = pt.seed;

		if(pt.threadid==0) printf( "init...\n");

		//note: add some random trues
		//note: correlates to filter-size...
		const int32_t num_random = (m_size_sq * pt.pct_random) / 100;
		for ( int i = 0, n = num_random; i < n; ++i )
		{
			const int idx_x = static_cast<int>( hash.iter() % m_size );
			const int idx_y = static_cast<int>( hash.iter() % m_size );
			mat( idx_x, idx_y ) = true;
		}

		if(pt.threadid==0)  printf( "prefilter...\n");
		//note: spread them out until the loop breaks due to a "perfect" distribution
		//note: for termination, gauss-distance must be smaller than min-distance between points
		for(;;)
		{
			//TODO: refilter?
			Max();
			Min();

			if ( m_min_x==m_max_x && m_min_y == m_max_y )
				break;
		}

		//note: count final number of points
		//      (it's most likely the same number as we started with)
		m_rank = 0;
		
		for ( int y = 0, h=m_size; y < h; ++y )
		{
			for ( int x = 0, w=m_size; x < w; ++x )
			{
				m_rank += static_cast<int32_t>( mat(x, y) );
			}
		}

		for ( int i=0, n=m_size_sq; i<n; ++i )
			m_matClone[i] = m_mat[i];

		m_rankClone = m_rank;

		//note: new empty void and cluster matrix
		int32_t *vnc = reinterpret_cast<int32_t*>( _aligned_malloc( m_size_sq*sizeof(int32_t), ALLOC_ALIGNMENT ) );
		assert( vnc != nullptr );
		//memset( vnc, 0, m_size_sq*sizeof(i32) );

		
		//note: count down the random points to 0% and put initial, lowest points here
		while ( m_rank > 0 )
		{
			if(pt.threadid==0) progress( "maxfilter", m_rankClone-m_rank, m_rankClone );
			m_rank--;

			//Max();
			Max_refilter();

			vnc[ m_max_y*m_size + m_max_x ] = m_rank;
		}
		if(pt.threadid==0) printf("-\n");

		//note: put the copies into use, start over, count up from 10% to 100%
		//TODO: set m_mat from vnc instead?
		bool *swaptemp = m_mat;
		m_mat = m_matClone;
		m_matClone = swaptemp;

		m_rank = m_rankClone;

		while ( m_rank < m_size_sq )
		{
			if(pt.threadid==0) progress( "minfilter", m_rank, m_size_sq );
			 
			//Min();
			Min_refilter( hash );

			vnc[ m_min_y * m_size + m_min_x ] = m_rank;
			m_rank++;
		}

		output.data = vnc;
		output.width = m_size;
		output.height = m_size;
	}

public:
	// ====
	void Execute( parms_thread_t &pt )
	{
		m_gaussSize = pt.filter_siz;
		m_size = pt.image_siz;
		m_size_sq = m_size * m_size;
		m_ofs = m_size - m_gaussSize / 2;

		m_mat      = reinterpret_cast<bool*>( _aligned_malloc( m_size_sq*sizeof(bool), ALLOC_ALIGNMENT ) );
		m_matClone = reinterpret_cast<bool*>( _aligned_malloc( m_size_sq*sizeof(bool), ALLOC_ALIGNMENT ) );
		m_gaussTmp = reinterpret_cast<float32_t*>( _aligned_malloc( m_size_sq*sizeof(float32_t), ALLOC_ALIGNMENT ) );
		m_gauss    = reinterpret_cast<float32_t*>( _aligned_malloc( m_size_sq*sizeof(float32_t), ALLOC_ALIGNMENT ) );
		
		img_t img;
		BlueNoiseMat( pt, img );

		int32_t maxrank = 0;
		for ( int i=0; i<m_size_sq; ++i )
			maxrank = ( img.data[i] > maxrank ) ? img.data[i] : maxrank;

		//std::cerr << "minbestsiz: " << max_best_siz_min << " maxbestsiz " << max_best_siz_max << " maxrank: " << maxrank << std::endl;

		for ( int y = 0, h=m_size; y < h; ++y )
		{
			for ( int x = 0, w=m_size; x < w; ++x )	
			{
				int32_t v = img(x,y); //note: rank
				const float64_t cf = static_cast<float64_t>( v ) / static_cast<float64_t>( m_size_sq );
				
				pt.dst[y*m_size + x] = cf;
			}
		}

		_aligned_free( img.data );

		_aligned_free( m_mat );
		_aligned_free( m_matClone );
		_aligned_free( m_gaussTmp );
		_aligned_free( m_gauss );
	}
};


//=============================================================================

void callwrapper( parms_thread_t *pt )
{
	BlueNoise bn;
	bn.Execute( *pt );
}

void write_to_disk( float64_t *rgba[4], const parms_shared_t &ps)
{
		printf( "write_to_disk...\n");

		unsigned char* bytedata = new unsigned char[4 * ps.image_siz * ps.image_siz];
		unsigned char* bytedata_tpdf = new unsigned char[4 * ps.image_siz * ps.image_siz];
		float32_t* floatdata = new float32_t[4 * ps.image_siz * ps.image_siz];

        float32_t floatmin =  FLT_MAX;
        float32_t floatmax = -FLT_MAX;
		for (int i = 0, n = ps.image_siz * ps.image_siz; i < n; ++i)
		{
			float64_t v_r = rgba[0][i];
			float64_t v_g = rgba[1][i];
			float64_t v_b = rgba[2][i];
			float64_t v_a = rgba[3][i];

			float32_t vf_r = static_cast<float32_t>( v_r );
			float32_t vf_g = static_cast<float32_t>( v_g );
			float32_t vf_b = static_cast<float32_t>( v_b );
			float32_t vf_a = static_cast<float32_t>( v_a );

			//note: float values should be [0;1]
			const float32_t rescale = static_cast<float32_t>(static_cast<float64_t>(ps.image_siz * ps.image_siz) / static_cast<float64_t>(ps.image_siz * ps.image_siz - 1) );
			floatdata[4 * i + 0] = vf_r * rescale;
			floatdata[4 * i + 1] = vf_g * rescale;
			floatdata[4 * i + 2] = vf_b * rescale;
			floatdata[4 * i + 3] = vf_a * rescale;
			floatmin = min(floatmin, floatdata[4 * i + 0]); floatmin = min(floatmin, floatdata[4 * i + 1]); floatmin = min(floatmin, floatdata[4 * i + 2]); floatmin = min(floatmin, floatdata[4 * i + 3]);
			floatmax = max(floatmax, floatdata[4 * i + 0]); floatmax = max(floatmax, floatdata[4 * i + 1]); floatmax = max(floatmax, floatdata[4 * i + 2]); floatmax = max(floatmax, floatdata[4 * i + 3]);

			//note: [0;255], with equal number of pixels with each value
			bytedata[4 * i + 0] = static_cast<uint8_t>(vf_r * 256.0f);
			bytedata[4 * i + 1] = static_cast<uint8_t>(vf_g * 256.0f);
			bytedata[4 * i + 2] = static_cast<uint8_t>(vf_b * 256.0f);
			bytedata[4 * i + 3] = static_cast<uint8_t>(vf_a * 256.0f);

			bytedata_tpdf[4 * i + 0] = static_cast<uint8_t>(remap_noise_tri_unity(vf_r) * 256.0f);
			bytedata_tpdf[4 * i + 1] = static_cast<uint8_t>(remap_noise_tri_unity(vf_g) * 256.0f);
			bytedata_tpdf[4 * i + 2] = static_cast<uint8_t>(remap_noise_tri_unity(vf_b) * 256.0f);
			bytedata_tpdf[4 * i + 3] = static_cast<uint8_t>(remap_noise_tri_unity(vf_a) * 256.0f);
		}

		//note: check for uniformity in uniform distribution
		{
			//TODO: median?
			int32_t histogram_buckets[256] = { 0 };
			for (int i = 0, n = ps.image_siz * ps.image_siz; i < n; ++i)
			{
				uint8_t b = bytedata[4 * i + 0];
				histogram_buckets[b] += 1;
			}
			int32_t mn = INT_MAX, mx = INT_MIN;
			for (int i = 0, n = 256; i < n; ++i)
			{
				int32_t b = histogram_buckets[i];
				mn = min(mn, b);
				mx = max(mx, b);
			}
			printf("uniform histogram num-buckets: min=%d, max=%d\n", mn, mx);
		}

		{
			char filename[512]; memset(filename, 0, 512);

			sprintf_s(filename, 512, "bluenoise_vnc_%dx%d_uni_f%d_rnd%d.hdr", ps.image_siz, ps.image_siz, ps.filter_siz, ps.pct_random);
			stbi_write_hdr(filename, ps.image_siz, ps.image_siz, 4, floatdata);
			std::cout << "wrote \"" << filename << "\" (floatvalue, min: " << floatmin << ", max: " << floatmax << ")\n";

			sprintf_s(filename, 512, "bluenoise_vnc_%dx%d_uni_f%d_rnd%d.bmp", ps.image_siz, ps.image_siz, ps.filter_siz, ps.pct_random);
			stbi_write_bmp(filename, ps.image_siz, ps.image_siz, 4, bytedata);
			std::cout << "wrote \"" << filename << "\"\n";

			sprintf_s(filename, 512, "bluenoise_vnc_%dx%d_tri_f%d_rnd%d.bmp", ps.image_siz, ps.image_siz, ps.filter_siz, ps.pct_random);
			stbi_write_bmp(filename, ps.image_siz, ps.image_siz, 4, bytedata_tpdf);
			std::cout << "wrote \"" << filename << "\"\n";
		}

		// note: debug histogram output
		//{
		//	int min_idx_uni = 0;
		//	int max_idx_uni = 0;
		//	for ( int i = 0; i<256; ++i )
		//	{
		//		const int32_t v_uni = hist_uni[ i ];
		//
		//		#if defined( SHOW_DBG_HISTOGRAM )
		//		if ( idx == 3 )
		//		{
		//			std::cerr << i << ' ';
		//
		//			for ( int j=0, m=v_uni; j<m; ++j )
		//				std::cerr << '.';
		//
		//			std::cerr << ' ';
		//
		//			for ( int j=0, m=v_tri; j<m; ++j )
		//				std::cerr << ',';
		//			std::cerr << std::endl;
		//		}
		//		#endif //SHOW_DBG_HISTOGRAM
		//
		//		if ( v_uni < hist_uni[min_idx_uni] )
		//			min_idx_uni = i;
		//		if ( v_uni > hist_uni[max_idx_uni] )
		//			max_idx_uni = i;
		//	}
		//	std::cerr << "\nmaxval: " << maxval << " minidx: " << min_idx_uni << "(" << hist_uni[min_idx_uni] << ") maxidx: " << max_idx_uni << "(" << hist_uni[max_idx_uni] << ")" << std::endl;
		//}

		delete[] bytedata;
		delete[] bytedata_tpdf;
		delete[] floatdata;
}

void calc_rgba( const parms_shared_t &ps )
{
	parms_thread_t pt[4];
	for ( int i=0,n=4; i<n; ++i )
	{
		pt[i].image_siz= ps.image_siz;
		pt[i].pct_random = ps.pct_random;
		pt[i].filter_siz = ps.filter_siz;
	}

	std::thread threads[4];
	uint32_t seeds[4] = { 83u, 1901u, 8159467u, 15481607u };
	for ( int i=0,n=4; i<n; ++i )
	{
		
		pt[i].dst = reinterpret_cast<float64_t*>(_aligned_malloc(pt[i].image_siz*pt[i].image_siz*sizeof(float64_t), ALLOC_ALIGNMENT));
		pt[i].seed = seeds[i];
		pt[i].threadid = i;
		threads[i] = std::thread( callwrapper, &pt[i] );
	}
	float64_t *rgba[4] = { pt[0].dst, pt[1].dst, pt[2].dst, pt[3].dst };

	for ( int i=0,n=4; i<n; ++i )
		threads[i].join();

	write_to_disk( rgba, ps );

	_aligned_free(rgba[0]);
	_aligned_free(rgba[1]);
	_aligned_free(rgba[2]);
	_aligned_free(rgba[3]);
}
 

int main( int argc, char **argv )
{
	parms_shared_t parms;
	parms.pct_random = 10;
	parms.filter_siz = 11;

	#if defined( NDEBUG )
	parms.image_siz = 256;
	#else
	parms.image_siz = 32;
	#endif

	if ( argc > 1 )
		parms.image_siz = atoi( argv[1] );

	std::cout << "calculating " << parms.image_siz << "x" << parms.image_siz << std::endl;

	//DEBUG, single call
	{
		//color_t *rgba_uni = reinterpret_cast<color_t*>(_aligned_malloc(siz*siz*sizeof(color_t), 16));
		//color_t *rgba_tri = reinterpret_cast<color_t*>(_aligned_malloc(siz*siz*sizeof(color_t), 16));
		//callwrapper( rgba_uni, rgba_tri, 0, siz, filtersiz );
		//return 0;
	}

	//note: test filtersizes
	{
		//for(int i = 0; i < 6; ++i)
		//{
		//	parms.filter_siz = 3 + 2 * i;
		//	calc_rgba( parms );
		//}
		//return 0;
	}

	//note: test number of random points
	{
		//parms.pct_random = 5; calc_rgba( parms );
		//parms.pct_random = 10; calc_rgba( parms );
		//parms.pct_random = 20; calc_rgba( parms );
		//parms.pct_random = 30; calc_rgba( parms );
		//parms.pct_random = 40; calc_rgba( parms );
		//parms.pct_random = 50; calc_rgba(  parms );
		//return 0;
	}

	calc_rgba( parms );

	return 0;
}
