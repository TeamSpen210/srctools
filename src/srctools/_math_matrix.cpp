#include <math.h>
#include "_math_matrix.h"


#ifdef USE_SIMD
#include <emmintrin.h>

// a = a - b * c
void _mat3_inverse_fmulsub(double* fa, double* fb, double c)
{
    __m128d* a = reinterpret_cast<__m128d*>(fa);
    __m128d* b = reinterpret_cast<__m128d*>(fb);

    __m128d v = _mm_set1_pd(c);
	a[0] = _mm_sub_pd(a[0], _mm_mul_pd(b[0], v));
	a[1] = _mm_sub_pd(a[1], _mm_mul_pd(b[1], v));
	a[2] = _mm_sub_pd(a[2], _mm_mul_pd(b[2], v));
}
void _mat3_inverse_div(double* fa, double c)
{
    __m128d* a = reinterpret_cast<__m128d*>(fa);

    __m128d v = _mm_set1_pd(c);
	a[0] = _mm_div_pd(a[0], v);
	a[1] = _mm_div_pd(a[1], v);
	a[2] = _mm_div_pd(a[2], v);
}
#else
// a = a - b * c
void _mat3_inverse_fmulsub(double* a, double* b, double c)
{
    for(int i = 0; i < 6; i++)
		a[i] -= b[i] * c;
}
void _mat3_inverse_div(double* a, double c)
{
    for(int i = 0; i < 6; i++)
		a[i] /= c;
}
#endif

bool mat3_inverse(const mat3_t* in, mat3_t* out)
{
    // We're already in row major
    // Augment in an identity matrix
#ifdef USE_SIMD
	__m128d smat[3][3] = 
    {{_mm_setr_pd(in->_aa, in->_ab), _mm_setr_pd(in->_ac, 1.0), _mm_setr_pd(0.0, 0.0) }, 
     {_mm_setr_pd(in->_ba, in->_bb), _mm_setr_pd(in->_bc, 0.0), _mm_setr_pd(1.0, 0.0) }, 
     {_mm_setr_pd(in->_ca, in->_cb), _mm_setr_pd(in->_cc, 0.0), _mm_setr_pd(0.0, 1.0) }}; 
    double (&omat) [3][6] = (double(&)[3][6])smat;
#else
    double omat[3][6] =
    {{in->_aa, in->_ab, in->_ac, 1.0, 0.0, 0.0 },
     {in->_ba, in->_bb, in->_bc, 0.0, 1.0, 0.0 },
     {in->_ca, in->_cb, in->_cc, 0.0, 0.0, 1.0 }};
#endif

    // Keep the matrix as pointers so we can swap rows quickly without damaging our augment in omat
    double* mat[3] =
    {(double*)omat[0],
     (double*)omat[1],
     (double*)omat[2]};
    
    // Get it into row echelon form
    for (int n = 0; n < 2; n++)
    {
        // Find pivots
        double la = 0;
        int pivrow = -1;
        for (int m = n; m < 3; m++)
        {
            double va = fabs(mat[m][n]);

            if (va > la)
            {
                pivrow = m;
                la = va;
            }
        }

        // No pivot? No solution!
        if (pivrow == -1)
            return false;

        // Swap pivot to highest
        double* pivot = mat[pivrow];
        mat[pivrow] = mat[n];
        mat[n] = pivot;

        // Apply our pivot row to the rows below 
        for (int m = n + 1; m < 3; m++)
        {
            // Get the multiplier
            double* row = mat[m];
            double v = row[n] / pivot[n];
            
			_mat3_inverse_fmulsub(row, pivot, v);
        }
    }

    // Get it into reduced row echelon form
    for (int n = 2; n; n--)
    {
        double* pivot = mat[n];
        for (int m = n - 1; m >= 0; m--)
        {
            // Get the multiplier
            double* row = mat[m];
            double v = row[n] / pivot[n];

            // Elminate!
			_mat3_inverse_fmulsub(row, pivot, v);
		}
    }

    // Clean up our diagonal
    for (int n = 0; n < 3; n++)
    {
        double* row = mat[n];
        double v = row[n];

        // Check for zeros along the diagonal
        if (fabs(v) <= 0.00001)
            return false;

        _mat3_inverse_div(row, v);
    }

    *out =
    {mat[0][3], mat[0][4], mat[0][5],
     mat[1][3], mat[1][4], mat[1][5],
     mat[2][3], mat[2][4], mat[2][5]};

    return true;
}
