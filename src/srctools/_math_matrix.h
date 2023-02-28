
struct mat3_t
{
    double _aa, _ab, _ac;
    double _ba, _bb, _bc;
    double _ca, _cb, _cc;
};


bool mat3_inverse(const mat3_t* in, mat3_t* out);

// This is here so Python can talk to us easier
typedef double mat_t[3][3]; 
inline bool mat3_inverse(const mat_t* in, mat_t* out) { return mat3_inverse((const mat3_t*)in, (mat3_t*)out); }
