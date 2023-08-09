
#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#include <math.h>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace std;


double c    = 29979245800.0; //cm.s-1
double k_B  = 1.380649e-23; //J.K-1
double h    = 6.62607015e-34; //J.s
double pi   = 3.141592653589793;


float add_flt(float a, float b){
    return a + b;
}


#define DBL_SIZE 8
#define EPI32_SIZE 4
    
void cpp_calc_matrix_fp64( double p,
                           double T, 
                           double tau23,
                           double* nu,
                           double* sigma_gRmin,
                           double* E0,
                           int* J_clip,
                           int* l0_arr,
                           double* aG1_arr,                      
                           double w_min, 
                           double dw, 
                           int N_w,
                           int N_G,
                           int chunksize,
                           int Nlines,
                           double* Wi_arr, 
                           double* S_kl) {

    int i, k0, k1, l0, offset0, offset1;
    double Bprim, Bbis;
    double k, tw, aw0r, aw0i, aw1r, aw1i, aG0, aG1;
    double Wi, wi, theta;

    double dwt = dw * tau23;
    double r_tan = 0.5/tan(0.5*dwt);
    double r_sin = 0.5/sin(0.5*dwt);
    
    unsigned long long addr0, addr1, base_addr = (unsigned long long)&S_kl[0];
    
    for (i=0; i < Nlines; i++){

        
        Bprim = exp(-h*c* E0[i]         /(k_B*T));
        Bbis  = exp(-h*c*(E0[i] + nu[i])/(k_B*T));

        Wi = sigma_gRmin[i] * abs(Bprim - Bbis); //This difference is always positive
        
        wi = 2*pi*c*nu[i];
        k = (wi - w_min) / dw; //wi/dw - w_min/dw; //
        k0 = (int)k;
        k1 = k0 + 1;
        tw = k - k0;       

        // if ((k0 >= 0) && (k1 < N_w)){
        
        
        theta = 0.5*(2*tw - 1)*dwt;
        aw1r =  r_sin*sin(theta) + 0.5;
        aw1i = -r_sin*cos(theta) + r_tan;
        aw0r = 1 - aw1r;
        aw0i = -aw1i;

        l0 = l0_arr[J_clip[i]];
        aG1 = aG1_arr[J_clip[i]];
        aG0 = 1 - aG1;
    
        offset0 = 2*(k0 * N_G + l0);
        offset1 = offset0 + 2*N_G;
        
        addr0 = base_addr + 8*offset0;
        addr1 = base_addr + 8*offset1;


        //Wi_arr[i] = (double)addr0;
        Wi_arr[i+0] = aw0r * aG0 * Wi;
        Wi_arr[i+1] = aw0i * aG0 * Wi;
        Wi_arr[i+2] = aw0r * aG1 * Wi;
        Wi_arr[i+3] = aw0i * aG1 * Wi;
        
        S_kl[offset1 + 0] += aw1r * aG0 * Wi;
        S_kl[offset1 + 1] += aw1i * aG0 * Wi;        
        S_kl[offset1 + 2] += aw1r * aG1 * Wi;
        S_kl[offset1 + 3] += aw1i * aG1 * Wi;

        S_kl[offset0 + 0] += aw0r * aG0 * Wi;
        S_kl[offset0 + 1] += aw0i * aG0 * Wi;        
        S_kl[offset0 + 2] += aw0r * aG1 * Wi;
        S_kl[offset0 + 3] += aw0i * aG1 * Wi;
    

        // }
    }    
}

#define ADD_LINE(IMM1, IMM2) { \
                                \
    /* {aG0*Wi, aG0*Wi, aG1*Wi, aG1*Wi}    */ \
    temp_pd_0 = _mm256_permute4x64_pd(aG1_Wi_vec, IMM1);\
    temp_pd_1 = _mm256_permute4x64_pd(Wi_vec,  IMM1);\
    temp_pd_1 = _mm256_sub_pd(temp_pd_1, temp_pd_0);\
    temp_pd_0 = _mm256_blend_pd(temp_pd_0, temp_pd_1, 0x03);\
    \
    /* {aw1r, aw1i, aw1r, aw1i} */\
    temp_pd_1 = _mm256_permute4x64_pd(aw1r_vec, IMM1);\
    temp_pd_2 = _mm256_permute4x64_pd(aw1i_vec, IMM1);\
    temp_pd_2 = _mm256_blend_pd(temp_pd_2, temp_pd_1, 0x05);\
    \
    /* {aw1r*aG0*Wi, aw1i*aG0*Wi, aw1r*aG1*Wi, aw1i*aG1*Wi} */\
    temp_pd_1 = _mm256_mul_pd(temp_pd_0, temp_pd_2);\
    \
    /*get address*/\
    S_k0l = reinterpret_cast<double*>(_mm256_extract_epi64(addr_0, IMM2)); \
    S_k1l = reinterpret_cast<double*>(_mm256_extract_epi64(addr_1, IMM2)); \
    \
    /* read-add-write: */\
    temp_pd_2 = _mm256_loadu_pd(S_k1l);    \
    temp_pd_2 = _mm256_add_pd(temp_pd_1, temp_pd_2);\
    _mm256_storeu_pd(S_k1l, temp_pd_2);    \
    \
    /* {aw0r*aG0*Wi, aw0i*aG0*Wi, aw0r*aG1*Wi, aw0i*aG1*Wi}*/\
    temp_pd_2 = _mm256_setzero_pd();\
    temp_pd_0 = _mm256_blend_pd(temp_pd_0, temp_pd_2, 0x0A);\
    temp_pd_1 = _mm256_sub_pd(temp_pd_0, temp_pd_1);\
    \
    /* read-add-write: */\
    temp_pd_2 = _mm256_loadu_pd(S_k0l);    \
    temp_pd_2 = _mm256_add_pd(temp_pd_1, temp_pd_2); \
    _mm256_storeu_pd(S_k0l, temp_pd_2);    \
    };

void simd_calc_matrix_fp64(double p,
                           double T, 
                           double tau23,
                           double* nu,
                           double* sigma_gRmin,
                           double* E0,
                           int* J_clip,
                           int* l0_arr,
                           double* aG1_arr,                      
                           double w_min, 
                           double dw, 
                           int N_w,
                           int N_G,
                           int chunksize,
                           int Nlines,
                           double* Wi_arr,
                           double* S_kl) {

    int i;
    double dwt = dw * tau23;
    double* S_k0l;
    double* S_k1l;
    
    __m256d E0_vec, nu_vec;
    __m256d Bprim_vec, Bbis_vec, Wi_vec, k_vec, sigma_gRmin_vec; //l_vec, Gamma_k_vec,
    __m256d temp_pd_0, temp_pd_1, temp_pd_2;
    __m128i v_index_vec, temp_i32_0, temp_i32_1, k0_vec, l0_vec;
    __m256i addr_0, addr_1;

    __m256d tw_vec;//, tG_vec;
    __m256d sin_vec, cos_vec;
    
    __m256d aw1r_vec, aw1i_vec, aG1_vec, aG1_Wi_vec;    
    
    __m256d factor_vec = _mm256_set1_pd(-h*c/(k_B*T));     
    __m256d two_pi_c   = _mm256_set1_pd(2*pi*c);             
    __m256d dw_recp    = _mm256_set1_pd(1.0/dw);            
    __m256d w_min_vec  = _mm256_set1_pd(w_min);  
    __m256d w_min_norm = _mm256_set1_pd(w_min/dw);              
    __m256d r_sin_vec  = _mm256_set1_pd(0.5/sin(0.5*dwt));
    __m256d r_tan_vec  = _mm256_set1_pd(0.5/tan(0.5*dwt));
    __m256d dwt_vec    = _mm256_set1_pd(dwt); 
    __m256d h_dwt_vec  = _mm256_set1_pd(0.5*dwt);
    __m256d h_vec      = _mm256_set1_pd(0.5);
    __m128i N_G_vec    = _mm_set1_epi32(N_G);
    __m256i S_kl_addr_vec = _mm256_set1_epi64x((unsigned long long)&S_kl[0]);


    #pragma omp parallel for firstprivate(i) lastprivate(i) private(\
    E0_vec, temp_pd_0, Bprim_vec, nu_vec, temp_pd_1, Bbis_vec, sigma_gRmin_vec, \
    Wi_vec, k_vec, k0_vec, tw_vec, sin_vec, cos_vec, aw1r_vec, aw1i_vec, v_index_vec, \
    aG1_vec, aG1_Wi_vec, l0_vec, temp_i32_0, temp_i32_1, addr_0, addr_1, \
    temp_pd_2, S_k0l, S_k1l) \
    schedule(dynamic, chunksize)
    for (i=0; i < Nlines; i+=4){

        //Bprim = exp( E0[i]*factor);
        E0_vec = _mm256_loadu_pd(&E0[i]);               
          temp_pd_0 = _mm256_mul_pd(factor_vec, E0_vec);  
        Bprim_vec = _mm256_exp_pd(temp_pd_0);                  

        //Bbis  = exp((E0[i] + nu[i])*factor);
        nu_vec = _mm256_loadu_pd(&nu[i]);                          
        temp_pd_1 = _mm256_fmadd_pd(nu_vec, factor_vec, temp_pd_0); 
        Bbis_vec = _mm256_exp_pd(temp_pd_1);                       

        //Wi = sigma_gRmin[i]*(Bprim - Bbis);
        temp_pd_0 = _mm256_sub_pd(Bprim_vec, Bbis_vec);            
        
        //temp_pd_0 = _mm256_andnot_pd(sign_bit, temp_pd_0); // not needed cause Bprim > Bbis       
        sigma_gRmin_vec = _mm256_loadu_pd(&sigma_gRmin[i]);  
        Wi_vec = _mm256_mul_pd(sigma_gRmin_vec, temp_pd_0);     

        //wi = 2*pi*c*nu[i];
        //k = (wi - w_min) / dw;
        //k0 = (int)k;
        temp_pd_0 = _mm256_mul_pd(nu_vec, two_pi_c);                      
        temp_pd_0 = _mm256_sub_pd(temp_pd_0, w_min_vec);
        k_vec     = _mm256_mul_pd(temp_pd_0, dw_recp);
        //k_vec = _mm256_fmsub_pd(temp_pd_0, dw_recp, w_min_norm);        //2 ms slower
        
        k0_vec = _mm256_cvttpd_epi32(k_vec);
        
        //tw = k - k0;
        temp_pd_0 = _mm256_cvtepi32_pd(k0_vec);
        tw_vec = _mm256_sub_pd(k_vec, temp_pd_0);

        //theta = 0.5*(2*tw - 1)*dwt = tw*dwt - 0.5*dwt;   
        temp_pd_0 = _mm256_fmsub_pd(tw_vec, dwt_vec, h_dwt_vec);
        sin_vec = _mm256_sincos_pd(&cos_vec, temp_pd_0);
        
        //aw1r =  r_sin*sin(theta) + 0.5;
        //aw1i = -r_sin*cos(theta) + r_tan;        
        aw1r_vec = _mm256_fmadd_pd (r_sin_vec, sin_vec, h_vec);
        aw1i_vec = _mm256_fnmadd_pd(r_sin_vec, cos_vec, r_tan_vec);

        //Gamma_k = (Gamma_JJ[J_clip + delta_J[i]] + Gamma_JJ[J_clip])
        v_index_vec = _mm_loadu_si128((__m128i*)&J_clip[i]);         
        aG1_vec     = _mm256_i32gather_pd(&aG1_arr[0], v_index_vec, DBL_SIZE);
        aG1_Wi_vec  = _mm256_mul_pd(aG1_vec, Wi_vec);

        //Calculating addresses:
        //DBL_SIZE * 2 *(k0_vec * N_G + l0_vec) 
        l0_vec     = _mm_i32gather_epi32(&l0_arr[0], v_index_vec, EPI32_SIZE);
        
        temp_i32_0 = _mm_mullo_epi32(N_G_vec, k0_vec);
        temp_i32_0 = _mm_add_epi32(temp_i32_0, l0_vec);
        temp_i32_1 = _mm_add_epi32(temp_i32_0, N_G_vec);
        
        //k0 index
        temp_i32_0 = _mm_slli_epi32(temp_i32_0, 4);
        addr_0 = _mm256_cvtepi32_epi64(temp_i32_0);
        addr_0 = _mm256_add_epi64(S_kl_addr_vec, addr_0);
        
        //k1 index (+ 2*N_G)
        temp_i32_1 = _mm_slli_epi32(temp_i32_1, 4);
        addr_1 = _mm256_cvtepi32_epi64(temp_i32_1);
        addr_1 = _mm256_add_epi64(S_kl_addr_vec, addr_1);

        //0x00, 0x55, 0xAA, 0xFF
        ADD_LINE(0x00,0x00);
        ADD_LINE(0x55,0x01);
        ADD_LINE(0xAA,0x02);
        ADD_LINE(0xFF,0x03);
    }
};

void cpp_mult1_fp64(int N_t,
                    int N_G,
                    double tau23,
                    double* w_arr,
                    double* S_kl,
                    double* S_kl_a){
    
    int k,ll;
    double theta0, sr0, si0, Sr0, Si0;    
    double theta1, sr1, si1, Sr1, Si1;    
    
    for (k=0; k<N_t; k+=2){
        theta0 = tau23*w_arr[k  ];
        theta1 = tau23*w_arr[k+1];
        
        sr0 = sin(0.5*pi - theta0); // cos(theta0)
        si0 = sin(         theta0);
        sr1 = sin(0.5*pi - theta1); // cos(theta1)
        si1 = sin(         theta1);
        
        for (ll=0; ll<2*N_G; ll+=2){ 
            Sr0 = S_kl[2*k*N_G       + ll  ];
            Si0 = S_kl[2*k*N_G       + ll+1];
            Sr1 = S_kl[2*k*N_G+2*N_G + ll  ];
            Si1 = S_kl[2*k*N_G+2*N_G + ll+1];
            
            S_kl_a[ll*N_t + 2*k  ] = Sr0*sr0 - Si0*si0;
            S_kl_a[ll*N_t + 2*k+1] = Sr0*si0 + Si0*sr0;
            S_kl_a[ll*N_t + 2*k+2] = Sr1*sr1 - Si1*si1;
            S_kl_a[ll*N_t + 2*k+3] = Sr1*si1 + Si1*sr1;
        };
    };
};



void simd_mult1_fp64(int N_t,
                    int N_G,
                    double tau23,
                    double* w_arr,
                    double* S_kl,
                    double* S_kl_a){
    
    int k,ll;
	__m256d hpi_vec = {0.5*pi, 0.0, 0.5*pi, 0.0};
	__m256d temp_pd_0, temp_pd_1, sA_vec, sB_vec, Sr_vec, Si_vec;
	double* base_addr0;

    for (k=0; k<N_t; k+=2){
        //theta0 = tau23*w_arr[k  ];
        //theta1 = tau23*w_arr[k+1];
		temp_pd_0 = _mm256_broadcast_pd(reinterpret_cast<__m128d*>(&w_arr[k]));     //{theta0, theta1, theta0, theta1}
		temp_pd_0 = _mm256_permute_pd(temp_pd_0, 0x0C); //{theta0, theta0, theta1, theta1}
		
        //sr0 = sin(0.5*pi - theta0); // cos(theta0)
        //si0 = sin(         theta0);
        //sr1 = sin(0.5*pi - theta1); // cos(theta1)
        //si1 = sin(         theta1);		
		temp_pd_0 = _mm256_addsub_pd(hpi_vec, temp_pd_0);
		sA_vec = _mm256_sin_pd(temp_pd_0);
		sB_vec = _mm256_shuffle_pd(sA_vec, sA_vec, 0x05); //0b000000101
		base_addr0 = &S_kl[2*k*N_G];
		
        for (ll=0; ll<2*N_G; ll+=2){ 
            //Sr0 = base_addr[        ll  ];
            //Si0 = base_addr[        ll+1];
            //Sr1 = base_addr[2*N_G + ll  ];
            //Si1 = base_addr[2*N_G + ll+1];
			temp_pd_0 = _mm256_broadcast_pd(reinterpret_cast<__m128d*>(&base_addr0[ll]));
			temp_pd_1 = _mm256_broadcast_pd(reinterpret_cast<__m128d*>(&base_addr0[ll + 2*N_G]));
			
			Sr_vec = _mm256_shuffle_pd(temp_pd_0, temp_pd_1, 0x00); //0b00000000
            Si_vec = _mm256_shuffle_pd(temp_pd_0, temp_pd_1, 0x0F); //0b00001111
			Sr_vec = _mm256_permute_pd(Sr_vec, 0x0C);//0b00001100
			Si_vec = _mm256_permute_pd(Si_vec, 0x0C);//0b00001100
			
            //S_kl_a[ll*N_t + 2*k  ] = Sr0*sr0 - Si0*si0;
            //S_kl_a[ll*N_t + 2*k+1] = Sr0*si0 + Si0*sr0;
            //S_kl_a[ll*N_t + 2*k+2] = Sr1*sr1 - Si1*si1;
            //S_kl_a[ll*N_t + 2*k+3] = Sr1*si1 + Si1*sr1;
			temp_pd_0 = _mm256_mul_pd(Si_vec, sB_vec);
			temp_pd_0 = _mm256_fmsubadd_pd(Sr_vec, sA_vec, temp_pd_0);
			_mm256_storeu_pd(&S_kl_a[ll*N_t + 2*k], temp_pd_0); 
			
        };
    };
};



void cpp_mult2_fp64( int N_G,
                            int N_t,
                            double tau23,
                            double log_G_min,
                            double dxG,                
                            double* E_probe,
                            double* t_arr,
                            double* S_kl,
                            double* E_CARS){
    
    int l, k;
    double G_l, factor, chi_r, chi_i;
    
    for (l=0; l<N_G; l++){
        G_l = exp(log_G_min + l*dxG);
        
        for (k=0; k<N_t; k++){
            factor = exp(-G_l * (t_arr[k] + tau23)) * N_t;
            
            chi_r = S_kl[2*N_t*l + 2*k] * factor;
            chi_i = S_kl[2*N_t*l + 2*k+1] * factor;
            
            E_CARS[2*k  ] += chi_r * E_probe[k];
            E_CARS[2*k+1] += chi_i * E_probe[k];
        };
    };
    
};


void simd_mult2_fp64( int N_G,
                            int N_t,
                            double tau23,
                            double log_G_min,
                            double dxG,                
                            double* E_probe,
                            double* t_arr,
                            double* S_kl,
                            double* E_CARS){
    
    int l, k;
    double G_l;
    __m256d N_t_vec = _mm256_set1_pd(N_t);
    double* base_addr;   
    __m256d G_l_vec, factor_vec, t_arr_vec, temp_pd_0, probe_vec, temp_probe_vec, temp_factor_vec, chi_vec, E_CARS_vec;
	
	__m256d tau_vec = _mm256_set1_pd(tau23);
    
    for (l=0; l<N_G; l++){
        G_l = exp(log_G_min + l*dxG);
        G_l_vec = _mm256_set1_pd(-G_l);
        base_addr = &S_kl[2*N_t*l]; 
        
        for (k=0; k<N_t; k+=4){

            //factor = exp(-G_l * (t_arr[k] + tau23)) * N_t; 
            t_arr_vec = _mm256_loadu_pd(&t_arr[k]);
            temp_pd_0 = _mm256_add_pd(t_arr_vec, tau_vec);
            temp_pd_0 = _mm256_mul_pd(temp_pd_0, G_l_vec); // can be single fmadd
            temp_pd_0 = _mm256_exp_pd(temp_pd_0);
            factor_vec = _mm256_mul_pd(temp_pd_0, N_t_vec); //TODO: *N_t shouldn't be here
            probe_vec = _mm256_loadu_pd(&E_probe[k]);
            
            
            //First pair:
            temp_factor_vec = _mm256_permute4x64_pd(factor_vec, 0x50); //0b01010000
            temp_probe_vec = _mm256_permute4x64_pd(probe_vec, 0x50); //0b01010000
            
            //chi_r = S_kl[2*N_t*l + 2*k] * factor;
            //chi_i = S_kl[2*N_t*l + 2*k+1] * factor;
            chi_vec = _mm256_loadu_pd(&base_addr[2*k]);
            chi_vec = _mm256_mul_pd(chi_vec, temp_factor_vec);
            
            //E_CARS[2*k  ] += chi_r * E_probe[k];
            //E_CARS[2*k+1] += chi_i * E_probe[k];
            chi_vec = _mm256_mul_pd(chi_vec, temp_probe_vec);
            E_CARS_vec = _mm256_loadu_pd(&E_CARS[2*k]);
            E_CARS_vec = _mm256_add_pd(E_CARS_vec, chi_vec);
            _mm256_storeu_pd(&E_CARS[2*k], E_CARS_vec);
            
            
            //Second pair:
            temp_factor_vec = _mm256_permute4x64_pd(factor_vec, 0xFA);   //0b11111010
            temp_probe_vec = _mm256_permute4x64_pd(probe_vec, 0xFA);   //0b11111010
            
            //chi_r = S_kl[2*N_t*l + 2*k] * factor;
            //chi_i = S_kl[2*N_t*l + 2*k+1] * factor;
            chi_vec = _mm256_loadu_pd(&base_addr[2*k+4]);
            chi_vec = _mm256_mul_pd(chi_vec, temp_factor_vec);
            
            //E_CARS[2*k  ] += chi_r * E_probe[k];
            //E_CARS[2*k+1] += chi_i * E_probe[k];
            chi_vec = _mm256_mul_pd(chi_vec, temp_probe_vec);
            E_CARS_vec = _mm256_loadu_pd(&E_CARS[2*k+4]);
            E_CARS_vec = _mm256_add_pd(E_CARS_vec, chi_vec);
            _mm256_storeu_pd(&E_CARS[2*k+4], E_CARS_vec);    
        };
    };
    
};


//simd_mult2_fp64();