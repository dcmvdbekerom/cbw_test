cdef extern from "crs_cpp.h":

    float add_flt(float a, float b);
    
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
                           double* S_kl);
                           
    void simd_calc_matrix_fp64( double p,
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
                           double* S_kl);

    void cpp_mult1_fp64(int N_t,
                    int N_G,
                    double tau23,
                    double* w_arr,
                    double* S_kl,
                    double* S_kl_a);

    void simd_mult1_fp64(int N_t,
                    int N_G,
                    double tau23,
                    double* w_arr,
                    double* S_kl,
                    double* S_kl_a);

                           
    void cpp_mult2_fp64( int N_G,
                            int N_t,
                            double tau23,
                            double log_G_min,
                            double dxG,
                            double* E_probe,
                            double* t_arr,
                            double* S_kl,
                            double* E_CARS);

    void simd_mult2_fp64( int N_G,
                            int N_t,
                            double tau23,
                            double log_G_min,
                            double dxG,
                            double* E_probe,
                            double* t_arr,
                            double* S_kl,
                            double* E_CARS);