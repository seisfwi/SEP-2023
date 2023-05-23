// Dongzhuo Li 05/06/2018
#include "Boundary.h"
#include "Cpml.h"
#include "Model.h"
#include "Parameter.h"
#include "Src_Rec.h"
#include "utilities.h"
#include <chrono>
#include <string>
using std::string;

/*
        double misfit
        double *grad_Lambda : gradients of Lambda (lame parameter)
        double *grad_Mu : gradients of Mu (shear modulus)
        double *grad_Den : gradients of density
        double *grad_stf : gradients of source time function
        double *Lambda : lame parameter (Pascal)
        double *Mu : shear modulus (Pascal)
        double *Den : density
        double *stf : source time function of all shots
        int calc_id :
                calc_id = 0  -- compute residual
                calc_id = 1  -- compute gradient
                calc_id = 2  -- compute observation only
        int gpu_id : CUDA_VISIBLE_DEVICES
        int group_size: number of shots in the group
        int *shot_ids : processing shot shot_ids
        string para_fname : parameter path
*/

extern "C" void cufd(float *misfit, float *grad_Lambda, float *grad_Mu,
                     float *grad_Den, float *grad_stf, const float *Lambda,
                     const float *Mu, const float *Den, const float *stf,
                     int calc_id, const int gpu_id, const int group_size,
                     const int *shot_ids, const string para_fname) {

  // Set GPU device
  CHECK(cudaSetDevice(gpu_id));
  auto start0 = std::chrono::high_resolution_clock::now();

  // Check calc_id
  if (calc_id < 0 || calc_id > 2) {
    printf("Invalid calc_id %d\n", calc_id);
    exit(0);
  }

  // Read parameter file
  Parameter para(para_fname, calc_id);
  int nz = para.nz();
  int nx = para.nx();
  int nSteps = para.nSteps();
  float dz = para.dz();
  float dx = para.dx();
  float dt = para.dt();
  int nPml = para.nPoints_pml();
  int nPad = para.nPad();
  float f0 = para.f0();

  // Set default values
  int iSnap = 500;
  int nrec = 1;
  // float win_ratio = 0.005;
  float amp_ratio = 1.0;

  // Transpose models and convert to float
  float *fLambda, *fMu, *fDen;
  fLambda = (float *)malloc(nz * nx * sizeof(float));
  fMu = (float *)malloc(nz * nx * sizeof(float));
  fDen = (float *)malloc(nz * nx * sizeof(float));
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      fLambda[j * nz + i] = Lambda[i * nx + j] * MEGA;
      fMu[j * nz + i] = Mu[i * nx + j] * MEGA;
      fDen[j * nz + i] = Den[i * nx + j];
    }
  }

  // Set up model, cmpl, 
  Model model(para, fLambda, fMu, fDen);

  // Set up CPML boundary conditions
  Cpml cpml(para, model);
  Bnd boundaries(para);

  // Set up source and receiver
  Src_Rec src_rec(para, para.survey_fname(), stf, group_size, shot_ids);

  // Compute Courant number
  compCourantNumber(model.h_Cp, nz * nx, dt, dz, dx);

  dim3 threads(TX, TY);
  dim3 blocks((nz + TX - 1) / TX, (nx + TY - 1) / TY);

  // Define device memory
  float *d_vz, *d_vx, *d_szz, *d_sxx, *d_sxz;
  float *d_vz_adj, *d_vx_adj, *d_szz_adj, *d_sxx_adj, *d_sxz_adj;
  float *d_mem_dvz_dz, *d_mem_dvz_dx, *d_mem_dvx_dz, *d_mem_dvx_dx;
  float *d_mem_dszz_dz, *d_mem_dsxx_dx, *d_mem_dsxz_dz, *d_mem_dsxz_dx;
  float *d_l2Obj_temp;
  float *d_l2Obj_temp_vx;
  float *d_l2Obj_temp_vz;
  float *d_l2Obj_temp_ett;
  float *h_l2Obj_temp = nullptr;
  float *h_l2Obj_temp_vx = nullptr;
  float *h_l2Obj_temp_vz = nullptr;
  float *h_l2Obj_temp_ett = nullptr;
  float h_l2Obj = 0.0;
  float *d_gauss_amp;
  float *d_data;
  float *d_data_vx;
  float *d_data_vz;
  float *d_data_ett;
  float *d_data_adj;  // adjoint data (pressure)
  float *d_data_obs;
  float *d_data_obs_vx;
  float *d_data_obs_vz;
  float *d_data_obs_ett;
  float *d_res;
  float *d_res_vx;
  float *d_res_vz;
  float *d_res_ett;
  double *d_rec_rxz;
  float *d_obs_normfact, *d_cal_normfact, *d_cross_normfact;

  // Allocate device memory
  CHECK(cudaMalloc((void **)&d_vz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vz_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz_adj, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dvz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvz_dx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dx, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dszz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxx_dx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dx, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_l2Obj_temp, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_l2Obj_temp_vx, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_l2Obj_temp_vz, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_l2Obj_temp_ett, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_gauss_amp, 81 * sizeof(float)));

  h_l2Obj_temp     = (float *)malloc(sizeof(float));
  h_l2Obj_temp_vx  = (float *)malloc(sizeof(float));
  h_l2Obj_temp_vz  = (float *)malloc(sizeof(float));
  h_l2Obj_temp_ett = (float *)malloc(sizeof(float));

  // Set the source gaussian amplitude (but is not used in fact)
  src_rec_gauss_amp<<<1, threads>>>(d_gauss_amp, 9, 9);

  float *h_snap, *h_snap_back, *h_snap_adj;
  h_snap = (float *)malloc(nz * nx * sizeof(float));
  h_snap_back = (float *)malloc(nz * nx * sizeof(float));
  h_snap_adj = (float *)malloc(nz * nx * sizeof(float));

  cudaStream_t streams[group_size];

  // Modeling over shots
  for (int iShot = 0; iShot < group_size; iShot++) {
    // printf("  Processing shot %d\n", shot_ids[iShot]);

    CHECK(cudaStreamCreate(&streams[iShot]));

    intialArrayGPU<<<blocks, threads>>>(d_vz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_szz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxx_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxz_adj, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, 0.0);

    nrec = src_rec.vec_nrec.at(iShot);

    // Allocate device memory for syn data
    CHECK(cudaMalloc((void **)&d_data, nrec * nSteps * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_data_vx, nrec * nSteps * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_data_vz, nrec * nSteps * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_data_ett, nrec * nSteps * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_data_adj, nrec * nSteps * sizeof(float)));

    intialArrayGPU<<<blocks, threads>>>(d_data, nSteps, nrec, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_data_vx, nSteps, nrec, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_data_vz, nSteps, nrec, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_data_ett, nSteps, nrec, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_data_adj, nSteps, nrec, 0.0);

    CHECK(cudaMalloc((void **)&d_rec_rxz, nrec * sizeof(double)));
    CHECK(cudaMemcpy(d_rec_rxz, src_rec.vec_rec_rxz.at(iShot),
                     nrec * sizeof(double), cudaMemcpyHostToDevice));

    if (para.if_res()) {
      fileBinLoad(src_rec.vec_data_obs.at(iShot), nSteps * nrec, 
            para.data_dir_name() + "/Shot_pr" + std::to_string(shot_ids[iShot]) + ".bin");
      fileBinLoad(src_rec.vec_data_obs_vx.at(iShot), nSteps * nrec, 
            para.data_dir_name() + "/Shot_vx" + std::to_string(shot_ids[iShot]) + ".bin");
      fileBinLoad(src_rec.vec_data_obs_vz.at(iShot), nSteps * nrec, 
            para.data_dir_name() + "/Shot_vz" + std::to_string(shot_ids[iShot]) + ".bin");
      fileBinLoad(src_rec.vec_data_obs_ett.at(iShot), nSteps * nrec, 
            para.data_dir_name() + "/Shot_ett" + std::to_string(shot_ids[iShot]) + ".bin");
      
      CHECK(cudaMalloc((void **)&d_data_obs,     nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_data_obs_vx,  nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_data_obs_vz,  nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_data_obs_ett, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res,          nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_vx,       nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_vz,       nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_ett,      nrec * nSteps * sizeof(float)));

      intialArrayGPU<<<blocks, threads>>>(d_data_obs,     nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_data_obs_vx,  nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_data_obs_vz,  nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_data_obs_ett, nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_res,          nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_res_vx,       nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_res_vz,       nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_res_ett,      nSteps, nrec, 0.0);

      CHECK(cudaMemcpyAsync(d_data_obs, src_rec.vec_data_obs.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
      CHECK(cudaMemcpyAsync(d_data_obs_vx, src_rec.vec_data_obs_vx.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
      CHECK(cudaMemcpyAsync(d_data_obs_vz, src_rec.vec_data_obs_vz.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
      CHECK(cudaMemcpyAsync(d_data_obs_ett, src_rec.vec_data_obs_ett.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));

      // initialize normalization factors
      if (para.if_cross_misfit()) {
        CHECK(cudaMalloc((void **)&d_obs_normfact, nrec * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_cal_normfact, nrec * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_cross_normfact, nrec * sizeof(float)));
        intialArrayGPU<<<1, 512>>>(d_obs_normfact, nrec, 1, 0.0);
        intialArrayGPU<<<1, 512>>>(d_cal_normfact, nrec, 1, 0.0);
        intialArrayGPU<<<1, 512>>>(d_cross_normfact, nrec, 1, 0.0);
      }
    }

    // ------------------------ time loop (elastic) ----------------------------
    for (int it = 0; it <= nSteps - 2; it++) {
      
      // save and record from the beginning
      if (para.withAdj()) {
        boundaries.field_from_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it);
      }

      // get snapshot at time it
      if (it == iSnap && iShot == 0) {
        CHECK(cudaMemcpy(h_snap, d_vx, nz * nx * sizeof(float), cudaMemcpyDeviceToHost));
      }
      
      // update stress
      el_stress<<<blocks, threads>>>(
          d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx,
          d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Mu, model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x,
          cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
          cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true, d_szz_adj,
          d_sxx_adj, d_sxz_adj, model.d_LambdaGrad, model.d_MuGrad);

      // add explosive source
      add_source<<<1, threads>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it],
                                 nz, true, 
                                 src_rec.vec_z_src.at(iShot),
                                 src_rec.vec_x_src.at(iShot), 
                                 dt, d_gauss_amp, src_rec.vec_src_rxz.at(iShot));
      
      // update velocity
      el_velocity<<<blocks, threads>>>(
          d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx,
          d_mem_dsxz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z, cpml.d_a_z,
          cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
          cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true, d_vz_adj,
          d_vx_adj, model.d_DenGrad);


      // record pressure data
      recording<<<(nrec + 31) / 32, 32>>>(
          d_szz, d_sxx, nz, d_data, iShot, it + 1, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
          d_rec_rxz);

      // record vx data
      recording_vx<<<(nrec + 31) / 32, 32>>>(
        d_vx, d_vz, nz, d_data_vx, iShot, it + 1, nSteps, nrec,
        src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
        d_rec_rxz);

      // record vz data
      recording_vz<<<(nrec + 31) / 32, 32>>>(
        d_vx, d_vz, nz, d_data_vz, iShot, it + 1, nSteps, nrec,
        src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
        d_rec_rxz);

      // record ett data
      recording_exx<<<(nrec + 31) / 32, 32>>>(
          d_vx, d_vz, nz, d_data_ett, iShot, it + 1, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
          d_rec_rxz);

    } // end of forward time loop

    if (!para.if_res()) {
      CHECK(cudaMemcpyAsync(src_rec.vec_data.at(iShot), d_data,
                            nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_vx.at(iShot), d_data_vx,
                            nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_vz.at(iShot), d_data_vz,
                            nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_ett.at(iShot), d_data_ett,
                            nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot])); 
      }

    // compute residuals
    if (para.if_res()) {
      dim3 blocksT((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);

      // windowing
      // if (para.if_win()) {
      //   cuda_window<<<blocksT, threads>>>(
      //       nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
      //       src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
      //       src_rec.vec_srcweights.at(iShot), win_ratio, d_data_obs);
      //   cuda_window<<<blocksT, threads>>>(
      //       nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
      //       src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
      //       src_rec.vec_srcweights.at(iShot), win_ratio, d_data);
      // } else {
      //   cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
      //                                     d_data_obs);
      //   cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
      //   d_data);
      // }

      // filtering
      // if (para.if_filter()) {
      //   bp_filter1d(nSteps, dt, nrec, d_data_obs, para.filter());
      //   bp_filter1d(nSteps, dt, nrec, d_data, para.filter());
      // }

      // normalization 09/26/2019
      // if (para.if_cross_misfit()) {
      //   cuda_find_normfact<<<nrec, 512>>>(nSteps, nrec, d_data_obs,
      //   d_data_obs,
      //                                     d_obs_normfact);
      //   cuda_find_normfact<<<nrec, 512>>>(nSteps, nrec, d_data, d_data,
      //                                     d_cal_normfact);
      //   cuda_find_normfact<<<nrec, 512>>>(nSteps, nrec, d_data_obs, d_data,
      //                                     d_cross_normfact);
      // }

      // Calculate source update and filter calculated data
      // if (para.if_src_update()) {
      //   amp_ratio =
      //       source_update(nSteps, dt, nrec, d_data_obs, d_data,
      //                     src_rec.d_vec_source.at(iShot), src_rec.d_coef);
      //   printf("	Source update => Processing shot %d, amp_ratio = %f\n",
      //          iShot, amp_ratio);
      // }
      // amp_ratio = 1.0;  // amplitude not used, so set to 1.0

      // objective function
      // if (!para.if_cross_misfit()) {
      //   gpuMinus<<<blocksT, threads>>>(d_res, d_data_obs, d_data, nSteps,
      //   nrec); cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp, d_res, nSteps *
      //   nrec);
      // } else {
      //   cuda_normal_misfit<<<1, 512>>>(nrec, d_cross_normfact,
      //   d_obs_normfact,
      //                                  d_cal_normfact, d_l2Obj_temp,
      //                                  src_rec.d_vec_weights.at(iShot),
      //                                  src_rec.vec_srcweights.at(iShot));
      // }

      gpuMinus<<<blocksT, threads>>>(d_res,     d_data_obs,     d_data,     nSteps, nrec);
      gpuMinus<<<blocksT, threads>>>(d_res_vx,  d_data_obs_vx,  d_data_vx,  nSteps, nrec);
      gpuMinus<<<blocksT, threads>>>(d_res_vz,  d_data_obs_vz,  d_data_vz,  nSteps, nrec);
      gpuMinus<<<blocksT, threads>>>(d_res_ett, d_data_obs_ett, d_data_ett, nSteps, nrec);

      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp,     d_res, nSteps * nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_vx,  d_res_vx, nSteps * nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_vz,  d_res_vz, nSteps * nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_ett, d_res_ett, nSteps * nrec);

      CHECK(cudaMemcpy(h_l2Obj_temp,    d_l2Obj_temp,      sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_l2Obj_temp_vx, d_l2Obj_temp_vx,   sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_l2Obj_temp_vz, d_l2Obj_temp_vz,   sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_l2Obj_temp_ett, d_l2Obj_temp_ett, sizeof(float), cudaMemcpyDeviceToHost));

      // h_l2Obj += h_l2Obj_temp[0];
      // h_l2Obj += h_l2Obj_temp_vx[0] + h_l2Obj_temp_vz[0];
      h_l2Obj += h_l2Obj_temp_ett[0];

      //  update source again (adjoint)
      if (para.if_src_update()) {
        source_update_adj(nSteps, dt, nrec, d_res, amp_ratio,
        src_rec.d_coef);
      }

      // compute negative adjoint source for the normalization objective
      // function 09/26/2019
      // if (para.if_cross_misfit()) {
      //   cuda_normal_adjoint_source<<<blocksT, threads>>>(
      //       nSteps, nrec, d_obs_normfact, d_cal_normfact, d_cross_normfact,
      //       d_data_obs, d_data, d_res, src_rec.d_vec_weights.at(iShot),
      //       src_rec.vec_srcweights.at(iShot));
      // }

      // filtering again (adjoint)
      // if (para.if_filter()) {
      //   bp_filter1d(nSteps, dt, nrec, d_res, para.filter());
      // }
      // windowing again (adjoint)
      // if (para.if_win()) {
      //   cuda_window<<<blocksT, threads>>>(
      //       nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
      //       src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
      //       src_rec.vec_srcweights.at(iShot), win_ratio, d_res);
      // } else {
      //   cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
      //   d_res);
      // }
      

      // data resiudal
      CHECK(cudaMemcpyAsync(src_rec.vec_res.at(iShot), d_res, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_res_vx.at(iShot), d_res_vx, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_res_vz.at(iShot), d_res_vz, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_res_ett.at(iShot), d_res_ett, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));

      // synthetic data
      CHECK(cudaMemcpyAsync(src_rec.vec_data.at(iShot), d_data, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_vx.at(iShot), d_data_vx, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_vz.at(iShot), d_data_vz, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_ett.at(iShot), d_data_ett, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));

      // observed data (preconditioned)
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs.at(iShot), d_data_obs, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_vx.at(iShot), d_data_obs_vx, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_vz.at(iShot), d_data_obs_vz, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_ett.at(iShot), d_data_obs_ett, nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[iShot]));

      CHECK(cudaMemcpy(src_rec.vec_source.at(iShot),
                       src_rec.d_vec_source.at(iShot), nSteps * sizeof(float),
                       cudaMemcpyDeviceToHost));


    }
    // synchronize all streams
    cudaDeviceSynchronize();


    if (para.withAdj()) {
      // --------------------- Backward ----------------------------
      // initialization
      intialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_sxx_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_sxz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(model.d_StfGrad, nSteps, 1, 0.0);
      initialArray(model.h_StfGrad, nSteps, 0.0);

      // update velocity
      el_velocity_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
          d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
          d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);

      // res_injection<<<(nrec + 31) / 32, 32>>>(
      //     d_szz_adj, d_sxx_adj, nz, d_res, nSteps - 1, dt, nSteps, nrec,
      //     src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
      //     d_rec_rxz);
      
      // update stress
      el_stress_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
          d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
          d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);


      for (int it = nSteps - 2; it >= 0; it--) {
        // source time function kernels
        source_grad<<<1, 1>>>(d_szz_adj, d_sxx_adj, nz, model.d_StfGrad, it, dt,
                              src_rec.vec_z_src.at(iShot),
                              src_rec.vec_x_src.at(iShot),
                              src_rec.vec_src_rxz.at(iShot));
            
        // update velocity
        el_velocity<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx,
            d_mem_dsxz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z, cpml.d_a_z,
            cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
            cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half,
            cpml.d_a_x_half, cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad,
            false, d_vz_adj, d_vx_adj, model.d_DenGrad);

        // inject boundary wavefields for reconstruction
        boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, false);
        
        // subtract the source
        add_source<<<1, threads>>>(
            d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz, false,
            src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), dt,
            d_gauss_amp, src_rec.vec_src_rxz.at(iShot));
        
        // update stress
        el_stress<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx,
            d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Mu, model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
            cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x,
            cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
            cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, false, d_szz_adj,
            d_sxx_adj, d_sxz_adj, model.d_LambdaGrad, model.d_MuGrad);

        // inject boundary wavefields for reconstruction
        boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, true);

        // update velocity of the adjoint wavefield
        el_velocity_adj<<<blocks, threads>>>(
            d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
            d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
            d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda,
            model.d_Mu, model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a,
            model.d_ave_Byc_b, cpml.d_K_z_half, cpml.d_a_z_half,
            cpml.d_b_z_half, cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half,
            cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, cpml.d_K_x, cpml.d_a_x,
            cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);

        // // record pressure data
        // recording<<<(nrec + 31) / 32, 32>>>(
        //   d_szz, d_sxx, nz, d_data_adj, iShot, it + 1, nSteps, nrec,
        //   src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
        //   d_rec_rxz);
      
        // pressure residual injection
        // res_injection<<<(nrec + 31) / 32, 32>>>(
        //     d_szz_adj, d_sxx_adj, nz, d_res, it, dt, nSteps, nrec,
        //     src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
        //     d_rec_rxz);
        
        res_injection_exx<<<(nrec + 31) / 32, 32>>>(
            d_vx_adj, d_vz_adj, nz, d_res_ett, it, dt, nSteps, nrec,
            src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
            d_rec_rxz);

        // res_injection_vx<<<(nrec + 31) / 32, 32>>>(
        //   d_vx_adj, d_vz_adj, nz, d_res_vx, it, dt, nSteps, nrec,
        //     src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), 
        //     d_rec_rxz);

        // res_injection_vz<<<(nrec + 31) / 32, 32>>>(
        //   d_vx_adj, d_vz_adj, nz, d_res_vz, it, dt, nSteps, nrec,
        //     src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot),
        //     d_rec_rxz);

        // update velocity of the adjoint wavefield
        el_stress_adj<<<blocks, threads>>>(
            d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
            d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
            d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda,
            model.d_Mu, model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a,
            model.d_ave_Byc_b, cpml.d_K_z_half, cpml.d_a_z_half,
            cpml.d_b_z_half, cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half,
            cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, cpml.d_K_x, cpml.d_a_x,
            cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);

        if (it == iSnap && iShot == 0) {
          CHECK(cudaMemcpy(h_snap_back, d_vz, nz * nx * sizeof(float),
                           cudaMemcpyDeviceToHost));
          CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz * nx * sizeof(float),
                           cudaMemcpyDeviceToHost));
        }
        
        
        // if (iShot == 0) {
          // CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz * nx * sizeof(float),
          //                  cudaMemcpyDeviceToHost));
          // fileBinWrite(h_snap_adj, nz * nx,
          //              "SnapGPU_adj_" + std::to_string(it) + ".bin");
          // CHECK(cudaMemcpy(h_snap, d_szz, nz * nx * sizeof(float),
          //                  cudaMemcpyDeviceToHost));
          // fileBinWrite(h_snap, nz * nx,
          //              "SnapGPU_" + std::to_string(it) + ".bin");
         // }
      
      
      } // the end of backward time loop
      

      // transfer the adjoint pressure data to cpu
      CHECK(cudaMemcpyAsync(src_rec.vec_data_adj.at(iShot), d_data_adj, nSteps * nrec * sizeof(float),
      cudaMemcpyDeviceToHost, streams[iShot]));

      
      // #ifdef DEBUG
      //       fileBinWrite(h_snap_back, nz * nx, "SnapGPU_back.bin");
      //       fileBinWrite(h_snap_adj, nz * nx, "SnapGPU_adj.bin");
      // #endif

      
      // transfer source gradient to cpu
      CHECK(cudaMemcpy(model.h_StfGrad, model.d_StfGrad, nSteps * sizeof(float),
                       cudaMemcpyDeviceToHost));

      for (int it = 0; it < nSteps; it++) {
        grad_stf[iShot * nSteps + it] = model.h_StfGrad[it];
      }

    } // end bracket of if adj


    // for (int iShot = 0; iShot < group_size; iShot++) {
    //   fileBinWrite(src_rec.vec_data_adj.at(iShot), nSteps * src_rec.vec_nrec.at(iShot),
    //                para.data_dir_name() + "/Shot_adj" + std::to_string(shot_ids[iShot]) + ".bin");
    // }


    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_data_vx));
    CHECK(cudaFree(d_data_vz));
    CHECK(cudaFree(d_data_ett));
    CHECK(cudaFree(d_data_adj));
    CHECK(cudaFree(d_rec_rxz));

    if (para.if_res()) {
      CHECK(cudaFree(d_data_obs));
      CHECK(cudaFree(d_data_obs_vx));
      CHECK(cudaFree(d_data_obs_vz));
      CHECK(cudaFree(d_data_obs_ett));
      CHECK(cudaFree(d_res));
      CHECK(cudaFree(d_res_vx));
      CHECK(cudaFree(d_res_vz));
      CHECK(cudaFree(d_res_ett));

      if (para.if_cross_misfit()) {
        CHECK(cudaFree(d_obs_normfact));
        CHECK(cudaFree(d_cal_normfact));
        CHECK(cudaFree(d_cross_normfact));
      }
    }

  } // the end of shot loop

  if (para.withAdj()) {
    // transfer gradients to cpu
    CHECK(cudaMemcpy(model.h_LambdaGrad, model.d_LambdaGrad,
                     nz * nx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(model.h_MuGrad, model.d_MuGrad, nz * nx * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(model.h_DenGrad, model.d_DenGrad, nz * nx * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < nz; i++) {
      for (int j = 0; j < nx; j++) {
        grad_Lambda[i * nx + j] = model.h_LambdaGrad[j * nz + i];
        grad_Mu[i * nx + j] = model.h_MuGrad[j * nz + i];
        grad_Den[i * nx + j] = model.h_DenGrad[j * nz + i];
      }
    }
    
    // #ifdef DEBUG
    //     fileBinWrite(model.h_LambdaGrad, nz * nx, "LambdaGradient.bin");
    //     fileBinWrite(model.h_MuGrad, nz * nx, "MuGradient.bin");
    //     fileBinWrite(model.h_DenGrad, nz * nx, "DenGradient.bin");
    // #endif

    if (para.if_save_scratch()) {
      for (int iShot = 0; iShot < group_size; iShot++) {
        fileBinWrite(src_rec.vec_res.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Residual_Shot" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Syn_Shot" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_obs.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/CondObs_Shot" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        if (para.if_src_update()) {
          fileBinWrite(src_rec.vec_source.at(iShot), nSteps,
                       para.scratch_dir_name() + "/src_updated" +
                           std::to_string(shot_ids[iShot]) + ".bin");
        }
      }
    }
  }

  if (!para.if_res()) {
    for (int iShot = 0; iShot < group_size; iShot++) {
      fileBinWrite(src_rec.vec_data.at(iShot), nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_pr" + std::to_string(shot_ids[iShot]) + ".bin");
    
      fileBinWrite(src_rec.vec_data_vx.at(iShot), nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_vx" + std::to_string(shot_ids[iShot]) + ".bin");
    
      fileBinWrite(src_rec.vec_data_vz.at(iShot), nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_vz" + std::to_string(shot_ids[iShot]) + ".bin");
    
      fileBinWrite(src_rec.vec_data_ett.at(iShot), nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_ett" + std::to_string(shot_ids[iShot]) + ".bin");
      }
  }



  
  // output residual
  if (para.if_res()) {
    h_l2Obj = 0.5 * h_l2Obj; // DL 02/21/2019 (need to make misfit accurate
                             // here rather than in the script)

    *misfit = h_l2Obj;
  }

  free(h_l2Obj_temp);
  free(h_l2Obj_temp_vx);
  free(h_l2Obj_temp_vz);
  free(h_l2Obj_temp_ett);
  free(h_snap);
  free(h_snap_back);
  free(h_snap_adj);
  free(fLambda);
  free(fMu);
  free(fDen);

  // destroy the streams
  for (int iShot = 0; iShot < group_size; iShot++)
    CHECK(cudaStreamDestroy(streams[iShot]));

  cudaFree(d_vz);
  cudaFree(d_vx);
  cudaFree(d_szz);
  cudaFree(d_sxx);
  cudaFree(d_sxz);
  cudaFree(d_vz_adj);
  cudaFree(d_vx_adj);
  cudaFree(d_szz_adj);
  cudaFree(d_sxx_adj);
  cudaFree(d_sxz_adj);
  cudaFree(d_mem_dvz_dz);
  cudaFree(d_mem_dvz_dx);
  cudaFree(d_mem_dvx_dz);
  cudaFree(d_mem_dvx_dx);
  cudaFree(d_mem_dszz_dz);
  cudaFree(d_mem_dsxx_dx);
  cudaFree(d_mem_dsxz_dz);
  cudaFree(d_mem_dsxz_dx);
  cudaFree(d_l2Obj_temp);
  cudaFree(d_l2Obj_temp_vx);
  cudaFree(d_l2Obj_temp_vz);
  cudaFree(d_l2Obj_temp_ett);
  cudaFree(d_gauss_amp);
}
