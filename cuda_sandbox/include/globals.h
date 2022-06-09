#ifndef GLOBALS_H
#define GLOBALS_H

constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
constexpr unsigned int num_ch_nodes = 16;

__global__ void block_copy(double* src, 
                           const int src_rows,
                           double* dst, 
                           const int dst_rows,
                           const int row_index,
                           const int col_index) {
    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int tidx = threadIdx.x;
    const unsigned int bidx = blockIdx.x;
    //blocks = columns
    //threads = rows

    dst[tidx+bidx*dst_rows] = src[row_index+col_index*src_rows+tidx+bidx*src_rows];
    // cublasDcopy(cublasH, dst_rows,
    //             src+row_index+col_index*src_rows+tid*src_rows, 1,
    //             dst+tid*dst_rows, 1);
}

#endif