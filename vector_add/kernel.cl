__kernel void vec_add(__global int *A,
                      __global int *B,
                      __global int *C)
{    
    uint dim = get_work_dim();
    size_t gs = get_global_size(dim);
    size_t ls = get_local_size(dim);
    size_t ng = get_num_groups(dim);

    printf("[DEBUG] %u, %u, %u, %u\n", dim, gs, ls, ng);

  int i = get_global_id(0);
  C[i] = A[i] + B[i];
}
