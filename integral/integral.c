#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int N = 536870912;

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char *get_source_code(const char *file_name, size_t *len) {
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char *)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';

    fclose(file);

    *len = length;
    return source_code;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

double f(double x) {
    return (3*x*x + 2*x + 1);
}

double integral_seq(int N) {
    double dx = (1.0 / (double) N);
    double sum = 0;
    int i;

    double start_time, end_time;
    start_time = get_time();
    for(i = 0; i < N; i++)
        sum += f(i*dx) * dx;
    end_time = get_time();
    printf("Elapsed time : %f sec\n", end_time - start_time);
    return sum;
}

double integral_opencl(int N) {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    char *kernel_source;
    size_t kernel_source_size;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**) &kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if( err == CL_BUILD_PROGRAM_FAILURE ) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*) malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL); 
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    kernel = clCreateKernel(program, "integral", &err);
    CHECK_ERROR(err);

    cl_mem buf_pSum;
    size_t global_size = N;
    size_t local_size = 256;
    size_t num_work_groups = global_size / local_size;

    buf_pSum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_work_groups, NULL, &err);
    CHECK_ERROR(err);

    double start_time, end_time;
    start_time = get_time();

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_pSum);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(double) * local_size, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &N);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    double *partial_sum = (double *) malloc(sizeof(double) * num_work_groups);
    err = clEnqueueReadBuffer(queue, buf_pSum, CL_TRUE, 0, sizeof(double) * num_work_groups, partial_sum, 0, NULL, NULL);
    CHECK_ERROR(err);

    double sum = 0.0;
    int i;
    for(i = 0; i <num_work_groups; i++) {
        sum += partial_sum[i];
    }

    end_time = get_time();
    printf("Elapsed time : %f sec\n", end_time - start_time);

    clReleaseMemObject(buf_pSum);
    free(partial_sum);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return sum;
}

int main() {
    double ans;
    printf("Sequential version...\n");
    ans = integral_seq(N);
    printf("Average: %f\n", ans);

    printf("OpenCL version...\n");
    ans = integral_opencl(N);
    printf("Average: %f\n", ans);

    return 0;
}
