#include "ex1.h"

#define THREADS_IN_KERNEL 1024
#define TILES_NUM (TILE_COUNT * TILE_COUNT)
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)
#define HIST_GRAYSCALE_VALUES 256
#define BULK_THREADS_IN_KERNEL 256
#define THREAD_GROUPS_COUNT (THREADS_IN_KERNEL / HIST_GRAYSCALE_VALUES)
#define TILE_HEIGHT (TILE_WIDTH)

 // global memory for histograms
__device__ int gpu_histogram[TILES_NUM * HIST_GRAYSCALE_VALUES];
__device__ int gpu_histogram_bulk[TILES_NUM * HIST_GRAYSCALE_VALUES * N_IMAGES]; 

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x % arr_size;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid < arr_size && tid >= stride) {
            increment = arr[tid - stride];
        }
        
        __syncthreads();

        //Accumulate the value into the current element
        if (tid < arr_size && tid >= stride) {
            arr[tid] += increment;
        }
        
        __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    // 1. compute image histogram per tile
    // 2. compute prefix sum of the histogram per tile
    // 3. create a map for each tile using given formula
    // 4. invoke interpolate_device to perform interpolation on the image
    
    // int tid = threadIdx.x;

    // Step 0: Clear the global histogram from the previous kernel launch
    // for(int i=0; i < (TILES_NUM * HIST_GRAYSCALE_VALUES) / THREADS_IN_KERNEL; i++) {
    //     gpu_histogram[tid + i * THREADS_IN_KERNEL] = 0;
    // }
    for(int i = threadIdx.x; i < TILES_NUM * HIST_GRAYSCALE_VALUES; i += blockDim.x) {
        gpu_histogram[i] = 0;
    }
    __syncthreads();

    // Step 1: Compute histogram per tile
    // for(int i=0; i < IMG_SIZE / THREADS_IN_KERNEL; i++) {
    //     int pixel_index = tid + i * THREADS_IN_KERNEL;
    //     int tile_x = (pixel_index % IMG_WIDTH) / TILE_WIDTH;
    //     int tile_y = pixel_index / IMG_WIDTH / TILE_HEIGHT;
    //     int tile_index = tile_x + tile_y * TILE_COUNT;
    //     int pixel_value = all_in[pixel_index];
    //     atomicAdd(&gpu_histogram[tile_index * HIST_GRAYSCALE_VALUES + pixel_value], 1);
    // }
    for(int pixel_idx = threadIdx.x; pixel_idx < IMG_SIZE; pixel_idx += blockDim.x) {
        int tile_x = (pixel_idx % IMG_WIDTH) / TILE_WIDTH;
        int tile_y = pixel_idx / IMG_WIDTH / TILE_HEIGHT;
        int tile_index = tile_x + tile_y * TILE_COUNT;
        
        int pixel_value = all_in[pixel_idx]; // 'pixel_idx' maps directly to the global index
        
        atomicAdd(&gpu_histogram[tile_index * HIST_GRAYSCALE_VALUES + pixel_value], 1);
    }
    __syncthreads();

    // // Step 2: Compute prefix sum of the histogram per tile
    // for(int tile_index = 0; tile_index < TILES_NUM ; tile_index += THREAD_GROUPS_COUNT) {
    //     // tid / HIST_GRAYSCALE_VALUES - Offsets the base tile index by the thread's group ID (tid / 256) 
    //     // so each concurrent thread group targets its own unique histogram.
    //     prefix_sum(&gpu_histogram[(tile_index + tid / HIST_GRAYSCALE_VALUES) * HIST_GRAYSCALE_VALUES], HIST_GRAYSCALE_VALUES);
    // }
    int thread_groups = blockDim.x / HIST_GRAYSCALE_VALUES;
    int group_id = threadIdx.x / HIST_GRAYSCALE_VALUES;

    for(int tile_index = 0; tile_index < TILES_NUM; tile_index += thread_groups) {
        
        int my_tile = tile_index + group_id;
        
        // A thread is active ONLY if its assigned tile actually exists 
        bool is_active = (my_tile < TILES_NUM);

        if (is_active) {
            prefix_sum(&gpu_histogram[my_tile * HIST_GRAYSCALE_VALUES], HIST_GRAYSCALE_VALUES);
        }        
    }
    __syncthreads();

    // Step 3: Create a map for each tile using the given formula
    for(int i = threadIdx.x; i < TILES_NUM * HIST_GRAYSCALE_VALUES; i += blockDim.x) {
        maps[i] = gpu_histogram[i] * (HIST_GRAYSCALE_VALUES - 1) / (TILE_WIDTH * TILE_WIDTH);
    }
    __syncthreads();

    // Step 4: invoke interpolate_device to perform interpolation on the image
    interpolate_device(maps, all_in, all_out);
}
__global__ void process_image_kernel_bulk(uchar *all_in, uchar *all_out, uchar *maps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Step 1: Compute histogram per tile
    for(int i=0; i < IMG_SIZE / blockDim.x; i++) {
        int pixel_index = tid + i * blockDim.x;
        int tile_x = (pixel_index % IMG_WIDTH) / TILE_WIDTH;
        int tile_y = pixel_index / IMG_WIDTH / TILE_HEIGHT;
        int tile_index = tile_x + tile_y * TILE_COUNT;
        int pixel_value = all_in[bid * IMG_SIZE + pixel_index];
        atomicAdd(&gpu_histogram_bulk[(bid * TILES_NUM + tile_index) * HIST_GRAYSCALE_VALUES + pixel_value], 1);
    }
    __syncthreads();

    // Step 2: Compute prefix sum of the histogram per tile
    for(int tile_index = 0; tile_index < TILES_NUM ; tile_index ++) {
        prefix_sum(&gpu_histogram_bulk[(bid * TILES_NUM + tile_index) * HIST_GRAYSCALE_VALUES], HIST_GRAYSCALE_VALUES);
    }
    __syncthreads();

    // Step 3: Create a map for each tile using the given formula
    for(int i = threadIdx.x; i < TILES_NUM * HIST_GRAYSCALE_VALUES; i += blockDim.x) {
        maps[bid * TILES_NUM * HIST_GRAYSCALE_VALUES + i] = gpu_histogram_bulk[bid * TILES_NUM * HIST_GRAYSCALE_VALUES+i] * (HIST_GRAYSCALE_VALUES - 1) / (TILE_WIDTH * TILE_WIDTH);
    }
    __syncthreads();

    // Step 4: invoke interpolate_device to perform interpolation on the image
    interpolate_device(maps + bid * TILES_NUM * HIST_GRAYSCALE_VALUES, all_in + bid * IMG_SIZE, all_out + bid * IMG_SIZE);
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // define task serial memory buffers
    uchar *gpu_in_img;
    uchar *gpu_out_img;
    uchar *gpu_maps;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    //allocate GPU memory for a single input image, a single output image, and maps
    cudaMalloc(&context->gpu_in_img, IMG_WIDTH * IMG_HEIGHT * sizeof(uchar));
    cudaMalloc(&context->gpu_out_img, IMG_WIDTH * IMG_HEIGHT * sizeof(uchar));
    cudaMalloc(&context->gpu_maps, TILE_COUNT * TILE_COUNT * 256 * sizeof(uchar));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    
    for(int i = 0; i < N_IMAGES; i++) {
        cudaMemcpy(context->gpu_in_img, images_in + i * IMG_WIDTH * IMG_HEIGHT, IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
        process_image_kernel<<<1, THREADS_IN_KERNEL>>>(context->gpu_in_img, context->gpu_out_img, context->gpu_maps);
        cudaMemcpy(images_out + i * IMG_WIDTH * IMG_HEIGHT, context->gpu_out_img, IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //free resources allocated in task_serial_init
    cudaFree(context->gpu_in_img);
    cudaFree(context->gpu_out_img);
    cudaFree(context->gpu_maps);

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // define bulk-GPU memory buffers
    uchar *gpu_in_imgs;
    uchar *gpu_out_imgs;
    uchar *gpu_all_imgs_maps;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //allocate GPU memory for all the input images, output images, and maps
    cudaMalloc(&context->gpu_in_imgs, N_IMAGES * IMG_WIDTH * IMG_HEIGHT * sizeof(uchar));
    cudaMalloc(&context->gpu_out_imgs, N_IMAGES * IMG_WIDTH * IMG_HEIGHT * sizeof(uchar));
    cudaMalloc(&context->gpu_all_imgs_maps, N_IMAGES * TILES_NUM * HIST_GRAYSCALE_VALUES * sizeof(uchar));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //copy all input images from images_in to the GPU memory you allocated
    //invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //copy output images from GPU memory to images_out
    cudaMemcpy(context->gpu_in_imgs, images_in, N_IMAGES * IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
    process_image_kernel_bulk<<<N_IMAGES, BULK_THREADS_IN_KERNEL>>>(context->gpu_in_imgs, context->gpu_out_imgs, context->gpu_all_imgs_maps);
    cudaMemcpy(images_out, context->gpu_out_imgs, N_IMAGES * IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //free resources allocated in gpu_bulk_init
    cudaFree(context->gpu_in_imgs);
    cudaFree(context->gpu_out_imgs);
    cudaFree(context->gpu_all_imgs_maps);

    free(context);
}
