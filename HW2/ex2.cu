#include "ex2.h"
#include <cuda/atomic>
#include <queue>
#include <iostream>
#include <memory>
#include <algorithm>

#define THREADS_IN_KERNEL 1024
#define TILES_NUM (TILE_COUNT * TILE_COUNT)
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)
#define HIST_GRAYSCALE_VALUES 256
#define TILE_HEIGHT (TILE_WIDTH)
#define INTERPOLATE_DEVICE_SMEM_BYTES 1024
#define SLOTS_NUM 16

__device__ void lock(cuda::std::atomic<int>* l) {
    do {
        // Spin until lock is free
        while (l->load(cuda::memory_order_relaxed) == 1) {}
    // Attempt atomic locking (TAS)
    } while (l->exchange(1, cuda::memory_order_acq_rel) == 1);
}

__device__ void unlock(cuda::std::atomic<int>* l) {
    l->store(0, cuda::memory_order_release);
}
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

__device__
void process_image(uchar *in, uchar *out, uchar* maps) {

    //Allocate shared memory for the TB
    __shared__ int gpu_histogram[TILES_NUM * HIST_GRAYSCALE_VALUES];

    // Clear the global histogram from the previous kernel launch
    for(int i = threadIdx.x; i < TILES_NUM * HIST_GRAYSCALE_VALUES; i += blockDim.x) {
        gpu_histogram[i] = 0;
    }
    __syncthreads();

    // Compute histogram per tile
    for(unsigned pixel_idx = threadIdx.x; pixel_idx < IMG_SIZE; pixel_idx += blockDim.x) {
        int tile_x = (pixel_idx % IMG_WIDTH) / TILE_WIDTH;
        int tile_y = pixel_idx / (IMG_WIDTH * TILE_HEIGHT);
        int tile_index = tile_x + (tile_y * TILE_COUNT);
        
        int pixel_value = in[pixel_idx]; // 'pixel_idx' maps directly to the global index
        
        atomicAdd(&gpu_histogram[tile_index * HIST_GRAYSCALE_VALUES + pixel_value], 1);
    }
    __syncthreads();

    // Compute prefix sum of the histogram per tile
    int thread_groups = blockDim.x / HIST_GRAYSCALE_VALUES;
    int group_id = threadIdx.x / HIST_GRAYSCALE_VALUES;

    for(int tile_index = 0; tile_index < TILES_NUM; tile_index += thread_groups) {
        
        int my_tile = tile_index + group_id;
        
        // A thread is active only if its assigned tile actually exists 
        bool is_active = (my_tile < TILES_NUM);

        if (is_active) {
            prefix_sum(&gpu_histogram[my_tile * HIST_GRAYSCALE_VALUES], HIST_GRAYSCALE_VALUES);
        }
    }
    __syncthreads();

    // Create a map for each tile using the given formula
    for(int i = threadIdx.x; i < TILES_NUM * HIST_GRAYSCALE_VALUES; i += blockDim.x) {
        maps[i] = gpu_histogram[i] * (HIST_GRAYSCALE_VALUES - 1) / (TILE_WIDTH * TILE_WIDTH);
    }
    __syncthreads();

    // invoke interpolate_device to perform interpolation on the image
    interpolate_device(maps, in, out);
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    cudaStream_t streams[STREAM_COUNT];
    uchar *gpu_in_imgs;
    uchar *gpu_out_imgs;
    uchar *gpu_maps;
    // index i for stream i, queue of img_ids that wasn't dequeued.
    std::queue<int> streams_to_img_ids[STREAM_COUNT]; 


public:
    streams_server()
    {

        //create cuda streams
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK( cudaStreamCreate(&streams[i]) );
        }

        //allocate GPU memory for all the input images, output images, and maps
        CUDA_CHECK( cudaMalloc(&gpu_in_imgs, STREAM_COUNT * IMG_WIDTH * IMG_HEIGHT * sizeof(uchar)) );
        CUDA_CHECK( cudaMalloc(&gpu_out_imgs, STREAM_COUNT * IMG_WIDTH * IMG_HEIGHT * sizeof(uchar)) );
        CUDA_CHECK( cudaMalloc(&gpu_maps, STREAM_COUNT * TILES_NUM * HIST_GRAYSCALE_VALUES * sizeof(uchar)) );
    }
    
    ~streams_server() override
    {
        CUDA_CHECK( cudaFree(gpu_in_imgs) );
        CUDA_CHECK( cudaFree(gpu_out_imgs) );
        CUDA_CHECK( cudaFree(gpu_maps) );
        
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK( cudaStreamDestroy(streams[i]) );
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        //use cudaStreamQuery to check if there is a stream is available for a new request. if so, place tasks on the stream.
        for(int i=0; i < STREAM_COUNT; i++) {
            if(cudaStreamQuery(streams[i]) == cudaSuccess) {

                streams_to_img_ids[i].push(img_id);
                
                //copy input image to GPU
                CUDA_CHECK( cudaMemcpyAsync(&gpu_in_imgs[i * IMG_SIZE], img_in, IMG_SIZE * sizeof(uchar), cudaMemcpyHostToDevice, streams[i]) );

                //invoke kernel to process the image
                process_image_kernel<<<1, THREADS_IN_KERNEL, 0, streams[i]>>>(&gpu_in_imgs[i * IMG_SIZE], &gpu_out_imgs[i * IMG_SIZE], &gpu_maps[i * TILES_NUM * HIST_GRAYSCALE_VALUES]);

                //copy output image back to CPU
                CUDA_CHECK( cudaMemcpyAsync(img_out, &gpu_out_imgs[i * IMG_SIZE], IMG_SIZE * sizeof(uchar), cudaMemcpyDeviceToHost, streams[i]) );
                                
                return true; 
            }
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        //go over all streams
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            cudaError_t status = cudaStreamQuery(streams[i]);
            switch (status) {
            case cudaSuccess:
                // return the img_id of the request that was completed.
                if(!streams_to_img_ids[i].empty()) {
                    *img_id = streams_to_img_ids[i].front();
                    streams_to_img_ids[i].pop();
                    return true;
                }
            default:
                continue; //stream is not ready yet, check the next stream
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}


// Helper function to calculate maximum concurrent threadblocks
// threads_per_block can be 256,512,1024
int get_max_concurrent_blocks(int threads_per_block) 
{
    int deviceId = 1;
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, deviceId);
    
    if (status != cudaSuccess) {
        std::cerr << "Failed to get device properties." << std::endl;
        return -1;
    }

    // Calculate the maximum blocks allowed by Thread Capacity
    int threadblocks_by_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;

    // Calculate the maximum blocks allowed by Shared Memory
    unsigned smem_per_threadblock = INTERPOLATE_DEVICE_SMEM_BYTES + 
                                (TILES_NUM * HIST_GRAYSCALE_VALUES * sizeof(int)) +
                                 sizeof(int) + (2 * sizeof(uchar*)) + (2* sizeof(bool));
    int threadblocks_by_smem = prop.sharedMemPerMultiprocessor / smem_per_threadblock;

    // Calculate the maximum blocks allowed by Registers
    unsigned registers_per_thread = 32; //given
    int total_registers_per_block = threads_per_block * registers_per_thread;
    int threadblocks_by_regs = prop.regsPerMultiprocessor / total_registers_per_block;

    // Calculate max_threadblocks_per_sm by the minimum of the constraints (The Bottleneck)
    int max_threadblocks_per_sm = std::min({
        threadblocks_by_threads, 
        threadblocks_by_smem, 
        threadblocks_by_regs, 
    });

    // Multiply by total SMs to get the GPU-wide concurrent threadblocks
    int total_concurrent_blocks = max_threadblocks_per_sm * prop.multiProcessorCount;

    return total_concurrent_blocks;
}
struct Slot
{
    int img_id;
    uchar* img_in;
    uchar* img_out;
};
class Queue {
private:
    Slot* buffer;
    int capacity;
    int mask;
    cuda::atomic<int> head;
    cuda::atomic<int> tail;

public:
    // Constructor
    __host__ __device__ Queue(int cap, Slot* preallocated_buffer) : 
        capacity(cap), 
        mask(cap - 1), 
        head(0), 
        tail(0),
        buffer(preallocated_buffer){}
        
    // Producer: enqueues at head
    __host__ __device__ bool enqueue(const Slot& item) {
        int current_head = head.load(cuda::memory_order_relaxed);
        int current_tail = tail.load(cuda::memory_order_acquire);

        if (current_head - current_tail >= capacity) {
            return false; 
        }

        buffer[current_head & mask] = item;
        
        head.store(current_head + 1, cuda::memory_order_release);
        return true;

    }   

    // Consumer: Dequeues at tail
    __host__ __device__ bool dequeue(Slot* item) {
        int current_tail = tail.load(cuda::memory_order_relaxed);
        int current_head = head.load(cuda::memory_order_acquire);

        if (current_head == current_tail) {
            return false; 
        }

        *item = buffer[current_tail & mask];
        
        tail.store(current_tail + 1, cuda::memory_order_release);
        return true;
    }
};
__global__ void process_queue_kernel(Queue* cpu_to_gpu_queue,Queue* gpu_to_cpu_queue, cuda::std::atomic<int>* lock_queue_in,cuda::std::atomic<int>* lock_queue_out, uchar* maps, volatile bool* terminate) {
    __shared__ Slot request;
    __shared__ bool has_request;
    __shared__ bool should_terminate;

    while (true) {
        // Lock the input queue and try to dequeue a request
        if (threadIdx.x == 0) {
            should_terminate = *terminate;
            has_request = false;
            if(!should_terminate) {
                lock(lock_queue_in);
                if (cpu_to_gpu_queue->dequeue(&request)) {
                    has_request = true;
                }
                unlock(lock_queue_in);
            }
        }

        __syncthreads();

        if(should_terminate) {
            break; // All threads exit the loop and terminate the kernel
        }

        if (has_request) {
            // Process the image
            process_image(request.img_in, request.img_out, maps + blockIdx.x * TILES_NUM * HIST_GRAYSCALE_VALUES ); 
            
            __syncthreads();

            // Lock the output queue and enqueue the result
            if (threadIdx.x == 0) {
                lock(lock_queue_out);
                gpu_to_cpu_queue->enqueue(request); 
                unlock(lock_queue_out);
            }
        } 
    }
}

class queue_server : public image_processing_server
{
private:
    char* pinned_host_buffer;
    Slot* pinned_cpu_gpu_slots_buffer;
    Slot* pinned_gpu_cpu_slots_buffer;
    Queue* cpu_to_gpu_queue;
    Queue* gpu_to_cpu_queue;
    cuda::std::atomic<int>* lock_queue_in;
    cuda::std::atomic<int>* lock_queue_out;
    unsigned threadblocks_concurrently;
    volatile bool* terminate;
    uchar* maps;
public:
    queue_server(int threads)
    {
        unsigned max_concurrent_threadblocks = get_max_concurrent_blocks(threads);
        
        // Calculate power of 2 capacity
        int capacity = SLOTS_NUM * max_concurrent_threadblocks;

        int power_of_two_capacity = 1;
        while (power_of_two_capacity < capacity) {
            power_of_two_capacity *= 2;
        }
        

        CUDA_CHECK( cudaMallocHost(&terminate, sizeof(bool)) );
        *terminate = false;

        CUDA_CHECK( cudaMallocHost(&pinned_host_buffer, sizeof(Queue) * 2) );
        CUDA_CHECK( cudaMalloc(&maps, max_concurrent_threadblocks * TILES_NUM * HIST_GRAYSCALE_VALUES *sizeof(uchar)) );
        CUDA_CHECK( cudaMallocHost(&pinned_cpu_gpu_slots_buffer, sizeof(Slot) * power_of_two_capacity) );
        CUDA_CHECK( cudaMallocHost(&pinned_gpu_cpu_slots_buffer, sizeof(Slot) * power_of_two_capacity) );
        
        cpu_to_gpu_queue = new (pinned_host_buffer) Queue(power_of_two_capacity, pinned_cpu_gpu_slots_buffer);
        gpu_to_cpu_queue = new (pinned_host_buffer + sizeof(Queue)) Queue(power_of_two_capacity, pinned_gpu_cpu_slots_buffer);


        CUDA_CHECK( cudaMalloc(&lock_queue_in, sizeof(cuda::std::atomic<int>)) );
        CUDA_CHECK( cudaMemset(lock_queue_in, 0, sizeof(cuda::std::atomic<int>)) );
        CUDA_CHECK( cudaMalloc(&lock_queue_out, sizeof(cuda::std::atomic<int>)) );
        CUDA_CHECK( cudaMemset(lock_queue_out, 0, sizeof(cuda::std::atomic<int>)) );
        
        // Invoke kernel asynchronously
        process_queue_kernel<<<max_concurrent_threadblocks, threads>>>(cpu_to_gpu_queue, gpu_to_cpu_queue, lock_queue_in, lock_queue_out, maps, terminate);
    }
    ~queue_server() override
    {
        *terminate = true;
        CUDA_CHECK( cudaDeviceSynchronize() );

        // Explicitly call queues destructors
        cpu_to_gpu_queue->~Queue();
        gpu_to_cpu_queue->~Queue();

        // Free CPU-side memory
        CUDA_CHECK( cudaFreeHost(pinned_host_buffer) );
        CUDA_CHECK( cudaFreeHost(pinned_cpu_gpu_slots_buffer) );
        CUDA_CHECK( cudaFreeHost(pinned_gpu_cpu_slots_buffer) );
        CUDA_CHECK( cudaFreeHost((void*)terminate) );

        // Free GPU-side memory
        CUDA_CHECK( cudaFree(maps) );
        CUDA_CHECK( cudaFree(lock_queue_in) );
        CUDA_CHECK( cudaFree(lock_queue_out) );
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        Slot new_request = {img_id, img_in, img_out};
        return cpu_to_gpu_queue->enqueue(new_request);
    }

    bool dequeue(int *img_id) override
    {
        Slot completed_task;
        
        // Query the GPU-CPU queue without blocking
        if (gpu_to_cpu_queue->dequeue(&completed_task)) {
            
            *img_id = completed_task.img_id;
            return true;
        }
        
        // Return false if the queue is currently empty
        return false;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
