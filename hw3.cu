// Your cuda program :)#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#define pi 3.1415926535897932384626433832795f

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;
typedef glm::mat3 mat3;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__constant__ float d_power = 8.0f;
__constant__ float d_md_iter = 24.0f;
__constant__ float d_ray_step = 10000.0f;
__constant__ float d_shadow_step = 1500.0f;
__constant__ float d_step_limiter = 0.2f;
__constant__ float d_ray_multiplier = 0.1f;
__constant__ float d_bailout = 2.0f;
__constant__ float d_eps = 0.0005f;
__constant__ float d_FOV = 1.5f;
__constant__ float d_far_plane = 100.0f;
__constant__ int d_AA = 3;

__device__ float md(vec3 p, float& trap) {
    vec3 v = p;
    float dr = 1.f;
    float r = glm::length(v);
    trap = r;

    for (int i = 0; i < d_md_iter; ++i) {
        float theta = atan2f(v.y, v.x) * d_power;
        float phi = asinf(v.z / r) * d_power;
        dr = d_power * __powf(r, d_power - 1.f) * dr + 1.f;
        v = p + __powf(r, d_power) *
                    vec3(__cosf(theta) * __cosf(phi), __cosf(phi) * __sinf(theta), -__sinf(phi));
        
        trap = glm::min(trap, r);
        r = glm::length(v);
        if (r > d_bailout) break;
    }
    return 0.5f * logf(r) * r / dr;
}

__device__ float map(vec3 p, float& trap, int& ID) {
    vec2 rt = vec2(__cosf(pi / 2.f), __sinf(pi / 2.f));
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) * p;
    ID = 1;
    return md(rp, trap);
}

__device__ float map(vec3 p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

__device__ vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2.f * pi * (c * t + d));
}

__device__ float softshadow(vec3 ro, vec3 rd, float k, vec3 camera_pos) {
    float res = 1.0f;
    float t = 0.f;
    for (int i = 0; i < d_shadow_step; ++i) {
        float h = map(ro + rd * t);
        res = glm::min(res, k * h / t);
        if (res < 0.02f) return 0.02f;
        t += glm::clamp(h, .001f, d_step_limiter);
    }
    return glm::clamp(res, .02f, 1.f);
}

__device__ vec3 calcNor(vec3 p) {
    vec2 e = vec2(d_eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),
                          map(p + e.yxy()) - map(p - e.yxy()),
                          map(p + e.yyx()) - map(p - e.yyx())));
}

__device__ float trace(vec3 ro, vec3 rd, float& trap, int& ID) {
    float t = 0.f;
    float len = 0.f;

    for (int i = 0; i < d_ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
        if (glm::abs(len) < d_eps || t > d_far_plane) break;
        t += len * d_ray_multiplier;
    }
    return t < d_far_plane ? t : -1.f;
}

__global__ void render_kernel(unsigned char* image, 
                              int width, int height,
                              vec3 camera_pos, vec3 target_pos) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  
    int i = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i >= height || j >= width) return;
    
    vec2 iResolution = vec2((float)width, (float)height);
    
    float fcol_r = 0.0f;
    float fcol_g = 0.0f;
    float fcol_b = 0.0f;
    
    for (int m = 0; m < d_AA; ++m) {
        for (int n = 0; n < d_AA; ++n) {
            vec2 p = vec2((float)j, (float)i) + vec2((float)m, (float)n) / (float)d_AA;
            
            vec2 uv = (-iResolution.xy() + 2.f * p) / iResolution.y;
            uv.y *= -1.f;
            
            vec3 ro = camera_pos;
            vec3 ta = target_pos;
            vec3 cf = glm::normalize(ta - ro);
            vec3 cs = glm::normalize(glm::cross(cf, vec3(0.f, 1.f, 0.f)));
            vec3 cu = glm::normalize(glm::cross(cs, cf));
            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + d_FOV * cf);
            
            float trap;
            int objID;
            float d = trace(ro, rd, trap, objID);
            
            vec3 col(0.f);
            vec3 sd = glm::normalize(camera_pos);
            vec3 sc = vec3(1.f, .9f, .717f);
            
            if (d < 0.f) {
                col = vec3(0.f);
            } else {
                vec3 pos = ro + rd * d;
                vec3 nr = calcNor(pos);
                vec3 hal = glm::normalize(sd - rd);
                
                col = pal(trap - .4f, vec3(.5f), vec3(.5f), vec3(1.f), vec3(.0f, .1f, .2f));
                vec3 ambc = vec3(0.3f);
                float gloss = 32.f;
                
                float amb = (0.7f + 0.3f * nr.y) * 
                            (0.2f + 0.8f * glm::clamp(0.05f * log(trap), 0.0f, 1.0f));
                float sdw = softshadow(pos + .001f * nr, sd, 16.f, camera_pos);
                float dif = glm::clamp(glm::dot(sd, nr), 0.f, 1.f) * sdw;
                float spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0.f, 1.f), gloss) * dif;
                
                vec3 lin(0.f);
                lin += ambc * (.05f + .95f * amb);
                lin += sc * dif * 0.8f;
                col *= lin;
                
                col = glm::pow(col, vec3(.7f, .9f, 1.f));
                col += spe * 0.8f;
            }
            
            col = glm::clamp(glm::pow(col, vec3(.4545f)), 0.f, 1.f);
            fcol_r += col.r;
            fcol_g += col.g;
            fcol_b += col.b;
        }
    }
    
    fcol_r /= (float)(d_AA * d_AA);
    fcol_g /= (float)(d_AA * d_AA);
    fcol_b /= (float)(d_AA * d_AA);
    
    fcol_r *= 255.0f;
    fcol_g *= 255.0f;
    fcol_b *= 255.0f;
    
    int idx = i * width * 4 + j * 4;
    image[idx + 0] = (unsigned char)fcol_r;
    image[idx + 1] = (unsigned char)fcol_g;
    image[idx + 2] = (unsigned char)fcol_b;
    image[idx + 3] = 255;
}

void write_png(const char* filename, unsigned char* raw_image, int width, int height) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    assert(argc == 10);
    
    vec3 camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    vec3 target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    int width = atoi(argv[7]);
    int height = atoi(argv[8]);
    
    printf("Rendering %dx%d image...\n", width, height);
    
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    unsigned char* h_image = new unsigned char[image_size];
    
    unsigned char* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, image_size));
    
    dim3 blockSize(16, 16); 
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("Grid size: %dx%d, Block size: %dx%d\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(kernel_start));
    
    render_kernel<<<gridSize, blockSize>>>(d_image, width, height, camera_pos, target_pos);
    
    // Record stop event
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate kernel execution time
    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop));
    
    CUDA_CHECK(cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost));
    
    write_png(argv[9], h_image, width, height);
    
    printf("Image saved to %s\n", argv[9]);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("\n=== Execution Time ===\n");
    printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
    printf("Total execution time: %lld ms\n", total_duration.count());
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaFree(d_image));
    delete[] h_image;
    
    return 0;
}