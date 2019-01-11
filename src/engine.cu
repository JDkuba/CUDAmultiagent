#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"
#include "IOUtils.h"
#include <stdint.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr float ALFA = M_PI;
constexpr int RESOLUTION = 180; 
constexpr int MAX_BOARDS = 10000;
constexpr int MULTIPLIER = 1000000000/(10*MAX_BOARDS); 

__device__ vo compute_simple_vo(const agent& A, const agent& B, int agent_radius){
    vo obs;
    // obs.apex = A.pos() + B.svect();
    obs.apex = A.pos() + B.svect() + (A.vect()*agent_radius);
    vec2 pAB = B.pos() - A.pos();
    float theta = asin(2 * agent_radius / pAB.length());
    obs.left = pAB.normalized().rotate(theta);
    obs.right = pAB.normalized().rotate(-theta);
    return obs;
}

__global__ void find_path(agent *agents, int n_agents, float agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if(ix >= n_agents)
        return;

    agent &james = agents[ix];
    james.set_vector((james.dest() - james.pos()).normalized());
    if(james.finished(agent_radius) or james.isdead())
        james.set_speed(0);
    else{
        james.set_speed(max_speed);
    }
}

__global__ void set_vo(agent *agents, vo *obstacles, int n_agents, int agent_radius) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = ix / n_agents;
    int i2 = ix % n_agents;
    if (ix >= n_agents * n_agents || i1 == i2) 
        return;

    obstacles[ix] = compute_simple_vo(agents[i1], agents[i2], agent_radius);
}

__global__ void clear_best_distances(int *best_distances, int rays_number){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= rays_number)
        return;
    best_distances[i] = INT32_MAX;
}

__global__ void get_best_intersects(agent *agents, vo *obstacles, int *best_distances, unsigned long long *best_intersects, int n_agents){
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = ix / n_agents;
    int i2 = ix % n_agents;
    if (ix >= n_agents * n_agents || i1 == i2) 
        return;

    agent &A = agents[i1];
    vo &obs = obstacles[ix];

    ray rays[2];
    rays[0] = obs.left_ray(); 
    rays[1] = obs.right_ray();
    vec2 left_angle = A.dest().rotate(ALFA/2);
    float alfa_eps = ALFA/RESOLUTION;
    float distA = A.dist(); 
    vec2 p[2];
    float d[2];
    bool A_in_obs = obs.contains(A.pos());

    for (int i = 0; i <= RESOLUTION; ++i){
        ray v_ray(A.pos(), left_angle.rotate(i*alfa_eps));
        for (int j = 0; j < 2; ++j){
            p[j] = intersect_rays(rays[j], v_ray);
            if (p[j].invalid()){
                if(A_in_obs)
                    d[j] = INT32_MAX;
                else{
                    p[j] = v_ray.dir * distA;
                    d[j] = distance(p[j], A.pos());
                }
            }
            else
                d[j] = distance(p[j], A.pos());
        }

        if(d[1] < d[0]){
            p[0] = p[1];
            d[0] = d[1];
        }

        if(d[0] == INT32_MAX)
            continue;

        // p[0] - nearest point from intersection of given ray with obstacle to A.dest() 
        unsigned long long vect;
        float* ptr = reinterpret_cast<float*>(&vect);
        *ptr = p[0].x();
        *(ptr + 1) = p[0].y();
        int pd = d[0]*MULTIPLIER; // multiply to give approximation
        int old = atomicMin(&best_distances[RESOLUTION*i1 + i], pd);
        if(pd < old) // some minor 'swaps' may occur
            atomicExch(&best_intersects[RESOLUTION*i1 + i], vect);
    }
    
}


__global__ void apply_best_intersects(agent *agents, int *best_distances, unsigned long long *best_intersects, int n_agents, float max_speed){
    int ai = blockDim.x * blockIdx.x + threadIdx.x;
    if (ai >= n_agents)
        return;

    agent &A = agents[ai];
    int min_dist = INT32_MAX;
    int best_i = -1;
    for (int i = 0; i <= RESOLUTION; ++i){
        int d = best_distances[RESOLUTION*ai+i];
        if (d < min_dist){
            best_i = i;
            min_dist = d;
        }
    }

    if(best_i == -1){
        A.set_speed(0);
        return;
    }

    float* ptr = reinterpret_cast<float*>(&best_intersects[RESOLUTION*ai + best_i]);
    float px = *ptr;
    float py = *(ptr+1);

    vec2 v(px, py);
    v = v - A.pos();
    A.set_speed(min(v.length(), max_speed));
    A.set_vector(v.normalized());
}

__global__ void move(agent *agents, vo *obstacles, int n_agents) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents)
        return;

    agents[ix].move();
}

void run(int n_agents, int n_generations, float agent_radius, float max_speed, int board_x, int board_y, agent* agents) {
    if(board_x > MAX_BOARDS || board_y > MAX_BOARDS)
        std::cout << "Exceeded MAX_BOARDS size. Bugs may occur\n";

    openFiles();
    putMetadataToFile(n_agents, n_generations, agent_radius, board_x, board_y);
    writeAgenstStartPosition(agents, n_agents);

    int rays_number = (n_agents * n_agents * (RESOLUTION+1));
    int pairs_number = n_agents * n_agents;
    agent *d_agents;
    vo *d_obstacles;
    int *d_best_distances;
    unsigned long long *d_best_intersects;
    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&d_obstacles, n_agents * n_agents * sizeof(vo)));
    gpuErrchk(cudaMalloc(&d_best_distances, rays_number * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_best_intersects, rays_number * sizeof(long long)));

    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size_agents = n_agents / block_size + 1;
    int grid_size_pairs = pairs_number / block_size + 1;
    // int grid_size_rays = rays_number / block_size + 1;

    for (int i = 0; i < n_generations; ++i) {
        find_path<<<grid_size_agents, block_size>>>(d_agents, n_agents, agent_radius, max_speed);
        gpuErrchk(cudaDeviceSynchronize());
        set_vo<<<grid_size_pairs, block_size>>>(d_agents, d_obstacles, n_agents, agent_radius);
        gpuErrchk(cudaDeviceSynchronize());
        clear_best_distances<<<grid_size_pairs, block_size>>>(d_best_distances, rays_number);
        gpuErrchk(cudaDeviceSynchronize());
        get_best_intersects<<<grid_size_pairs, block_size>>>(d_agents, d_obstacles, d_best_distances, d_best_intersects, n_agents);
        gpuErrchk(cudaDeviceSynchronize());
        apply_best_intersects<<<grid_size_agents, block_size>>>(d_agents, d_best_distances, d_best_intersects, n_agents, max_speed);
        gpuErrchk(cudaDeviceSynchronize());
        move<<<grid_size_pairs, block_size>>>(d_agents, d_obstacles, n_agents);

        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));
        writeAgentsPositions(agents, n_agents);
    }

    closeFiles();
    cudaFree(d_agents);
    cudaFree(d_obstacles);
}