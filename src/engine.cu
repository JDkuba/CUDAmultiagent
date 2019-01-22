#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"
#include "IOUtils.h"
#include <stdint.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#define __shared__
#define __syncthreads
#endif

bool DEBUG_FLAG = false;
void set_debug(){ DEBUG_FLAG = true; }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr float ALFA = static_cast<float>(M_PI);
constexpr int VECTOR_PACE_NUM = 20;
constexpr int RESOLUTION = 50;
constexpr int RESOLUTION_SHIFT = RESOLUTION + 1;
constexpr int MAX_BOARDS = 10000;
constexpr float COLLISION_RADIUS_MULT = 25;
constexpr float APEX_SHIFT = 0.1;
constexpr float CONTAINS_EPS = 0.01;

__device__ vo compute_vo(const agent &A, const agent &B, int agent_radius, float max_speed) {
    vo obs;
    vec2 pAB = B.pos() - A.pos();
    vec2 pABn = pAB.normalized();
    float R = 2.0f * agent_radius;

    float theta = static_cast<float>(asin(R / (pAB.length())));
    obs.apex = A.pos() + B.svect();
    if((obs.apex - A.pos()).is_zero())
        obs.apex = A.pos()-(pABn*APEX_SHIFT);
    obs.left = pABn.rotate(theta);
    obs.right = pABn.rotate(-theta);
    return obs;
}

__global__ void find_path(agent *agents, int n_agents, float agent_radius, float max_speed) {
    int agent_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (agent_idx >= n_agents)
        return;

    agent &A = agents[agent_idx];
    A.set_vector((A.dest() - A.pos()).normalized());
    if (A.finished(agent_radius)) {
        A.set_speed(0);
        A.set_vector({0.0f, 0.0f});
    }
    else
        A.set_speed(max_speed);
}

__global__ void set_vo(agent *agents, vo *obstacles, int n_agents, int agent_radius, float max_speed) {
    int vo_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int ia1 = vo_idx / n_agents;
    int ia2 = vo_idx % n_agents;
    if (vo_idx >= n_agents * n_agents || ia1 == ia2)
        return;

    agent &A = agents[ia1];
    agent &B = agents[ia2];
    if (distance(A.pos(), B.pos()) < (COLLISION_RADIUS_MULT * agent_radius))
        obstacles[vo_idx] = compute_vo(agents[ia1], agents[ia2], agent_radius, max_speed);
    else
        obstacles[vo_idx].set_invalid();
}

__global__ void clear_obstacles(vo *obstacles, int n_agents) {
    int vo_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vo_idx >= n_agents * n_agents)
        return;

    obstacles[vo_idx].set_invalid();
}

__global__ void clear_best_distances(int *best_distances, int rays_number) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= rays_number)
        return;

    best_distances[i] = INT32_MAX;
}

__global__ void generate_vectors(vec2 *vectors, agent *agents, int n_agents, float max_speed) {
    int agent_idx = blockIdx.x;
    float resolution_idx = threadIdx.x;
    float pace_idx = threadIdx.y;

    float angle, pace;
    agent A = agents[agent_idx];

    angle = (ALFA / 2) * 2 * (0.5f - (resolution_idx / RESOLUTION));
    pace = pace_idx / (VECTOR_PACE_NUM-1);

    vec2 vector = A.vect().rotate(angle).normalized();
    vector = vector * (max_speed * pace);
    vector = vector + A.pos();
    vectors[VECTOR_PACE_NUM * RESOLUTION_SHIFT * agent_idx + threadIdx.x * VECTOR_PACE_NUM + threadIdx.y] = vector;
}

__global__ void check_vectors(vo *obstacles, vec2 *vectors, int n_agents) {
    int agent_idx = blockIdx.x;
    int vector_idx = threadIdx.x;

    vec2 vector = vectors[agent_idx * blockDim.x + vector_idx];
    
    __shared__ vo shared_obs[1024];
    
    if (vector_idx < n_agents) {
        shared_obs[vector_idx] = obstacles[agent_idx * n_agents + vector_idx];
    }

    __syncthreads();

    for (int i = 0; i < n_agents; i++) {
        if (i != agent_idx && !shared_obs[i].is_invalid() && shared_obs[i].contains(vector, CONTAINS_EPS)) {
            vectors[agent_idx * blockDim.x + vector_idx].set_invalid();
            break;
        }
    }
}

__global__ void apply_best_velocities(agent *agents, vec2* vectors, int n_agents){
    int agent_idx = blockIdx.x;
    int vector_idx =  threadIdx.x;

    agent agent = agents[agent_idx];

    extern __shared__ std::pair<float, int> shared_distances[];
    vec2 vector = vectors[agent_idx * blockDim.x + vector_idx];

    if (vector.is_invalid()) {
        shared_distances[vector_idx].first = INT32_MAX;
    } else {
        shared_distances[vector_idx].first = distance_without_sqrt(vector, agent.dest());
    }

    shared_distances[vector_idx].second = vector_idx;

    __syncthreads();

    for(int i = blockDim.x / 2; i >= 1; i /= 2) {
        if(vector_idx < i) {
            if(shared_distances[vector_idx].first > shared_distances[vector_idx + i].first) {
                shared_distances[vector_idx].first = shared_distances[vector_idx + i].first;
                shared_distances[vector_idx].second = shared_distances[vector_idx + i].second;
            }
        }
        __syncthreads();
    }

    if(vector_idx == 0) {
        vec2 best(0.0f, 0.0f);
        if (distance_without_sqrt(agent.pos(), agent.dest()) > shared_distances[0].first) {
            best = vectors[agent_idx * blockDim.x + shared_distances[0].second] - agent.pos();
        }
        agents[agent_idx].set_speed(best.length());
        agents[agent_idx].set_vector(best.normalized());
    }
}

__global__ void move(agent *agents, int n_agents, int move_divider) {
    int agent_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (agent_idx >= n_agents)
        return;

    if (!agents[agent_idx].isdead()) agents[agent_idx].move(move_divider);
}

void print_details(agent *agents, vo *obstacles, int n) {
    bool ifprint = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && !obstacles[i * n + j].is_invalid()) {
                obstacles[i * n + j].print(i, j);
                ifprint = true;
            }
        }
    }
    if (ifprint) {
        for (int i = 0; i < n; ++i) agents[i].print(i);
        printf("\n");
    }
}

void run(int n_agents, int n_generations, float agent_radius, float max_speed, int board_x, int board_y, int move_divider, agent* agents) {
    auto start = std::chrono::steady_clock::now();

    if (board_x > MAX_BOARDS || board_y > MAX_BOARDS)
        std::cout << "Exceeded MAX_BOARDS size. Bugs may occur\n";
//    cudaSetDevice(2);
    openFiles();
    putMetadataToFile(n_agents, n_generations, agent_radius, board_x, board_y);
    writeAgenstStartPosition(agents, n_agents);

    int pairs_number = n_agents * n_agents;
    agent *d_agents;
    vo *d_obstacles;
    vo *h_obstacles = new vo[n_agents * n_agents];      //to remove
    vec2* d_vectors;

    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&d_obstacles, n_agents * n_agents * sizeof(vo)));
    gpuErrchk(cudaMalloc(&d_vectors, n_agents * RESOLUTION_SHIFT * VECTOR_PACE_NUM * sizeof(vec2)));

    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size_agents = n_agents / block_size + 1;
    int grid_size_pairs = pairs_number / block_size + 1;

    auto stop = std::chrono::steady_clock::now();
    std::cerr << "  Preparation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms\n";

    long long cuda_time = 0;
    long long copying_time = 0;
    for (int i = 0; i < n_generations; ++i) {
        start = std::chrono::steady_clock::now();
        clear_obstacles <<<grid_size_pairs, block_size>>>(d_obstacles, n_agents);
        gpuErrchk(cudaDeviceSynchronize());

        find_path<<<grid_size_agents, block_size>>>(d_agents, n_agents, agent_radius, max_speed);
        gpuErrchk(cudaDeviceSynchronize());

        set_vo<<<grid_size_pairs, block_size>>>(d_agents, d_obstacles, n_agents, agent_radius, max_speed);
        gpuErrchk(cudaDeviceSynchronize());

        generate_vectors<<<n_agents, {RESOLUTION_SHIFT, VECTOR_PACE_NUM}>>>(d_vectors, d_agents, n_agents, max_speed);
        gpuErrchk(cudaDeviceSynchronize());

        check_vectors<<<n_agents, RESOLUTION_SHIFT * VECTOR_PACE_NUM>>>(d_obstacles, d_vectors, n_agents);
        gpuErrchk(cudaDeviceSynchronize());

        apply_best_velocities<<<n_agents, RESOLUTION_SHIFT * VECTOR_PACE_NUM, 
            RESOLUTION_SHIFT * VECTOR_PACE_NUM * sizeof(std::pair<float, int>)>>>(d_agents, d_vectors, n_agents);
        gpuErrchk(cudaDeviceSynchronize());
        stop = std::chrono::steady_clock::now();
        cuda_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();

        start = std::chrono::steady_clock::now();
        if (DEBUG_FLAG)
            print_details(agents, h_obstacles, n_agents);

        move<<<grid_size_agents, block_size>>>(d_agents, n_agents, move_divider);
        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));
        writeAgentsPositions(agents, n_agents);

        stop = std::chrono::steady_clock::now();
        copying_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    }

    std::cerr << "  GPU total time: " << cuda_time << "ms\n";
    std::cerr << "  Copying total time: " << copying_time << "ms\n";

    closeFiles();
    cudaFree(d_agents);
    cudaFree(d_obstacles);
    cudaFree(d_vectors);
}