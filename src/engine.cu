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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#endif

bool DEBUG_FLAG = false;
void set_debug(){ DEBUG_FLAG = true; }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)*/

constexpr float ALFA = M_PI;
constexpr int RANDOM_VECTORS_NUM = 150;
constexpr int VECTOR_PACE_NUM = 20;
constexpr int RESOLUTION = 40;
constexpr int RESOLUTION_SHIFT = RESOLUTION + 1;
constexpr int MAX_BOARDS = 10000;
constexpr float COLLISION_RADIUS_MULT = 10;
constexpr float ALFA_EPS = ALFA / RESOLUTION;
constexpr int MULTIPLIER = 1000000000 / (10 * MAX_BOARDS);
constexpr float APEX_SHIFT = 0.1;
constexpr float CONTAINS_EPS = 0.01;

__device__ vo compute_simple_vo(const agent &A, const agent &B, int agent_radius, float max_speed) {
    vo obs;
    vec2 pAB = B.pos() - A.pos();
    vec2 pABn = pAB.normalized();
    float R = 2.0f * agent_radius;

    float theta = asin(R / (pAB.length()));
    obs.apex = A.pos() + B.svect();
    if((obs.apex - A.pos()).is_zero()){
        obs.apex = A.pos()-(pABn*APEX_SHIFT);
    }
    obs.left = pABn.rotate(theta);
    obs.right = pABn.rotate(-theta);

    return obs;
}

__global__ void find_path(agent *agents, int n_agents, float agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents)
        return;

    agent &james = agents[ix];
    james.set_vector((james.dest() - james.pos()).normalized());
    if (james.finished(agent_radius)) {
        james.set_speed(0);
        james.set_vector({0.0f, 0.0f});
    }
    else
        james.set_speed(max_speed);
}

__global__ void set_vo(agent *agents, vo *obstacles, int n_agents, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = ix / n_agents;
    int i2 = ix % n_agents;
    if (ix >= n_agents * n_agents || i1 == i2)
        return;

    agent &A = agents[i1];
    agent &B = agents[i2];
    if (distance(A.pos(), B.pos()) < (COLLISION_RADIUS_MULT * agent_radius))
        obstacles[ix] = compute_simple_vo(agents[i1], agents[i2], agent_radius, max_speed);
    else
        obstacles[ix].set_invalid();
}

__global__ void clear_vo(vo *obstacles, int n_agents) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents * n_agents) return;
    obstacles[ix].set_invalid();
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
    agent &A = agents[agent_idx];

    angle = (ALFA / 2) * 2 * (0.5f - (resolution_idx / RESOLUTION));
    pace = pace_idx / (VECTOR_PACE_NUM-1);

    vec2 vector = A.vect().rotate(angle).normalized();
    vector = vector * (max_speed * pace);
    vector = vector + A.pos();
    vectors[VECTOR_PACE_NUM * RESOLUTION_SHIFT * agent_idx + threadIdx.x * VECTOR_PACE_NUM + threadIdx.y] = vector;
}

__global__ void check_vectors(vo *obstacles, vec2 *vectors, int n_agents) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = ix / n_agents;
    int i2 = ix % n_agents;

    if (ix >= n_agents * n_agents || i1 == i2 || obstacles[ix].invalid())
        return;

    vo obs = obstacles[ix];

    const int max = (i1 + 1) * VECTOR_PACE_NUM * RESOLUTION_SHIFT;
    for (int i = i1*VECTOR_PACE_NUM * RESOLUTION_SHIFT; i < max; i++) {
        if (obs.contains(vectors[i], CONTAINS_EPS)) {
            vectors[i].set_invalid();
        }
    }
}

__global__ void apply_best_velocities(agent *agents, int *best_distances, vec2* vectors, int n_agents, float max_speed){
    int agentIdx = blockIdx.x;
    if (agentIdx >= n_agents)
        return;

    float best_dist = INT32_MAX;
    agent Agent = agents[agentIdx];
    vec2 best_p(Agent.pos().x(), Agent.pos().y());

    for (int i = 0; i < VECTOR_PACE_NUM * RESOLUTION_SHIFT; i++) {
        vec2 current = vectors[VECTOR_PACE_NUM * RESOLUTION_SHIFT * agentIdx + i];
        if (current.is_invalid()) {
            continue;
        }
        float dist = distance_without_sqrt(current, Agent.dest());
        if (dist < best_dist) {
            best_dist = dist;
            best_p = current;
        }
    }

    best_p = best_p - Agent.pos();
    agents[agentIdx].set_speed(best_p.length());
    agents[agentIdx].set_vector(best_p.normalized());
}

__global__ void move(agent *agents, int n_agents, int move_divider) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents)
        return;

    if (!agents[ix].isdead()) agents[ix].move(move_divider);
}

void print_details(agent *agents, vo *obstacles, int n) {
    bool ifprint = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && !obstacles[i * n + j].invalid()) {
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

void run(int n_agents, int n_generations, float agent_radius, float max_speed, int board_x, int board_y, int move_divider, int fake_move_divider, agent* agents) {
    if (board_x > MAX_BOARDS || board_y > MAX_BOARDS)
        std::cout << "Exceeded MAX_BOARDS size. Bugs may occur\n";
//    cudaSetDevice(2);
    openFiles();
    putMetadataToFile(n_agents, n_generations, agent_radius, board_x, board_y);
    writeAgenstStartPosition(agents, n_agents);

    int rays_number = (n_agents * n_agents * RESOLUTION_SHIFT);
    int pairs_number = n_agents * n_agents;
    agent *d_agents;
    vo *d_obstacles;
    vo *h_obstacles = new vo[n_agents * n_agents];      //to remove
    int *d_best_distances;
    unsigned long long *d_best_intersects;

    vec2* d_vectors;

    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&d_obstacles, n_agents * n_agents * sizeof(vo)));
    gpuErrchk(cudaMalloc(&d_best_distances, rays_number * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_best_intersects, rays_number * sizeof(long long)));

    gpuErrchk(cudaMalloc(&d_vectors, n_agents * RESOLUTION_SHIFT * VECTOR_PACE_NUM * sizeof(vec2)));

    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size_agents = n_agents / block_size + 1;
    int grid_size_pairs = pairs_number / block_size + 1;
    int grid_size_rays = rays_number / block_size + 1;

    for (int i = 0; i < n_generations; ++i) {
        if(i % fake_move_divider == 0){
            clear_best_distances<<<grid_size_rays, block_size>>>(d_best_distances, rays_number);
            clear_vo<<<grid_size_pairs, block_size>>>(d_obstacles, n_agents);
            gpuErrchk(cudaDeviceSynchronize());
            find_path<<<grid_size_agents, block_size>>>(d_agents, n_agents, agent_radius, max_speed);
            gpuErrchk(cudaDeviceSynchronize());

            set_vo<<<grid_size_pairs, block_size>>>(d_agents, d_obstacles, n_agents, agent_radius, max_speed);
            gpuErrchk(cudaDeviceSynchronize());
            if (DEBUG_FLAG) gpuErrchk(cudaMemcpy(h_obstacles, d_obstacles, n_agents * n_agents * sizeof(vo), cudaMemcpyDeviceToHost));

            generate_vectors<<<n_agents, {RESOLUTION_SHIFT, VECTOR_PACE_NUM}>>>(d_vectors, d_agents, n_agents, max_speed);
            gpuErrchk(cudaDeviceSynchronize());

            check_vectors<<<grid_size_pairs, block_size>>>(d_obstacles, d_vectors, n_agents);
            gpuErrchk(cudaDeviceSynchronize());

            apply_best_velocities<<<n_agents, 1>>>(d_agents, d_best_distances, d_vectors, n_agents, max_speed);
            gpuErrchk(cudaDeviceSynchronize());
        }

        if (DEBUG_FLAG) print_details(agents, h_obstacles, n_agents);
        move<<<grid_size_pairs, block_size>>>(d_agents, n_agents, move_divider);

        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));
        writeAgentsPositions(agents, n_agents);
    }

    closeFiles();
    delete h_obstacles;
    cudaFree(d_agents);
    cudaFree(d_obstacles);
    cudaFree(d_best_distances);
    cudaFree(d_best_intersects);
}