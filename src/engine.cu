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

bool DEBUG_FLAG = false;
void set_debug(){ DEBUG_FLAG = true; }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr float ALFA = M_PI;
constexpr int RESOLUTION = 180;
constexpr int RESOLUTION_SHIFT = RESOLUTION + 1;
constexpr int MAX_BOARDS = 10000;
constexpr float COLLISION_RADIUS_MULT = 10;
constexpr float ALFA_EPS = ALFA / RESOLUTION;
constexpr int MULTIPLIER = 1000000000 / (10 * MAX_BOARDS);


__device__ vo compute_simple_vo(const agent &A, const agent &B, int agent_radius, float max_speed) {
    vo obs;
    vec2 pAB = B.pos() - A.pos();
    vec2 pABn = pAB.normalized();
    float R = 2.0f * agent_radius;

    float theta = asin(R / (pAB.length()));
    obs.apex = A.pos() + B.svect();
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
    if (james.finished(agent_radius))
        james.set_speed(0);
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

__global__ void get_worst_intersects(agent *agents, vo *obstacles, int *best_distances,
                                     unsigned long long *best_intersects, int n_agents, float max_speed, int agent_radius) {

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = ix / n_agents;
    int i2 = ix % n_agents;

    if (ix >= n_agents * n_agents || i1 == i2 || obstacles[ix].invalid())
        return;

    vo &obs = obstacles[ix];
    agent &A = agents[i1];

    ray rays[2];
    rays[0] = obs.left_ray();
    rays[1] = obs.right_ray();
    vec2 left_angle = A.vect().rotate(ALFA / 2);

    float distA = A.dist();
    vec2 p[2]; // points of intersection v_ray with angle
    float d[2]; // distance from p[i] to A.pos() - we need to get closest point

//    if (obs.contains(A.pos())) printf("ERROR!\n");
    for (int i = 0; i <= RESOLUTION; ++i) {
        ray v_ray(A.pos(), left_angle.rotate(-i * ALFA_EPS).normalized());
        for (int j = 0; j < 2; ++j) {
            p[j] = intersect_rays(rays[j], v_ray);
            if (p[j].invalid()) {
                p[j] = v_ray.pos + (v_ray.dir * max_speed);
            }
            d[j] = min(max_speed, distance(p[j], A.pos()));
            p[j] = v_ray.pos + (v_ray.dir * d[j]);
        }

        if (d[1] < d[0]) {
            p[0] = p[1];
            d[0] = d[1];
        }

        unsigned long long point;
        float *ptr = reinterpret_cast<float *>(&point);
        *ptr = p[0].x();
        *(ptr + 1) = p[0].y();
        int point_distance = d[0] * MULTIPLIER; // multiply to give approximation
        int old = atomicMin(&best_distances[RESOLUTION_SHIFT * i1 + i], point_distance);
        if (point_distance < old) // some minor 'swaps' may occur
            atomicExch(&best_intersects[RESOLUTION_SHIFT * i1 + i], point);
    }
}


__global__ void apply_best_velocities(agent *agents, int *best_distances, unsigned long long *intersects, int n_agents, float max_speed){
    int ai = blockDim.x * blockIdx.x + threadIdx.x;
    if (ai >= n_agents)
        return;

    agent &A = agents[ai];
    float best_dist = 0;
    bool assigned = false;
    vec2 best_p, p;
    for (int i = 0; i <= RESOLUTION; ++i) {
        if (best_distances[RESOLUTION_SHIFT * ai + i] == INT32_MAX) { // ray is free
            vec2 v = A.vect().rotate(ALFA / 2).rotate(-i * ALFA_EPS).normalized();
            p = A.pos() + (v * max_speed);
        } else {
            float *ptr = reinterpret_cast<float *>(&intersects[RESOLUTION_SHIFT * ai + i]);
            p.set(*ptr, *(ptr + 1));
        }
        float dist = distance(p, A.dest());
        if (!assigned || dist < best_dist) {
            best_dist = dist;
            best_p = p;
            assigned = true;
        }
    }

    best_p = best_p - A.pos();
    A.set_speed(best_p.length());
    A.set_vector(best_p.normalized());
}

__global__ void move(agent *agents, int n_agents, int move_divider) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents)
        return;

    if (!agents[ix].isdead()) agents[ix].move(move_divider);
}

void print_details(agent *agents, vo *obstacles, int n) {
    bool ifprint = false;
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
    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&d_obstacles, n_agents * n_agents * sizeof(vo)));
    gpuErrchk(cudaMalloc(&d_best_distances, rays_number * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_best_intersects, rays_number * sizeof(long long)));

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

            get_worst_intersects<<<grid_size_pairs, block_size>>>(d_agents, d_obstacles, d_best_distances, d_best_intersects, n_agents, max_speed, agent_radius);
            gpuErrchk(cudaDeviceSynchronize());

            apply_best_velocities<<<grid_size_agents, block_size>>>(d_agents, d_best_distances, d_best_intersects, n_agents, max_speed);
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