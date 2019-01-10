#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"
#include "IOUtils.h"

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

__device__ vo compute_vo(const agent& A, const agent& B, int agent_radius, float max_speed) {
    vo obs;
    vec2 pAB = B.pos() - A.pos();
    vec2 pABn = pAB.normalized();
    int rAB = 2 * agent_radius;

    vec2 apex, left, right;

    if (pAB.length() > rAB) {
        float theta = asin(rAB / pAB.length());
        right = pABn.rotate(-theta);
        left = pABn.rotate(theta);

        float sin2theta = 2.0f * sin(theta) * cos(theta);
        float s;
        if (det(B.pos() - A.pos(), A.vect() - B.vect()) > 0.0f) {
            s = 0.5f * det(A.vect() + B.vect(), left) / sin2theta;
            apex = B.vect() + s * right;
        } else {
            s = 0.5f * det(A.vect() + B.vect(), right) / sin2theta;
            apex = B.vect() + s * left;
        }
    } else {
        apex = 0.5f * (A.vect() + B.vect() - pABn * (rAB - pAB.normalized()) / max_speed);
        right = vec2(pABn.y(), -pABn.x());
        left = 0 - right;
    }

    obs.apex = apex;
    obs.left = left;
    obs.right = right;
    return obs;
}

__global__ void path(agent *agents, int n_agents, float max_speed, float agent_radius) {  //A* maybe later
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agent &james = agents[ix];
        if(james.finished(agent_radius))
            james.set_speed(0);
        else{
            james.set_vector((james.dest() - james.pos()).normalized());
            james.set_speed(max_speed);
        }
    }
}

__global__ void set_vo(agent *agents, vo *obstacles, int n_agents, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents * n_agents) return;
    //first, second to are indices of considered agents
    int first = ix / n_agents;
    int second = ix % n_agents;
    agent &agent1 = agents[first];
    agent &agent2 = agents[second];
    if (first != second) {
        //todo, agents should pass each other
        //something like that http://gamma.cs.unc.edu/RVO/icra2008.pdf
        //or that https://www.youtube.com/watch?v=Hc6kng5A8lQ
        obstacles[ix] = compute_vo(agent1, agent2, agent_radius, max_speed);
    }
}


__global__ void move(agent *agents, vo *obstacles, int n_agents) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agents[ix].move();
    }
}

void run(int n_agents, int n_generations, float agent_radius, float max_speed, int board_x, int board_y, agent* agents) {
    openFiles();
    putMetadataToFile(n_agents, n_generations, agent_radius, board_x, board_y);
    writeAgenstStartPosition(agents, n_agents);

    agent *d_agents;
    vo *obstacles;
    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&obstacles, n_agents * n_agents * sizeof(vo)));
    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size = (n_agents * n_agents) / block_size + 1;
    for (int i = 0; i < n_generations; ++i) {
        path<<<grid_size, block_size>>>(d_agents, n_agents, max_speed, agent_radius);
        cudaDeviceSynchronize();
        set_vo<<<grid_size, block_size>>>(d_agents, obstacles, n_agents, agent_radius, max_speed);
        cudaDeviceSynchronize();
        move<<<grid_size, block_size>>>(d_agents, obstacles, n_agents);

        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));
        writeAgentsPositions(agents, n_agents);
    }

    closeFiles();
    cudaFree(d_agents);
    cudaFree(obstacles);
}