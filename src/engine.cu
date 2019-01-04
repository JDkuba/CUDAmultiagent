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

float rand_float(float min, float max) {
    if (max == min) return max;
    if (max < min) return ((((float) rand()) / (float) RAND_MAX) * (min - max)) + max;
    return ((((float) rand()) / (float) RAND_MAX) * (max - min)) + min;
}

__device__ float distance(vec2 a, vec2 b) {
    return sqrt((a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()));
}

__device__ float det(vec2 a, vec2 b) {
    return a.x()*b.y() - a.y()*b.x();
}

__device__ vo compute_vo(agent* A, agent* B, int agent_radius, float max_speed) {
    vo obs;
    vec2 pAB = B->pos() - A->pos();
    vec2 pABn = pAB.normalized();
    int rAB = 2 * agent_radius;

    vec2 apex, left, right;

    // no collision
    if (pAB.length() > rAB){
        apex = (A->vect() + B->vect()) / 2.0f;

        float theta = asin(rAB / pAB.length());
        right = pAB.rotate(-theta);
        left = pAB.rotate(theta);

        float sin2theta = 2.0f * sin(theta) * cos(theta);
        float s;
        if (det(B->pos() - A->pos(), A->vect() - B->vect()) > 0.0f){
            s = 0.5f * det(A->vect() + B->vect(), left) / sin2theta;
            apex = B->vect() + s * right;
        }
        else {
            s = 0.5f * det(A->vect() + B->vect(), right) / sin2theta;
            apex = B->vect() + s * left;
        }
    }

    // Collsion
    else {
        apex = 0.5f * (A->vect() + B->vect() - pAB.normalized() * (rAB - pAB.normalized()) / max_speed);
        right = vec2(pAB.normalized().y(),-pAB.normalized().x());
        left = 0 - right;
    }

    obs.apex = apex;
    obs.left = left;
    obs.right = right;
    return obs;
}

__global__ void path(agent *agents, int n_agents, int board_x, int board_y, float max_speed) {  //A* maybe later
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agent *james = &agents[ix];
        james->set_vector((james->dest()-james->pos()).normalized());
    }
}

__global__ void set(agent *agents, vo *obstacles, int n_agents, int board_x, int board_y, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents * n_agents) return;
    //first, second to are indices of considered agents
    int first = ix / n_agents;
    int second = ix % n_agents;
    agent *first_agent = &agents[first];
    agent *second_agent = &agents[second];
    if (first != second) {
        //todo, agents should pass each other
        //something like that http://gamma.cs.unc.edu/RVO/icra2008.pdf
        //or that https://www.youtube.com/watch?v=Hc6kng5A8lQ
        obstacles[ix] = compute_vo(first_agent, second_agent, agent_radius, max_speed);
    }
}


__global__ void move(agent *agents, vo *obstacles, int n_agents, int board_x, int board_y, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        //if agent is close to destination he should stay there
        //todo check obstacles and choose best vector
        agent *james = &agents[ix];
        if (!distance(james->pos(), james->dest()) < agent_radius) {
            agents[ix].move(max_speed);
        }
    }
}

void run(int n_agents, int n_generations, float agent_radius, int board_x, int board_y) {
    float max_speed = 1;

    openFiles();
    putMetadataToFile(n_agents, n_generations, agent_radius, board_x, board_y);

    agent *agents = new agent[n_agents];
    srand(time(NULL));
    for (int i = 0; i < n_agents; ++i) {
        agents[i].set_agent(rand_float(0, board_x), rand_float(0, board_y), rand_float(0, board_x), rand_float(0, board_y));
        agents[i].vect() = agents[i].vect().normalized();
    }

    // printAgentsStartPositions(agents, n_agents);
    writeAgenstStartPosition(agents, n_agents);

    int block_size = 1024;
    int grid_size = (n_agents * n_agents) / block_size + 1;     //number of pairs

    agent *d_agents;
    vo *obstacles;

    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&obstacles, n_agents * n_agents * sizeof(vo)));
    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    for (int i = 0; i < n_generations; ++i) {
        path<<<grid_size, block_size>>>(d_agents, n_agents, board_x, board_y, max_speed);
        cudaDeviceSynchronize();
        set<<<grid_size, block_size>>>(d_agents, obstacles, n_agents, board_x, board_y, agent_radius, max_speed);
        cudaDeviceSynchronize();
        move<<<grid_size, block_size>>>(d_agents, obstacles, n_agents, board_x, board_y, agent_radius, max_speed);
        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));

        // printAgentsPositions(agents, n_agents);
        writeAgentsPositions(agents, n_agents);
    }
    
    closeFiles();

    cudaFree(d_agents);
    cudaFree(obstacles);
}