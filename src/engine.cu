#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "IOUtils.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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

__device__ int distance(vec2 a, vec2 b) {
    return sqrt((a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()));
}

__device__ vec2 compute_vo(agent* a1, agent* a2) {
    vec2 v;
    return v;
}

__global__ void path(agent *agents, int n_agents, int board_x, int board_y, float max_speed) {  //A* maybe later
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agent *james = &agents[ix];
        james->set_vector(james->dest()-james->pos());
        james->vect().normalize();
    }
}

__global__ void set(agent *agents, vec2 *obstacles, int n_agents, int board_x, int board_y, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents * n_agents) return;
    //first, second to indeksy ktore bedzie sprawdzal dany watek
    int first = ix / n_agents;
    int second = ix % n_agents;
    agent *first_agent = &agents[first];
    agent *second_agent = &agents[second];
    if (first != second) {
        //todo, agents should pass each other
        //something like that http://gamma.cs.unc.edu/RVO/icra2008.pdf
        //or that https://www.youtube.com/watch?v=Hc6kng5A8lQ
        obstacles[ix] = compute_vo(first_agent, second_agent);
    }
}


__global__ void move(agent *agents, vec2 *obstacles, int n_agents, int board_x, int board_y, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        //jesli blisko to niech sie nie ruszaja
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
        agents[i].vect().normalize();
    }

    // printAgentsStartPositions(agents, n_agents);
    writeAgenstStartPosition(agents, n_agents);

    int block_size = 1024;
    int grid_size = (n_agents * n_agents) / block_size + 1;     //number of pairs

    agent *d_agents;
    vec2 *obstacles;

    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&obstacles, n_agents * n_agents * sizeof(vec2)));
    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    for (int i = 0; i < n_generations; ++i) {
        path<<<grid_size, block_size>>>(d_agents, n_agents, board_x, board_y, max_speed);
        cudaDeviceSynchronize();
//        set<<<grid_size, block_size>>>(d_agents, obstacles, n_agents, board_x, board_y, agent_radius, max_speed);
        cudaDeviceSynchronize();
        move<<<grid_size, block_size>>>(d_agents, obstacles, n_agents, board_x, board_y, agent_radius, max_speed);
        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));

        // printAgentsPositions(agents, n_agents);
        writeAgentsPositions(agents, n_agents);
    }
    
    closeFiles();

    cudaFree(d_agents);
}