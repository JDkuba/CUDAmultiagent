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

__device__ int distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

__global__ void path(agent *agents, int n_agents, int board_x, int board_y, float max_speed) {  //A* maybe later
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agent *james = &agents[ix];
        james->set_vector(james->d_x()-james->x(), james->d_y()-james->y());
        james->normalize();
    }
}

__global__ void set(agent *agents, int n_agents, int board_x, int board_y, int agent_radius, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= n_agents * n_agents) return;
    //first, second to indeksy ktore bedzie sprawdzal dany watek
    int first = ix / n_agents;
    int second = ix % n_agents;
    agent *first_agent = &agents[first];
    agent *second_agent = &agents[second];
    if (first < second) {
        if (distance(first_agent->x() + first_agent->vx() * max_speed,
                     first_agent->y() + first_agent->vy() * max_speed,
                     second_agent->x() + second_agent->vx() * max_speed,
                     second_agent->y() + second_agent->vy() * max_speed) < agent_radius) {
            first_agent->set_vector(-first_agent->vx(), -first_agent->vy());
            second_agent->set_vector(-second_agent->vx(), -second_agent->vy());
        }
    }
}


__global__ void move(agent *agents, int n_agents, int board_x, int board_y, float max_speed) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        //jesli blisko to sie nie ruszaja
        agent *james = &agents[ix];
        float next_x = james->x() + james->vx() * max_speed;
        float next_y = james->y() + james->vy() * max_speed;
        if (next_x > board_x || next_x < 0) james->vx() = -james->vx();
        if (next_y > board_y || next_y < 0) james->vy() = -james->vy();
        agents[ix].move(max_speed);
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
        agents[i].normalize();
    }

    // printAgentsStartPositions(agents, n_agents);
    writeAgenstStartPosition(agents, n_agents);

    int block_size = 1024;
    int grid_size = (n_agents * n_agents) / block_size + 1;     //number of pairs

    agent *d_agents;

    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    for (int i = 0; i < n_generations; ++i) {
        path<<<grid_size, block_size>>>(d_agents, n_agents, board_x, board_y, max_speed);
        cudaDeviceSynchronize();
        set<<<grid_size, block_size>>>(d_agents, n_agents, board_x, board_y, agent_radius, max_speed);
        cudaDeviceSynchronize();
        move<<<grid_size, block_size>>>(d_agents, n_agents, board_x, board_y, max_speed);
        gpuErrchk(cudaMemcpy(agents, d_agents, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));

        // printAgentsPositions(agents, n_agents);
        writeAgentsPositions(agents, n_agents);
    }
    
    closeFiles();

    cudaFree(d_agents);
}