#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

float rand_float(int b) {
    return ((float) rand()) / (float) RAND_MAX * b;
}

__device__ int distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}


__global__ void move(agent *agents, agent *agents_out, int n_agents, int board_x, int board_y, int thickness) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agents_out[ix] = agents[ix];
    }
    if (ix > n_agents * n_agents) return;
    //first, second to indeksy ktore bedzie sprawdzal dany watek
    int first = ix / n_agents;
    int second = ix % n_agents;
    agent first_agent_in = agents[first];
    agent second_agent_in = agents[second];
    agent first_agent_out = agents[first];
    agent second_agent_out = agents[second];
    if (first != second) {
        if (distance(first_agent_in.x() + first_agent_in.vx() * first_agent_in.max_speed(),
                     first_agent_in.y() + first_agent_in.vy() * first_agent_in.max_speed(),
                     second_agent_in.x() + second_agent_in.vx() * second_agent_in.max_speed(),
                     second_agent_in.y() + second_agent_in.vy() * second_agent_in.max_speed()) < thickness) {
            first_agent_out.set_vector(-first_agent_in.vx(), -first_agent_in.vy());
            second_agent_out.set_vector(-second_agent_in.vx(), -second_agent_in.vy());
        }
    }
    if (ix < n_agents) {
        agent james = agents_out[ix];
        float next_x = james.x() + james.vx() * james.max_speed();
        float next_y = james.y() + james.vy() * james.max_speed();
        if (next_x < board_x && next_x > 0 && next_y > 0 && next_y < board_y) agents_out[ix].move();
    }
}

void run(int n_agents, int board_x, int board_y) {
    int n_generations = 10000;
    float thickness = 0.02; //thickness of every agent
    agent *agents = new agent[n_agents];
    srand(time(NULL));
    for (int i = 0; i < n_agents; ++i) {
        agents[i].set_agent(rand_float(board_x), rand_float(board_y), rand_float(1), rand_float(1), 0.1);
        agents[i].normalize();
    }
    for (int i = 0; i < n_agents; ++i) {
        printf("%f %f %f %f %f\n", agents[i].x(), agents[i].y(), agents[i].vx(), agents[i].vy(), agents[i].max_speed());
    }
    printf("\n");

    int block_size = 1024;
    int grid_size = (n_agents * n_agents) / block_size + 1;     //number of pairs

    agent *d_agents;
    agent *d_agents_out;

    gpuErrchk(cudaMalloc(&d_agents, n_agents * sizeof(agent)));
    gpuErrchk(cudaMalloc(&d_agents_out, n_agents * sizeof(agent)));
    gpuErrchk(cudaMemcpy(d_agents, agents, n_agents * sizeof(agent), cudaMemcpyHostToDevice));

    for (int i = 0; i < n_generations; ++i) {
        move<<<grid_size, block_size>>>(d_agents, d_agents_out, n_agents, board_x, board_y, thickness);
        gpuErrchk(cudaMemcpy(d_agents, d_agents_out, n_agents * sizeof(agent), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(agents, d_agents_out, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));
    }
    gpuErrchk(cudaMemcpy(agents, d_agents_out, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_agents; ++i) {
        printf("%f %f %f %f %f\n", agents[i].x(), agents[i].y(), agents[i].vx(), agents[i].vy(), agents[i].max_speed());
    }

    cudaFree(d_agents);
    cudaFree(d_agents_out);
}