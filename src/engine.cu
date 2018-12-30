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

__global__ void move(agent *agents, agent *agents_out, int n_agents, int board_x, int board_y) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_agents) {
        agents_out[ix] = agents[ix];
    }
    if (ix > n_agents * n_agents) return;
    int first = ix / n_agents;
    int second = ix % n_agents;
//    if (first >= second) return;
//    printf("%d %d\n", ix/n_agents, ix%n_agents);
    //first, second to indeksy ktore bedzie sprawdzal dany watek


    //todo od teraz jedno wielkie todo, tylko zrobione zeby jak kolwiek sie ruszali, na razie bez sprawdzania
    if (ix < n_agents) {
        printf("%d\n", ix);
        agents_out[ix].move();
    }

}

void run(int n_agents, int board_x, int board_y) {
    float thickness = 1; //thickness of every agent
    agent* agents = new agent[n_agents];
    srand (time(NULL));
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

    move << < grid_size, block_size >> > (d_agents, d_agents_out, n_agents, board_x, board_y);

    gpuErrchk(cudaMemcpy(agents, d_agents_out, n_agents * sizeof(agent), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_agents; ++i) {
        printf("%f %f %f %f %f\n", agents[i].x(), agents[i].y(), agents[i].vx(), agents[i].vy(), agents[i].max_speed());
    }

    cudaFree(d_agents);
    cudaFree(d_agents_out);
}