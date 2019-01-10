#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <iostream>

using namespace std;

float rand_float(float min, float max) {
    if (max == min) return max;
    if (max < min) return ((((float) rand()) / (float) RAND_MAX) * (min - max)) + max;
    return ((((float) rand()) / (float) RAND_MAX) * (max - min)) + min;
}

int main(int argc, char const *argv[])
{
    int n_agents, n_generations, board_x, board_y;
    float max_speed;
    float agent_radius;
    cin >> n_agents >> n_generations >> agent_radius >> board_x >> board_y >> max_speed;
    
    agent *agents = new agent[n_agents];
    srand(time(NULL));
    for (int i = 0; i < n_agents; ++i) {
        agents[i].set_agent(rand_float(0, board_x), rand_float(0, board_y), rand_float(0, board_x),
                            rand_float(0, board_y));
        agents[i].vect() = agents[i].vect().normalized();
    }

    run(n_agents, n_generations, agent_radius, max_speed, board_x, board_y, agents);
    return 0;
}