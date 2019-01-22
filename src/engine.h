#ifndef ENGINE_H
#define ENGINE_H
#include "agent.h"

void set_debug();
void run(int n_agents, int n_generations, float agent_radius, float max_speed, int board_x, int board_y, int move_divider, agent* agents);

#endif