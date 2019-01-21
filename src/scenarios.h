#ifndef CUDAMULTIAGENT_SCENARIOS_H
#define CUDAMULTIAGENT_SCENARIOS_H


#include "agent.h"

void circle_scenario(int n_agents, agent *agents, int board_x, int board_y);
void random_scenario(int n_agents, agent *agents, int board_x, int board_y);
void cross_scenario(agent *agents, int board_x, int board_y);

#endif
