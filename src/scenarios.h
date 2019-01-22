#ifndef CUDAMULTIAGENT_SCENARIOS_H
#define CUDAMULTIAGENT_SCENARIOS_H


#include "agent.h"

void circle(int n_agents, agent *agents, int board_x, int board_y);
void uniform(int n_agents, agent *agents, int board_x, int board_y, int agent_radius);
void simple_cross(agent *agents, int board_x, int board_y);
void cross(int n_agents, agent *agents, int board_x, int board_y, int agent_radius);
void lanes(int n_agents, agent *agents, int board_x, int board_y, int agent_radius);


#endif
