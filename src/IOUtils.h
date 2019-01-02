#ifndef IOUTILS_H
#define IOUTILS_H
#include "agent.h"

constexpr char METADATA_FILE_PATH[] = "metadata.out";
constexpr char AGENTS_POSITIONS_PATH[] = "agents_positions.out";

void printAgentsPositions(agent* agents, int n_agents);

void printAgentsStartPositions(agent* agents, int n_agents);

void putMetadataToFile(int n_agents, int n_generations, float agent_radius, int board_x, int board_y);

void openFiles();

void writeAgenstStartPosition(agent* agents, int n_agents);

void writeAgentsPositions(agent* agents, int n_agents);

void closeFiles();

#endif