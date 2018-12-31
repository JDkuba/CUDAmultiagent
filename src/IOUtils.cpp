#include "IOUtils.h"
#include <fstream>
#include <iostream>
using namespace std;

void printAgentsPositions(agent* agents, int n_agents){
    for (int i = 0; i < n_agents; ++i)
        cout << agents[i].x() << ' ' << agents[i].y() << '\n';
}

void printAgentsStartPositions(agent* agents, int n_agents){
    cout << n_agents << '\n';
    printAgentsPositions(agents, n_agents);
}

void putMetadataToFile(int n_agents, float agent_radius, int board_x, int board_y){
    ofstream pos_file;
    pos_file.open (METADATA_FILE_PATH, ios::out);
    pos_file << n_agents << ' ' << agent_radius << ' ' << board_x << ' ' << board_y << '\n';
    pos_file.close();
}