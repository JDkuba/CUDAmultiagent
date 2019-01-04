#include "IOUtils.h"
#include <fstream>
#include <iostream>
using namespace std;

static ofstream meta_file;
    

void printAgentsPositions(agent* agents, int n_agents){
    for (int i = 0; i < n_agents; ++i)
        cout << agents[i].pos().x() << ' ' << agents[i].pos().y() << '\n';
}

void printAgentsStartPositions(agent* agents, int n_agents){
    cout << n_agents << '\n';
    printAgentsPositions(agents, n_agents);
}

void putMetadataToFile(int n_agents, int n_generations, float agent_radius, int board_x, int board_y){
    ofstream pos_file;
    pos_file.open (METADATA_FILE_PATH, ios::out);
    pos_file << n_agents << ' ' << n_generations << ' ' << agent_radius << ' ' << board_x << ' ' << board_y << '\n';
    pos_file.close();
}

// file.write(reinterpret_cast<const char *>(&num), sizeof(num))

void openFiles(){
    meta_file.open (AGENTS_POSITIONS_PATH, ios::binary);
}

void writeAgenstStartPosition(agent* agents, int n_agents){
    meta_file.write(reinterpret_cast<const char *>(&n_agents), sizeof(n_agents));
    writeAgentsPositions(agents, n_agents);    
}

void writeAgentsPositions(agent* agents, int n_agents){
    for (int i = 0; i < n_agents; ++i)
    {
        int x = round(agents[i].pos().x());
        int y = round(agents[i].pos().y());
        meta_file.write(reinterpret_cast<const char *>(&x), sizeof(x));
        meta_file.write(reinterpret_cast<const char *>(&y), sizeof(y));
    }
}

void closeFiles(){
    meta_file.close();
}