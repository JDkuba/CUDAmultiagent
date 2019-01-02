#include "agent.h"
#include "engine.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int main(int argc, char const *argv[])
{
    int agents, generations, boardx, boardy;
    float agent_radius;
    cin >> agents >> generations >> agent_radius >> boardx >> boardy;
    // run(agents, generations, agent_radius, boardx, boardy)
    run(agents, generations, agent_radius, boardx, boardy);
    return 0;
}