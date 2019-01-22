#include "agent.h"
#include "engine.h"
#include "smath.h"
#include "scenarios.h"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <assert.h>

using namespace std;

static void testRayIntersect();

static void testReinpretCast();

static void testVects();

static void test() {
    testRayIntersect();
    testReinpretCast();
    testVects();
}


int main(int argc, char const *argv[]) {
    srand(time(NULL));
    if (argc > 1 and strcmp(argv[1], "--test") == 0) {
        cout << "testing...\n";
        test();
        return 0;
    }

    if(argc < 3){
        cout << "signature is: --testType n_agents <--debug>\n";
        return 1;   
    }

    if(argc == 4 and strcmp(argv[3], "--debug") == 0)
        set_debug();

    int agent_radius;
    float max_speed;
    int n_agents, n_generations, board_x, board_y, move_divider;
    cin >> n_generations >> agent_radius >> board_x >> board_y >> max_speed >> move_divider;
    n_agents = atoi(argv[2]);

    printf("GENERATING POSITIONS...\n");
    fflush(stdout);

    auto *agents = new agent[n_agents];
    if(strcmp(argv[1], "--random") == 0)
        uniform(n_agents, agents, board_x, board_y, agent_radius);
    else if(strcmp(argv[1], "--circle") == 0)
        circle(n_agents, agents, board_x, board_y);
    else if(strcmp(argv[1], "--cross") == 0)
        cross(n_agents, agents, board_x, board_y, agent_radius);
    else if(strcmp(argv[1], "--simple_cross") == 0){
        n_agents = 2;
        simple_cross(agents, board_x, board_y);
    }

    printf("SIMULATING...\n");
    fflush(stdout);

    run(n_agents, n_generations, agent_radius, max_speed, board_x, board_y, move_divider, agents);
    return 0;
}

static void testRayIntersect() {
    {
        ray ray1(0, 0, 0, 1);
        ray ray2(2, 3, -1, 0);
        vec2 p = intersect_rays(ray1, ray2);
        assert(p.x() == 0 and p.y() == 3);
        cout << p.x() << ' ' << p.y() << '\n';
    }
    {
        ray ray1(0, 0, 0, 1);
        ray ray2(0, 0, 1, 0);
        vec2 p = intersect_rays(ray1, ray2);
        assert(p.x() == 0 and p.y() == 0);
        cout << p.x() << ' ' << p.y() << '\n';
    }
    {
        ray ray1(0, 1, 0, 1);
        ray ray2(1, 1, 0, 1);
        vec2 p = intersect_rays(ray1, ray2);
        assert(isnan(p.x()) and isnan(p.y()));
        cout << p.x() << ' ' << p.y() << '\n';
    }
    cout << "rayTest: OK\n";
}

static void testVects() {
    {
        vec2 v1(0, 1);
        vec2 v2(1, 0);
        assert(angle(v1, v2) < 0);
    }
    cout << "vectsTEst: OK\n";
}

static void testReinpretCast() {
    unsigned long long point;
    float *ptr = reinterpret_cast<float *>(&point);
    *ptr = 25.0f;
    *(ptr + 1) = 41.0f;
    float *ptr1 = reinterpret_cast<float *>(&point);
    cout << *ptr1 << ' ' << *(ptr1 + 1) << '\n';
}