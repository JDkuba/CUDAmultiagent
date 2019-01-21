#include "agent.h"
#include "engine.h"
#include "smath.h"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <assert.h>

using namespace std;

float rand_float(float min, float max) {
    if (max == min) return max;
    if (max < min) return ((((float) rand()) / (float) RAND_MAX) * (min - max)) + max;
    return ((((float) rand()) / (float) RAND_MAX) * (max - min)) + min;
}

static void testRayIntersect();

static void testReinpretCast();

static void testVects();

static void test() {
    testRayIntersect();
    testReinpretCast();
    testVects();
}

void circle_scenario(int n_agents, agent *agents, int board_x, int board_y) {
    vec2 v, op;
    float angle = 2.0f * 3.141f / n_agents;
    v.set(0, (board_y - 50) / 2.0f);
    op.set(0, -(board_y - 50) / 2.0f);
    for (int i = 0; i < n_agents; ++i) {
        vec2 tmp = vec2(rand_float(-25.0f, 25.0f), rand_float(-25.0f, 25.0f));
        agents[i].set_agent(v + vec2(board_x / 2.0f, board_y / 2.0f) + tmp,
                            op + vec2(board_x / 2.0f, board_y / 2.0f) - tmp);
        v = v.rotate(angle);
        op = op.rotate(angle);
    }
}

void random_scenario(int n_agents, agent *agents, int board_x, int board_y) {
    for (int i = 0; i < n_agents; ++i) {
        agents[i].set_agent(rand_float(0, board_x), rand_float(0, board_y), rand_float(0, board_x), rand_float(0, board_y));
        agents[i].vect() = agents[i].vect().normalized();
    }
}

//2 agents
void cross_scenario(agent *agents, int board_x, int board_y) {
    agents[0].set_agent(100, 100, board_x - 100, board_y - 100);
    agents[1].set_agent(99, board_y - 99, board_x - 99, 99);
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

    float agent_radius, max_speed;
    int n_agents, n_generations, board_x, board_y, move_divider, fake_move_divider;
    cin >> n_generations >> agent_radius >> board_x >> board_y >> max_speed >> move_divider >> fake_move_divider;
    n_agents = atoi(argv[2]);

    auto *agents = new agent[n_agents];
    if(strcmp(argv[1], "--random") == 0)
        random_scenario(n_agents, agents, board_x, board_y);
    else if(strcmp(argv[1], "--circle") == 0)
        circle_scenario(n_agents, agents, board_x, board_y);
    else if(strcmp(argv[1], "--cross") == 0){
        n_agents = 2;
        cross_scenario(agents, board_x, board_y);
    }

    run(n_agents, n_generations, agent_radius, max_speed, board_x, board_y, move_divider, fake_move_divider, agents);
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