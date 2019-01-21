#include "scenarios.h"
#include <random>

using namespace std;

random_device rd;
mt19937 gen(rd());


void circle(int n_agents, agent *agents, int board_x, int board_y) {
    vec2 v, op;
    uniform_real_distribution<float> dist(-20.0f, 20.0f);
    float angle = 2.0f * 3.141f / n_agents;
    v.set(0, (board_y - 50) / 2.0f);
    op.set(0, -(board_y - 50) / 2.0f);
    for (int i = 0; i < n_agents; ++i) {
        vec2 rand_vec = vec2(dist(gen), dist(gen));
        agents[i].set_agent(v + vec2(board_x / 2.0f, board_y / 2.0f) + rand_vec,
                            op + vec2(board_x / 2.0f, board_y / 2.0f) - rand_vec);
        v = v.rotate(angle);
        op = op.rotate(angle);
    }
}

void uniform(int n_agents, agent *agents, int board_x, int board_y, int agent_radius) {
    vector<vec2> start_pos;
    vector<vec2> dest_pos;
    uniform_real_distribution<float> dist_x(0, board_x);
    uniform_real_distribution<float> dist_y(0, board_y);
    bool fit;
    vec2 prop;
    for (int i = 0; i < n_agents; ++i) {
        prop = vec2(dist_x(gen), dist_y(gen));
        fit = true;
        for (const auto &p: start_pos) {
            if (distance(p, prop) < 2.5f * agent_radius) {
                --i;
                fit = false;
                break;
            }
        }
        if (fit) start_pos.push_back(prop);
    }

    for (int i = 0; i < n_agents; ++i) {
        prop = vec2(dist_x(gen), dist_y(gen));
        fit = true;
        for (const auto &p: dest_pos) {
            if (distance(p, prop) < 2.5f * agent_radius) {
                --i;
                fit = false;
                break;
            }
        }
        if (fit) dest_pos.push_back(prop);
    }

    for (int i = 0; i < n_agents; ++i) {
        agents[i].set_agent(start_pos[i], dest_pos[i]);
    }
}

//2 agents
void simple_cross(agent *agents, int board_x, int board_y) {
    agents[0].set_agent(100, 100, board_x - 100, board_y - 100);
    agents[1].set_agent(99, board_y - 99, board_x - 99, 99);
}