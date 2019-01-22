#include "scenarios.h"
#include <random>
#include <iostream>

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

void lanes(int n_agents, agent *agents, int board_x, int board_y, int agent_radius) {
    vector<vec2> start_pos;
    vector<vec2> dest_pos;
    bool fit;
    vec2 prop;
    uniform_real_distribution<float> dist_x(-agent_radius * n_agents / 15.0f, agent_radius * n_agents / 15.0f);
    uniform_real_distribution<float> dist_y(10 + agent_radius, board_y * 2.0f / 5.0f - 10 - agent_radius);
    int i = n_agents;
    while (i >= 0) {
        float tmp_x = dist_x(gen);
        float tmp_y = dist_y(gen);
        fit = true;
        if (i % 4 == 0) {
            prop = vec2(board_x / 4.0f + tmp_x, tmp_y);
        } else if (i % 4 == 1) {
            prop = vec2(board_x / 4.0f + tmp_x, board_y - tmp_y);
        } else if (i % 4 == 2) {
            prop = vec2(board_x * 3.0f / 4.0f + tmp_x, tmp_y);
        } else if (i % 4 == 3) {
            prop = vec2(board_x * 3.0f / 4.0f + tmp_x, board_y - tmp_y);
        }
        for (const auto &p: start_pos) {
            if (distance(p, prop) < 4.0f * agent_radius) {
                ++i;
                fit = false;
                break;
            }
        }
        if (fit) {
            start_pos.push_back(prop);
            if (i % 2 == 0) {
                dest_pos.emplace_back(prop.x(), prop.y() + board_y * 3.0f / 5.0f);
            } else {
                dest_pos.emplace_back(prop.x(), prop.y() - board_y * 3.0f / 5.0f);
            }
        }
        --i;
    }
    for (i = 0; i < n_agents; ++i) {
        agents[i].set_agent(start_pos[i], dest_pos[i]);
    }
}

void cross(int n_agents, agent *agents, int board_x, int board_y, int agent_radius) {
    vector<vec2> start_pos;
    vector<vec2> dest_pos;
    bool fit;
    vec2 prop;
    uniform_real_distribution<float> dist_x(10 + agent_radius, board_x * 1.0f / 3.0f);
    uniform_real_distribution<float> dist_y(-agent_radius * n_agents / 10.0f, agent_radius * n_agents / 10.0f);
    int i = n_agents;
    while (i >= 0) {
        float tmp = dist_x(gen);
        fit = true;
        if (i % 2) {
            prop = vec2(tmp, tmp + dist_y(gen));
        } else {
            prop = vec2(tmp, board_y - (tmp + dist_y(gen)));
        }
        if (prop.y() < agent_radius || prop.y() > board_y - agent_radius)
            continue;
        for (const auto &p: start_pos) {
            if (distance(p, prop) < 4.0f * agent_radius) {
                ++i;
                fit = false;
                break;
            }
        }
        if (fit) {
            start_pos.push_back(prop);
            if (i % 2)
                dest_pos.push_back(prop + vec2(board_x * 3.0f / 5.0f - (10 + agent_radius),
                                               board_x * 3.0f / 5.0f - 2 * (10 + agent_radius)));
            else
                dest_pos.push_back(prop + vec2(board_x * 3.0f / 5.0f - (10 + agent_radius),
                                               -board_x * 3.0f / 5.0f + 2 * (10 + agent_radius)));
        }
        --i;
    }
    for (i = 0; i < n_agents; ++i) {
        agents[i].set_agent(start_pos[i], dest_pos[i]);
    }
}
