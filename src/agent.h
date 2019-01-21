#ifndef PROJEKT_AGENT_H
#define PROJEKT_AGENT_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include <cstdio>
#include "smath.h"

class agent {
    vec2 p;         //(x,y) position
    vec2 v;      //vector should be normed, so after every change of vector, we have to call normalize() method
    vec2 d;
    float s;
    bool dead = false;
public:

    HD inline vec2 &pos() { return p; }

    HD inline vec2 &vect() { return v; }

    HD inline vec2 &dest() { return d; }

    HD inline vec2 pos() const { return p; }

    HD inline vec2 vect() const { return v; }

    HD inline vec2 dest() const { return d; }

    HD inline float speed() const { return s; }

    HD inline vec2 svect() const { return v * s; }

    HD inline float dist() const { return distance(p, d); }

    HD agent() {}

    HD agent(float x, float y, float d_x, float d_y) {
        set_agent(x, y, d_x, d_y);
    }

    HD void set_agent(float x, float y, float d_x, float d_y) {
        p.set(x, y);
        d.set(d_x, d_y);
    }

    HD void set_agent(const vec2 &pos, const vec2 &dest) {
        p = pos;
        d = dest;
    }

    HD void set_vector(vec2 vect) { v = vect; }

    HD void set_speed(float speed) { s = speed; }

    HD void move(int move_divider) { p = p + v * (s / move_divider); }

    HD inline bool finished(float my_radius) const {
        return distance(p, d) < my_radius;
    }

    HD bool isdead() const {
        return dead;
    }

    HD void killme() {
        dead = true;
    }

    HD void print(int id) {
        printf("agent: %d, p: (%.5f, %.5f), v: (%+.5f, %+.5f), d: (%.5f, %.5f)\n", id, p.x(), p.y(), v.x(), v.y(),
               d.x(), d.y());
    }
};

#endif