#ifndef PROJEKT_AGENT_H
#define PROJEKT_AGENT_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include "smath.h"

class agent {
    vec2 p;         //(x,y) position
    vec2 v;      //vector should be normed, so after every change of vector, we have to call normalize() method
    vec2 d;
    float s;
public:
    
    HD inline vec2 &pos() { return p; }

    HD inline vec2 &vect() { return v; }

    HD inline vec2 &dest() { return d; }

    HD inline vec2 pos() const { return p; }

    HD inline vec2 vect() const { return v; }

    HD inline vec2 dest() const { return d; }

    HD inline float speed() const { return s; }

    HD inline vec2 svect() const { return v*s; }

    HD agent() {}

    HD agent(float x, float y, float d_x, float d_y) {
        set_agent(x, y, d_x, d_y);
    }

    HD void set_agent(float x, float y, float d_x, float d_y) {
        p.set(x, y);
        d.set(d_x, d_y);
    }

    HD void set_vector(vec2 vect) { v = vect; }

    HD void set_speed(float speed){ s = speed; }

    HD void move() { p = p + v * s; }

    HD bool finished(float my_radius){
        return distance(p, d) < my_radius;
    }
};

#endif