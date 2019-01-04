#ifndef PROJEKT_AGENT_H
#define PROJEKT_AGENT_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include <cmath>

class vec2 {
public:
    float x_v, y_v;
    HD inline float &x() { return x_v; }

    HD inline float &y() { return y_v; }

    HD inline float x() const { return x_v; }

    HD inline float y() const { return y_v; }

    HD vec2() {}

    HD vec2(float x, float y) {
        x_v = x;
        y_v = y;
    }

    HD static vec2 rep(float v) { return vec2(v,v); }

    HD void set(float x, float y) {
        x_v = x;
        y_v = y;
    }

    HD void normalize() {
        if (x_v == 0 && y_v == 0) return;
        float len = (float) sqrt(x_v * x_v + y_v * y_v);
        x_v = x_v / len;
        y_v = y_v / len;
    }

    HD vec2 operator+(const vec2 &c) const {
        return vec2(x() + c.x(), y() + c.y());
    }

    HD vec2 operator-(const vec2 &c) const {
        return vec2(x() - c.x(), y() - c.y());
    }

    HD vec2 operator*(const vec2 &c) const {
        return vec2(x() * c.x(), y() * c.y());
    }
};


HD inline vec2 operator*(float scalar, const vec2& b) { return vec2::rep(scalar)*b; }
HD inline vec2 operator*(const vec2& a, float scalar) { return a*vec2::rep(scalar); }

class agent {
public:
    vec2 p;         //(x,y) position
    vec2 v;      //vector should be normed, so after every change of vector, we have to call normalize() method
    vec2 d;
    HD inline vec2 &pos() { return p; }

    HD inline vec2 &vect() { return v; }

    HD inline vec2 &dest() { return d; }

    HD inline vec2 pos() const { return p; }


    HD agent() {}

    HD agent(float x, float y, float d_x, float d_y) {
        p.set(x, y);
        d.set(d_x, d_y);
    }

    HD void set_agent(float x, float y, float d_x, float d_y) {
        p.set(x, y);
        d.set(d_x, d_y);
    }

    HD void set_vector(vec2 vect) {
        v = vect;
    }


/**
 * moves agent with given speed s along vector vect
 **/
    HD void move(float s) {
        p = p + v * s;
    }
};

#endif