#ifndef PROJEKT_AGENT_H
#define PROJEKT_AGENT_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include <math.h>

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

    HD static vec2 rep(float v) { return {v,v}; }

    HD void set(float x, float y) {
        x_v = x;
        y_v = y;
    }

    HD float length() {
        return sqrt(x_v * x_v + y_v * y_v);
    }

    HD vec2 normalized() {
        float len = length();
        if (len == 0) return rep(0);
        return {x_v / len, y_v = y_v / len};
    }

    HD vec2 rotate(float theta) {
        return {x_v * cos(theta) - y_v * sin(theta), x_v * sin(theta) + y_v * cos(theta)};
    }

    HD vec2 operator+(const vec2 &c) const {
        return {x() + c.x(), y() + c.y()};
    }

    HD vec2 operator-(const vec2 &c) const {
        return {x() - c.x(), y() - c.y()};
    }

    HD vec2 operator*(const vec2 &c) const {
        return {x() * c.x(), y() * c.y()};
    }

    HD vec2 operator/(const vec2 &c) const {
        return {x() / c.x(), y() / c.y()};
    }
};


HD inline vec2 operator*(float scalar, const vec2 &b) { return vec2::rep(scalar) * b; }

HD inline vec2 operator*(const vec2 &a, float scalar) { return a * vec2::rep(scalar); }

HD inline vec2 operator/(float scalar, const vec2 &b) { return vec2::rep(scalar) / b; }

HD inline vec2 operator/(const vec2 &a, float scalar) { return a / vec2::rep(scalar); }

HD inline vec2 operator+(float scalar, const vec2 &b) { return vec2::rep(scalar) + b; }

HD inline vec2 operator+(const vec2 &a, float scalar) { return a + vec2::rep(scalar); }

HD inline vec2 operator-(float scalar, const vec2 &b) { return vec2::rep(scalar) - b; }

HD inline vec2 operator-(const vec2 &a, float scalar) { return a - vec2::rep(scalar); }

class agent {
public:
    vec2 p;         //(x,y) position
    vec2 v;      //vector should be normed, so after every change of vector, we have to call normalize() method
    vec2 d;
    HD inline vec2 &pos() { return p; }

    HD inline vec2 &vect() { return v; }

    HD inline vec2 &dest() { return d; }

    HD inline vec2 pos() const { return p; }

    HD inline vec2 vect() const { return v; }

    HD inline vec2 dest() const { return d; }

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
 * moves agent with given speed s along vector v
 **/
    HD void move(float s) {
        p = p + v * s;
    }
};

struct vo {
    vec2 apex;
    vec2 left;
    vec2 right;
};

#endif