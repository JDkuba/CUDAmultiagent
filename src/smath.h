#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif 

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

    HD static vec2 rep(float v) { return {v, v}; }

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

struct vo {
    vec2 apex;
    vec2 left;
    vec2 right;
};


HD inline vec2 operator*(float scalar, const vec2 &b) { return vec2::rep(scalar) * b; }

HD inline vec2 operator*(const vec2 &a, float scalar) { return a * vec2::rep(scalar); }

HD inline vec2 operator/(float scalar, const vec2 &b) { return vec2::rep(scalar) / b; }

HD inline vec2 operator/(const vec2 &a, float scalar) { return a / vec2::rep(scalar); }

HD inline vec2 operator+(float scalar, const vec2 &b) { return vec2::rep(scalar) + b; }

HD inline vec2 operator+(const vec2 &a, float scalar) { return a + vec2::rep(scalar); }

HD inline vec2 operator-(float scalar, const vec2 &b) { return vec2::rep(scalar) - b; }

HD inline vec2 operator-(const vec2 &a, float scalar) { return a - vec2::rep(scalar); }

HD inline float distance(const vec2& a, const vec2& b) {
    return sqrt((a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()));
}

HD inline float det(const vec2& a, const vec2& b) {
    return a.x() * b.y() - a.y() * b.x();
}


#endif