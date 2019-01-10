#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif 

#define EPS 0.000001

HD inline bool isFloatZero(float x){
    return x-EPS <= 0 and x+EPS >= 0;
}

class vec2 {
public:
    float x_v, y_v;
    HD inline float &x() { return x_v; }

    HD inline float &y() { return y_v; }

    HD inline float x() const { return x_v; }

    HD inline float y() const { return y_v; }

    HD inline float isZero() const {
        return isFloatZero(x_v) and isFloatZero(y_v);
    }

    HD inline bool invalid() const {
        return isnan(x_v) or isnan(y_v);
    }

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

struct ray {
    vec2 pos;
    vec2 dir; // should be normalized

    HD ray(float x, float y, float d_x, float d_y){
        pos = vec2(x, y);
        dir = vec2(d_x, d_y).normalized();
    }

    HD inline float A() const {
        vec2 p = pos + dir;
        return pos.y() - p.y();
    }
    HD inline float B() const {
        vec2 p = pos + dir;
        return p.x() - pos.x();
    }
    HD inline float C() const{
        vec2 p = pos + dir;
        return pos.x()*p.y() - pos.y()*p.x();
    } 
};

HD inline bool are_parallel(const ray &r1, const ray &r2){
    return (r1.dir - r2.dir).isZero() or (r1.dir + r2.dir).isZero();
}

HD inline vec2 intersect_rays(const ray &r1, const ray &r2){
    if(are_parallel(r1, r2))
        return vec2(nanf("1"), nanf("1"));

    float A1 = r1.A(), B1 = r1.B(), C1 = r1.C();
    float A2 = r2.A(), B2 = r2.B(), C2 = r2.C();
    float W = A1*B2 - A2*B1;
    float Wx = B1*C2 - B2*C1;
    float Wy = C1*A2 - C2*A1;
    vec2 intersec(Wx/W, Wy/W);

    vec2 dir1 = (intersec - r1.pos).normalized();
    vec2 dir2 = (intersec - r2.pos).normalized();

    if((r1.dir + dir1).isZero() or (r2.dir + dir2).isZero())
        return vec2(nanf("1"), nanf("1"));

    return intersec;
}


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