#ifndef PROJEKT_AGENT_H
#define PROJEKT_AGENT_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include <cmath>

class agent {
public:
    float cords[2];
    float vect[2];      //vector should be normed, so after every change of vector, we have to call normalize() method
    HD inline float &x() { return cords[0]; }

    HD inline float &y() { return cords[1]; }

    HD inline float &vx() { return vect[0]; }

    HD inline float &vy() { return vect[1]; }

    HD inline float x() const { return cords[0]; }

    HD inline float y() const { return cords[1]; }

    HD inline float vx() const { return vect[0]; }

    HD inline float vy() const { return vect[1]; }

    HD agent() {}

    HD agent(float x, float y, float vx, float vy) {
        cords[0] = x;
        cords[1] = y;
        vect[0] = vx;
        vect[1] = vy;
    }

    HD void set_agent(float x, float y, float vx, float vy) {
        cords[0] = x;
        cords[1] = y;
        vect[0] = vx;
        vect[1] = vy;
    }

    HD void set_vector(float vx, float vy) {
        vect[0] = vx;
        vect[1] = vy;
    }

    HD void normalize() {
        if (vx() == 0 && vy() == 0) return;
        float len = sqrt(vx() * vx() + vy() * vy());
        vx() = vx() / len;
        vy() = vy() / len;
    }

/**
 * moves agent with given speed s along vector vect
 **/
    HD void move(float s) {
        x() += vx() * s;
        y() += vy() * s;
    }
};

#endif