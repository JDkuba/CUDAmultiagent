#ifndef CUDAMULTIAGENT_VO_H
#define CUDAMULTIAGENT_VO_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif


class vo {
public:
    float vect[2];
    HD inline float &vx() { return vect[0]; }

    HD inline float &vy() { return vect[1]; }

    HD inline float vx() const { return vect[0]; }

    HD inline float vy() const { return vect[1]; }

    HD vo() { };

    HD vo(float x, float y) {
        vect[0] = x;
        vect[1] = y;
    }
};


#endif
