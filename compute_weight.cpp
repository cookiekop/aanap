//
// Created by Michael Zhang on 2018/8/2.
//

#include "compute_weight.h"

vector<float> compute_weight(const Point2f &p, const Point2f &K1, const Point2f &K2) {
    vector<float> r(2);
    if (p.x < K1.x) {
        r[0] = 0;
        r[1] = 1;
        return r;
    }

    if (p.x > K2.x) {
        r[0] = 1;
        r[1] = 0;
        return r;
    }

    float a = (p.x - K1.x) * (K2.x - K1.x);
    float b = (p.y - K1.y) * (K2.y - K1.y);
    float c = powf(K2.x - K1.x,2) + powf(K2.y - K1.y,2);

    r[0] = abs(a + b)/c;
    r[1] = 1 - r[0];
    return r;
}