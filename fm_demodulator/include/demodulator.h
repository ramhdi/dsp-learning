#ifndef DEMODULATOR_H
#define DEMODULATOR_H

#include "common.h"

void demodulator_init(void);
void fm_demodulate(int16_t *i_samples, int16_t *q_samples, int num_samples);

#endif