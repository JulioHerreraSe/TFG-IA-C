#ifndef EJEMPLOC_DATAREADER_H
#define EJEMPLOC_DATAREADER_H

#include "../includes/config.h"

void read_csv(const char* filePath, double*** features, double** target, int* numSamples);

#endif //EJEMPLOC_DATAREADER_H
