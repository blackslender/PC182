#ifndef SUPPORT_H
#define SUPPORT_H

#define SETSIZE 32
#define DEFAULT_RECORDS_COUNT (1<<16)
#define SUP_RATE 0.1
#define CON_RATE 0.9

void setbit(int arr[], int index, bool value);
bool getbit(int arr[], int index);
int bitcount(int arr[]);

#endif