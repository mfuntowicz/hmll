#ifndef HMLL_H
#define HMLL_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HMLL_EXTERN
#ifndef HMLL_STATIC
#ifdef _WIN32
#define HMLL_EXTERN __declspec(dllimport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define HMLL_EXTERN __attribute__((visibility("default")))
#else
#define HMLL_EXTERN
#endif
#else
#define HMLL_EXTERN
#endif
#endif

#include "status.h"


#ifdef __cplusplus
}
#endif
#endif // HMLL_H
