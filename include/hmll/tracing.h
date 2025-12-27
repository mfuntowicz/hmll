#ifndef HMLL_TRACING_H
#define HMLL_TRACING_H

#if defined(__HMLL_PROFILE_ENABLED__)
#include <tracy/TracyC.h>
#define HMLL_ZONE_START(name) \
    TracyCZone(const name, __HMLL_PROFILE_ENABLED__)
#define HMLL_ZONE_END(name) TracyCZoneEnd(name)
#define HMLL_ZONE_END_COLOR(name, color) \
    TracyCZoneColor(name, color)         \
    TracyCZoneEnd(name)                  \

#define HMLL_ZONE_END_SUCCESS(name) HMLL_ZONE_END_COLOR(name, 0x00ff00)
#define HMLL_ZONE_END_WARNING(name) HMLL_ZONE_END_COLOR(name, 0xffa500)
#define HMLL_ZONE_END_ERROR(name) HMLL_ZONE_END_COLOR(name, 0xff0000)

#else
#define HMLL_ZONE(name)
#endif
#endif // HMLL_TRACING_H
