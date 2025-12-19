//
// Created by mfuntowicz on 12/16/25.
//

#include <limits.h>
#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"
#include "hmll/unix/iouring.h"

TEST_CASE("io_uring set slot busy", "[io_uring][slot]")
{
    long long mask = 0;
    hmll_io_uring_slot_set_busy(&mask, 0);
    REQUIRE(mask == 1LL << 0);

    mask = 0;
    hmll_io_uring_slot_set_busy(&mask, 13);
    REQUIRE(mask == 1LL << 13);

    mask = 0;
    hmll_io_uring_slot_set_busy(&mask, 56);
    REQUIRE(mask == 1LL << 56);
}

TEST_CASE("io_uring set slot available", "[io_uring][slot]")
{
    long long mask = 0;
    hmll_io_uring_slot_set_busy(&mask, 0);
    REQUIRE(mask > 0);
    hmll_io_uring_slot_set_available(&mask, 0);
    REQUIRE(mask == 0);

    mask = 0;
    hmll_io_uring_slot_set_busy(&mask, 13);
    REQUIRE(mask == 1LL << 13);
    hmll_io_uring_slot_set_available(&mask, 13);
    REQUIRE(mask == 0);

    mask = 0;
    hmll_io_uring_slot_set_busy(&mask, 56);
    REQUIRE(mask == 1LL << 56);
    hmll_io_uring_slot_set_available(&mask, 56);
    REQUIRE(mask == 0);

    mask = 1;
    hmll_io_uring_slot_set_busy(&mask, 38);
    REQUIRE(mask == (1LL << 38) + 1);
    hmll_io_uring_slot_set_available(&mask, 38);
    REQUIRE(mask == 1);
}

SCENARIO("io_uring find slot", "[io_uring][slot]")
{
    GIVEN("A bitmap where all the slot are available")
    {
        long long mask = 0;

        THEN("The first slot available is 0")
        {
            REQUIRE(mask == 0);
            REQUIRE(hmll_io_uring_slot_find_available(mask) == 0);
        }

        WHEN("The first slot becomes unavailable")
        {
            hmll_io_uring_slot_set_busy(&mask, 0);
            THEN("The next slot available is 1")
            {
                REQUIRE(hmll_io_uring_slot_find_available(mask) == 1);
            }
        }

        WHEN("Then another slot becomes unavailable")
        {
            hmll_io_uring_slot_set_busy(&mask, 0);
            hmll_io_uring_slot_set_busy(&mask, 1);
            THEN("The next slot available is 2")
            {
                REQUIRE(hmll_io_uring_slot_find_available(mask) == 2);
            }
        }

        WHEN("A block is returned to the available pool")
        {
            hmll_io_uring_slot_set_busy(&mask, 0);
            hmll_io_uring_slot_set_busy(&mask, 1);
            hmll_io_uring_slot_set_available(&mask, 1);
            THEN("The next slot available is 1")
            {
                REQUIRE(hmll_io_uring_slot_find_available(mask) == 1);
            }
        }

        WHEN("All the slot becomes unavailable")
        {
            mask = 0xFFFFFFFFFFFFFFFF;
            THEN("No slot are available and the next available slot is -1")
            {
                REQUIRE(hmll_io_uring_slot_find_available(mask) == -1);
            }
        }

        WHEN("A block becomes available")
        {
            mask = 0xFFFFFFFFFFFFFFFF;
            hmll_io_uring_slot_set_available(&mask, sizeof(mask) * 8 - 1);
            THEN("The next slot available is 63")
            {
                REQUIRE(hmll_io_uring_slot_find_available(mask) == 63);
            }
        }
    }
}