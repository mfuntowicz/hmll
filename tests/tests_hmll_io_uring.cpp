//
// Created by mfuntowicz on 12/16/25.
//

#include <limits.h>
#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"
#include "hmll/unix/iouring.h"

TEST_CASE("io_uring set slot busy", "[io_uring][slot]")
{
    struct hmll_iouring_iobusy iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 0);
    REQUIRE(iobusy.lsb == 1LL << 0);
    REQUIRE(iobusy.msb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 13);
    REQUIRE(iobusy.lsb == 1LL << 13);
    REQUIRE(iobusy.msb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 56);
    REQUIRE(iobusy.lsb == 1LL << 56);
    REQUIRE(iobusy.msb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 64);
    REQUIRE(iobusy.lsb == 0);
    REQUIRE(iobusy.msb == 1LL << 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 99);
    REQUIRE(iobusy.lsb == 0);
    REQUIRE(iobusy.msb == 1LL << 35);
}

TEST_CASE("io_uring set slot available", "[io_uring][slot]")
{
    struct hmll_iouring_iobusy iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 0);
    REQUIRE(iobusy.lsb > 0);
    hmll_iouring_slot_set_available(&iobusy, 0);
    REQUIRE(iobusy.lsb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 13);
    REQUIRE(iobusy.lsb == 1LL << 13);
    hmll_iouring_slot_set_available(&iobusy, 13);
    REQUIRE(iobusy.lsb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 56);
    REQUIRE(iobusy.lsb == 1LL << 56);
    hmll_iouring_slot_set_available(&iobusy, 56);
    REQUIRE(iobusy.lsb == 0);

    iobusy = {1, 0};
    hmll_iouring_slot_set_busy(&iobusy, 38);
    REQUIRE(iobusy.lsb == (1LL << 38));
    hmll_iouring_slot_set_available(&iobusy, 38);
    REQUIRE(iobusy.lsb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 64);
    REQUIRE(iobusy.msb == 1LL << 0);
    hmll_iouring_slot_set_available(&iobusy, 64);
    REQUIRE(iobusy.msb == 0);

    iobusy = {0, 0};
    hmll_iouring_slot_set_busy(&iobusy, 99);
    REQUIRE(iobusy.msb == 1LL << 35);
    hmll_iouring_slot_set_available(&iobusy, 99);
    REQUIRE(iobusy.msb == 0);
}

SCENARIO("io_uring find slot", "[io_uring][slot]")
{
    GIVEN("A bitmap where all the slot are available")
    {
        struct hmll_iouring_iobusy iobusy = {0, 0};

        THEN("The first slot available is 0")
        {
            REQUIRE(iobusy.lsb == 0);
            REQUIRE(hmll_iouring_slot_find_available(iobusy) == 0);
        }

        WHEN("The first slot becomes unavailable")
        {
            hmll_iouring_slot_set_busy(&iobusy, 0);
            THEN("The next slot available is 1")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == 1);
            }
        }

        WHEN("Then another slot becomes unavailable")
        {
            hmll_iouring_slot_set_busy(&iobusy, 0);
            hmll_iouring_slot_set_busy(&iobusy, 1);
            THEN("The next slot available is 2")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == 2);
            }
        }

        WHEN("A block is returned to the available pool")
        {
            hmll_iouring_slot_set_busy(&iobusy, 0);
            hmll_iouring_slot_set_busy(&iobusy, 1);
            hmll_iouring_slot_set_available(&iobusy, 1);
            THEN("The next slot available is 1")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == 1);
            }
        }

        WHEN("All LSB slots become unavailable")
        {
            iobusy.lsb = 0xFFFFFFFFFFFFFFFF;
            iobusy.msb = 0;
            THEN("The next available slot is 64 (first MSB slot)")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == 64);
            }
        }

        WHEN("All the slots become unavailable")
        {
            iobusy.lsb = 0xFFFFFFFFFFFFFFFF;
            iobusy.msb = 0xFFFFFFFFFFFFFFFF;
            THEN("No slot are available and the next available slot is -1")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == -1);
            }
        }

        WHEN("A block in LSB becomes available")
        {
            iobusy.lsb = 0xFFFFFFFFFFFFFFFF;
            iobusy.msb = 0;
            hmll_iouring_slot_set_available(&iobusy, 63);
            THEN("The next slot available is 63")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == 63);
            }
        }

        WHEN("A block in MSB becomes available")
        {
            iobusy.lsb = 0xFFFFFFFFFFFFFFFF;
            iobusy.msb = 0xFFFFFFFFFFFFFFFF;
            hmll_iouring_slot_set_available(&iobusy, 100);
            THEN("The next slot available is 100")
            {
                REQUIRE(hmll_iouring_slot_find_available(iobusy) == 100);
            }
        }
    }
}