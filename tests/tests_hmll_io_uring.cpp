//
// Created by mfuntowicz on 12/16/25.
//

#include <limits.h>
#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"
#include "../include/hmll/linux/backend/iouring.h"

TEST_CASE("io_uring set slot busy", "[linux][io_uring][slot]")
{
    struct hmll_iouring_iobusy iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 0);
    REQUIRE(iobusy.bits[0] == 1ULL << 0);
    REQUIRE(iobusy.bits[1] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 13);
    REQUIRE(iobusy.bits[0] == 1ULL << 13);
    REQUIRE(iobusy.bits[1] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 56);
    REQUIRE(iobusy.bits[0] == 1ULL << 56);
    REQUIRE(iobusy.bits[1] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 64);
    REQUIRE(iobusy.bits[0] == 0);
    REQUIRE(iobusy.bits[1] == 1ULL << 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 99);
    REQUIRE(iobusy.bits[0] == 0);
    REQUIRE(iobusy.bits[1] == 1ULL << 35);
}

TEST_CASE("io_uring set slot available", "[linux][io_uring][slot]")
{
    struct hmll_iouring_iobusy iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 0);
    REQUIRE(iobusy.bits[0] > 0);
    hmll_io_uring_slot_set_available(&iobusy, 0);
    REQUIRE(iobusy.bits[0] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 13);
    REQUIRE(iobusy.bits[0] == 1ULL << 13);
    hmll_io_uring_slot_set_available(&iobusy, 13);
    REQUIRE(iobusy.bits[0] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 56);
    REQUIRE(iobusy.bits[0] == 1ULL << 56);
    hmll_io_uring_slot_set_available(&iobusy, 56);
    REQUIRE(iobusy.bits[0] == 0);

    iobusy = {{0, 1}};
    hmll_io_uring_slot_set_busy(&iobusy, 38);
    REQUIRE(iobusy.bits[0] == (1ULL << 38));
    hmll_io_uring_slot_set_available(&iobusy, 38);
    REQUIRE(iobusy.bits[0] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 64);
    REQUIRE(iobusy.bits[1] == 1ULL << 0);
    hmll_io_uring_slot_set_available(&iobusy, 64);
    REQUIRE(iobusy.bits[1] == 0);

    iobusy = {};
    hmll_io_uring_slot_set_busy(&iobusy, 99);
    REQUIRE(iobusy.bits[1] == 1ULL << 35);
    hmll_io_uring_slot_set_available(&iobusy, 99);
    REQUIRE(iobusy.bits[1] == 0);
}

SCENARIO("io_uring find slot", "[linux][io_uring][slot]")
{
    GIVEN("A bitmap where all the slot are available")
    {
        struct hmll_iouring_iobusy iobusy = {};

        THEN("The first slot available is 0")
        {
            REQUIRE(iobusy.bits[0] == 0);
            REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 0);
        }

        WHEN("The first slot becomes unavailable")
        {
            hmll_io_uring_slot_set_busy(&iobusy, 0);
            THEN("The next slot available is 1")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 1);
            }
        }

        WHEN("Then another slot becomes unavailable")
        {
            hmll_io_uring_slot_set_busy(&iobusy, 0);
            hmll_io_uring_slot_set_busy(&iobusy, 1);
            THEN("The next slot available is 2")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 2);
            }
        }

        WHEN("A block is returned to the available pool")
        {
            hmll_io_uring_slot_set_busy(&iobusy, 0);
            hmll_io_uring_slot_set_busy(&iobusy, 1);
            hmll_io_uring_slot_set_available(&iobusy, 1);
            THEN("The next slot available is 1")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 1);
            }
        }

        WHEN("All lower word slots become unavailable")
        {
            iobusy.bits[0] = 0xFFFFFFFFFFFFFFFF;
            iobusy.bits[1] = 0;
            THEN("The next available slot is 64")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 64);
            }
        }

        WHEN("All the slots become unavailable")
        {
            iobusy.bits[0] = 0xFFFFFFFFFFFFFFFF;
            iobusy.bits[1] = 0xFFFFFFFFFFFFFFFF;
            THEN("No slot are available and the next available slot is -1")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == -1);
            }
        }

        WHEN("A block in the lower word becomes available")
        {
            iobusy.bits[0] = 0xFFFFFFFFFFFFFFFF;
            iobusy.bits[1] = 0;
            hmll_io_uring_slot_set_available(&iobusy, 63);
            THEN("The next slot available is 63")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 63);
            }
        }

        WHEN("A block in the upper word becomes available")
        {
            iobusy.bits[0] = 0xFFFFFFFFFFFFFFFF;
            iobusy.bits[1] = 0xFFFFFFFFFFFFFFFF;
            hmll_io_uring_slot_set_available(&iobusy, 100);
            THEN("The next slot available is 100")
            {
                REQUIRE(hmll_io_uring_slot_find_available(iobusy) == 100);
            }
        }
    }
}

TEST_CASE("io_uring compute throughput", "[linux][io_uring]")
{
    constexpr size_t nbytes = 7 * 1000 * 1000U;
    const size_t throughput = hmll_iouring_throughput(nbytes, 7U * 1e9);
    REQUIRE(throughput == 1000);
}

SCENARIO("io_uring congestion control algorithm", "[linux][io_uring]")
{
    hmll_iouring_cca cca = {0};
    hmll_io_uring_cca_init(&cca);

    WHEN("initialization")
    {
        REQUIRE(cca.window == 1);
        REQUIRE(cca.throughput == 0);
    }

    WHEN("update the throughput")
    {
        constexpr timespec ts_start = {0, 0};
        constexpr timespec ts_end = {1, 0};
        const unsigned prev = hmll_io_uring_cca_update(&cca, 7U * 1000 * 1000, ts_start, ts_end);
        THEN("update cca window")
        {
            REQUIRE(prev == 1);
            REQUIRE(cca.throughput > 0);
            REQUIRE(cca.window == 2);
        }
    }

    WHEN("throughput increase")
    {
        for (size_t i = 0; i < 2; i++) {
            constexpr timespec ts_end = {0, 5000};
            constexpr timespec ts_start = {0, 0};
            const unsigned throughput = cca.throughput;
            const unsigned prev = hmll_io_uring_cca_update(&cca, 7U * 1000 * 1000, ts_start, ts_end);
            THEN("update cca window")
            {
                REQUIRE(prev < cca.window);
                REQUIRE(throughput < cca.throughput);
                REQUIRE(cca.window == i + 2);
            }
        }
    }

    WHEN("throughput increase and then decrease (I/O limit)")
    {
        for (size_t i = 0; i < 5; i++) {
            constexpr timespec ts_end = {0, 5000};
            constexpr timespec ts_start = {0, 0};
            const unsigned throughput = cca.throughput;
            const unsigned prev = hmll_io_uring_cca_update(&cca, (i + 1) * 7U * 1000 * 1000, ts_start, ts_end);
            THEN("update cca window")
            {
                REQUIRE(prev < cca.window);
                REQUIRE(throughput < cca.throughput);
                REQUIRE(cca.window == i + 2);
            }
        }

        constexpr timespec ts_end = {5, 5000};
        constexpr timespec ts_start = {0, 0};
        const unsigned throughput = cca.throughput;
        const unsigned prev = hmll_io_uring_cca_update(&cca, 7U * 1000 * 1000, ts_start, ts_end);
        THEN("update cca window")
        {
            REQUIRE(prev > cca.window);
            REQUIRE(throughput > cca.throughput);
            REQUIRE(cca.window == 5);
        }
    }
}