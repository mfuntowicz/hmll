//
// Created by mfuntowicz on 12/16/25.
//

#include <limits.h>
#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"
#include "hmll/unix/iouring.h"

TEST_CASE("io_uring set slot busy", "[linux][io_uring][slot]")
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

TEST_CASE("io_uring set slot available", "[linux][io_uring][slot]")
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

SCENARIO("io_uring find slot", "[linux][io_uring][slot]")
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

TEST_CASE("io_uring compute throughput", "[linux][io_uring]")
{
    constexpr size_t nbytes = 7 * 1000 * 1000U;
    const size_t throughput = hmll_iouring_throughput(nbytes, 7U * 1e9);
    REQUIRE(throughput == 1000);
}

SCENARIO("io_uring congestion control algorithm", "[linux][io_uring]")
{
    hmll_iouring_cca cca = {0};
    hmll_iouring_cca_init(&cca);

    WHEN("initialization")
    {
        REQUIRE(cca.window == 1);
        REQUIRE(cca.throughput == 0);
    }

    WHEN("update the throughput")
    {
        constexpr timespec ts_start = {0, 0};
        constexpr timespec ts_end = {1, 0};
        const unsigned prev = hmll_iouring_cca_update(&cca, 7U * 1000 * 1000, ts_start, ts_end);
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
            const unsigned prev = hmll_iouring_cca_update(&cca, 7U * 1000 * 1000, ts_start, ts_end);
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
            const unsigned prev = hmll_iouring_cca_update(&cca, (i + 1) * 7U * 1000 * 1000, ts_start, ts_end);
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
        const unsigned prev = hmll_iouring_cca_update(&cca, 7U * 1000 * 1000, ts_start, ts_end);
        THEN("update cca window")
        {
            REQUIRE(prev > cca.window);
            REQUIRE(throughput > cca.throughput);
            REQUIRE(cca.window == 5);
        }
    }
}