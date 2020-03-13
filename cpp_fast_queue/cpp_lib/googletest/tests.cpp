#include <array>
#include <vector>

#include "gtest/gtest.h"

#include "fast_queue.hpp"


TEST(fast_queue, queue_test) {
    const auto q_size = queue_object_size();
    std::vector<uint8_t> q_buffer(q_size);  // allocate memory for the queue object (normally would be Python shared memory)
    void *q = q_buffer.data();

    constexpr float tm = 1.0;
    constexpr size_t max_size_bytes = 100;
    create_queue(q, max_size_bytes);

    std::array<uint8_t, max_size_bytes> buffer{};  // memory for the circular buffer

    // adding messages to the queue
    std::array<uint8_t, 5> msg0{0, 1, 2, 3, 42};
    auto status = queue_put(q, buffer.data(), msg0.data(), sizeof(msg0), false, tm);
    EXPECT_EQ(status, Q_SUCCESS);

    std::array<uint8_t, 80> msg1{};
    status = queue_put(q, buffer.data(), msg1.data(), sizeof(msg1), true, tm);
    EXPECT_EQ(status, Q_FULL);

    std::array<uint8_t, 79> msg2{};
    msg2[1] = 0xff;
    msg2[78] = 0xee;
    status = queue_put(q, buffer.data(), msg2.data(), sizeof(msg2), true, tm);
    EXPECT_EQ(status, Q_SUCCESS);

    uint8_t msg3;
    status = queue_put(q, buffer.data(), &msg3, sizeof(msg3), true, tm);
    EXPECT_EQ(status, Q_FULL);

    // reading messages from the queue
    size_t msgs_read, bytes_read, msgs_size;

    // try to read one message, while providing insufficient buffer size
    std::array<uint8_t, 10> msg_buffer10{};
    status = queue_get(q, buffer.data(), msg_buffer10.data(), sizeof(msg_buffer10), 1, 100, &msgs_read, &bytes_read, &msgs_size, true, tm);
    EXPECT_EQ(status, Q_MSG_BUFFER_TOO_SMALL);
    EXPECT_EQ(msgs_read, 0);
    EXPECT_EQ(bytes_read, 0);
    EXPECT_EQ(msgs_size, sizeof(msgs_size) + sizeof(msg0));

    // allocate a bigger buffer that fits the first message + message size
    std::array<uint8_t, 13> msg_buffer13{};
    status = queue_get(q, buffer.data(), msg_buffer13.data(), sizeof(msg_buffer13), 1, 100, &msgs_read, &bytes_read, &msgs_size, true, tm);
    EXPECT_EQ(status, Q_SUCCESS);
    EXPECT_EQ(msgs_read, 1);
    EXPECT_EQ(bytes_read, sizeof(msg_buffer13));
    EXPECT_EQ(msgs_size, sizeof(msgs_size) + sizeof(msg0));  // we read as many messages as we wanted, so this is just equal to bytes_read
    EXPECT_EQ(*(size_t *)msg_buffer13.data(), sizeof(msg0));  // first 8 bytes of the message should contain the size of the message
    EXPECT_EQ(memcmp(msg_buffer13.data() + sizeof(size_t), msg0.data(), sizeof(msg0)), 0);  // the message we read is identical to the message we put in a queue

    // attempt to read the next (big) message using small buffer
    status = queue_get(q, buffer.data(), msg_buffer13.data(), sizeof(msg_buffer13), 100, 100, &msgs_read, &bytes_read, &msgs_size, true, tm);
    EXPECT_EQ(status, Q_MSG_BUFFER_TOO_SMALL);
    EXPECT_EQ(msgs_read, 0);
    EXPECT_EQ(bytes_read, 0);
    EXPECT_EQ(msgs_size, sizeof(msgs_size) + sizeof(msg2));  // this is how many bytes we need to actually read the next message

    // allocate a bigger buffer and read the next message
    std::array<uint8_t, max_size_bytes> msg_buffer100{};
    status = queue_get(q, buffer.data(), msg_buffer100.data(), sizeof(msg_buffer100), 100, 100, &msgs_read, &bytes_read, &msgs_size, true, tm);
    EXPECT_EQ(status, Q_SUCCESS);
    EXPECT_EQ(msgs_read, 1);
    EXPECT_EQ(bytes_read, sizeof(msgs_size) + sizeof(msg2));
    EXPECT_EQ(msgs_size, bytes_read);
    EXPECT_EQ(memcmp(msg_buffer100.data() + sizeof(size_t), msg2.data(), sizeof(msg2)), 0);  // the message we read is identical to the message we put in a queue

    // at this point the queue is empty, any attempt to read messages will be unsuccessful
    status = queue_get(q, buffer.data(), msg_buffer100.data(), sizeof(msg_buffer100), 100, 100, &msgs_read, &bytes_read, &msgs_size, true, tm);
    EXPECT_EQ(status, Q_EMPTY);
    EXPECT_EQ(msgs_read, 0);
    EXPECT_EQ(bytes_read, 0);
    EXPECT_EQ(msgs_size, 0);
    status = queue_get(q, buffer.data(), msg_buffer100.data(), sizeof(msg_buffer100), 1, 1, &msgs_read, &bytes_read, &msgs_size, true, tm);
    EXPECT_EQ(status, Q_EMPTY);
}
