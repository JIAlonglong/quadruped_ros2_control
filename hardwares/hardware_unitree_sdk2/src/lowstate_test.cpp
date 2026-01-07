#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#define TOPIC_LOWSTATE "rt/lowstate"

using namespace unitree::robot;

static bool running = true;

void handleSigint(int)
{
    running = false;
}

int main(int argc, char** argv)
{
    int domain = 1;
    std::string interface = "eth2";
    if (argc >= 2)
    {
        interface = argv[1];
    }
    if (argc >= 3)
    {
        domain = std::stoi(argv[2]);
    }

    ChannelFactory::Instance()->Init(domain, interface);

    unitree_go::msg::dds_::LowState_ low_state;
    auto low_sub = std::make_shared<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(TOPIC_LOWSTATE);
    low_sub->InitChannel(
        [&](const void* msg)
        {
            low_state = *static_cast<const unitree_go::msg::dds_::LowState_*>(msg);
        },
        1);

    std::signal(SIGINT, handleSigint);

    std::cout << "Listening LowState on interface " << interface << " domain " << domain << std::endl;

    while (running)
    {
        std::cout << "q: ";
        for (int i = 0; i < 12; ++i)
        {
            std::cout << low_state.motor_state()[i].q() << " ";
        }
        std::cout << std::endl;

        std::cout << "foot_force: ";
        for (int i = 0; i < 4; ++i)
        {
            std::cout << low_state.foot_force()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "------------------------" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}
