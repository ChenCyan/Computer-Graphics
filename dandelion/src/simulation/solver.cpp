#include "solver.h"

#include <Eigen/Core>

using Eigen::Vector3f;

// External Force does not changed.

// Function to calculate the derivative of KineticState
KineticState derivative(const KineticState& state)
{
    return KineticState(state.velocity, state.acceleration, Eigen::Vector3f(0, 0, 0));
}

// Function to perform a single Forward Euler step
KineticState forward_euler_step([[maybe_unused]] const KineticState& previous,
                                const KineticState& current)
{
    KineticState next_state;
    next_state.acceleration = current.acceleration;
    next_state.position = current.position + time_step * current.velocity;
    next_state.velocity = current.velocity + time_step * current.acceleration;
    return next_state;
}

// Function to perform a single Runge-Kutta step
KineticState runge_kutta_step([[maybe_unused]] const KineticState& previous,
                              const KineticState& current)
{
    KineticState next_state = current;
    //Because the force keeps the same
    //The acceleration will not change
    //Thus the k_v will not change
    Vector3f k1_v = current.acceleration;
    Vector3f k2_v = current.acceleration;
    Vector3f k3_v = current.acceleration;
    Vector3f k4_v = current.acceleration;
    next_state.velocity = current.velocity + (time_step / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v);
    

    Vector3f k1_x = current.velocity;
    Vector3f k2_x = current.velocity + (time_step / 2) * current.acceleration;
    Vector3f k3_x = current.velocity + (time_step / 2) * current.acceleration;
    Vector3f k4_x = current.velocity + time_step * current.acceleration;
    next_state.position = current.position + (time_step / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x);
    return next_state;
}

// Function to perform a single Backward Euler step
KineticState backward_euler_step([[maybe_unused]] const KineticState& previous,
                                 const KineticState& current)
{
     KineticState next_state;
    next_state.acceleration = current.acceleration;
    next_state.velocity = current.velocity + time_step * next_state.acceleration;
    next_state.position = current.position + time_step * next_state.velocity;
    return next_state;
}

// Function to perform a single Symplectic Euler step
KineticState symplectic_euler_step(const KineticState& previous, const KineticState& current)
{
    (void)previous;
    KineticState next_state;
    next_state.acceleration = current.acceleration;
    next_state.velocity = current.velocity + time_step * current.acceleration;
    next_state.position = current.position + time_step * next_state.velocity;
    
    return next_state;
}
