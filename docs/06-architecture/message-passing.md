# Event Loops, Signals, and Slots

Sample Factory uses a custom mechanism for communication between components inspired by Qt's signals and slots.
Unlike in Qt, signals and slots can be used not only across threads, but also across processes.
The implementation of this mechanism is available as a separate repository [here](https://github.com/alex-petrenko/signal-slot).

The main idea can be summaries as follows:

* Application is a collection of `EventLoop`s. Each `EventLoop` is an infinite loop that occupies a thread or a process.
* Logic of the system is implemented in `EventLoopObject` components that live on `EventLoop`s.
Each `EventLoop` can support multiple `EventLoopObject`s.
* Components (i.e. `EventLoopObject`s) can emit signals. A signal "message" contains a name of the signal
and the payload (arbitrary data).
* Components can also connect to signals emitted by other components by specifying a `slot` function to be called when the signal is received
by the EventLoop.

The majority of communication between components is done via signals and slots. Some examples:

* Rollout workers emit `"p{policy_id}_trajectories"` signal when a new trajectory is available, and Batcher's
`on_new_trajectories()` slot is connected to this signal.
* Inference workers emit `"advance{rollout_worker_idx}"` signal when actions are ready for the next rollout step,
and RolloutWorker's `advance_rollouts()` slot is connected to this signal.

## Implementation details

* There's not argument validation for signals and slots. If you connect a slot to a signal with a different signature,
it will fail at runtime. This can also be used to your advantage by allowing to propagate arbitrary data as
payload with appropriate runtime checks.
* Signals can be connected to slots only before the processes are spawned, i.e. only during system initialization.
This is mostly done by the `Runner` in `connect_components()`.
* Signal-slot mechanism in the current implementation can represent one-to-one, one-to-many, and many-to-many connections,
but not many-to-one. This is because by design each EventLoop has its own corresponding event queue.
For many-to-one connections we use queues explicitly, e.g. RolloutWorkers explicitly push requests for new actions
into queues corresponding to a policy that controls the agent, and this queue can be processed by multiple InferenceWorkers:
`inference_queues[policy_id].put(policy_request)`

Please see https://github.com/alex-petrenko/signal-slot for more information.

## Multiprocessing queues

At the core of the signal-slot mechanism are the queues that are used to pass messages between processes.
Python provides a default implementation `multiprocessing.Queue`, which turns out to be rather slow.

Sample Factory uses a custom queue implementation written in C++ using POSIX API that is significantly faster:
https://github.com/alex-petrenko/faster-fifo