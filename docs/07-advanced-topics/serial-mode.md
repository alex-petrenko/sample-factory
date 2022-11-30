# Serial Mode

Debugging an asynchronous system can be hard.
In order to streamline debugging and development process, we provide a way to run all components of Sample Factory in a single process.
Enable serial mode by setting `--serial_mode` to `True`.

Serial regime is achieved via [signal-slot](../06-architecture/message-passing.md) mechanism.
Components interact by sending and receiving signals. Thus they actually don't care
whether they are running in the same process or in multiple processes. This allows us to put them
all on the same event loop in the main process.

## Applications

The main use case for serial mode is debugging and development.
If you're debugging your environment code, or any part of SF codebase, it is almost always easier to do it in serial
mode.

That said, for highly-vectorized GPU-accelerated environments it can be beneficial to run the whole system in serial mode,
which is exactly what we do by default with [IsaacGym](../09-environment-integrations/isaacgym.md).
One advantage of serial mode is that we minimize the number of CUDA contexts and thus VRAM usage.