# Troubleshooting / FAQ

Frequently asked questions and answers.

## FPS and throughput drop to 0 shortly after startup

This is caused by one of the key system components crashing during startup.
Examples include: all rollout workers crash due to a bug in the enviroment creation code or `reset()`, learner crashes 
due to CUDA or OOM issues, or any other similar reason.

To diagnose the problem you should check the full log file to see exactly what happened (i.e. to find the exception trace).
Stdout/stderr output can be viewed in the terminal, or in the `sf_log.txt` file in the experiment folder.

Try to look for the original issue that caused the system to halt; periodic log messages may
cause it not to appear on the screen at the time the problem is noticed.
