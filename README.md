[![PyPI - Version](https://img.shields.io/pypi/v/asimov_agents.svg)](https://pypi.org/project/asimov_agents)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/asimov_agents.svg)](https://pypi.org/project/asimov_agents)

## General Thoughts

### Agents

Agents are comprised of planners, executors, observers, and discriminators.

For a given task the general flow is a planner will receive a task and generate an objective and a directed graph of either parallelizeable or sequential steps that executors will attempt to complete to achieve this.

Each output is passed to a discriminator who qualitatively and or functionally (depending on the task) asseses it. The discriminator will pass or fail an output which depdening on the configuration can either be retried or hard failed and kicked back up to the planner to decide next steps. 

Observers are present during each stage and function to provide feedback to executors and discriminators. They do not make decisions and cannot directly affect execution but their output is supplied as context for a discriminator or executor to alter their outputs.

If an output succeeds it will be passed downstream to the next executor and the process repeats until completion.

Planners receive updates on step completion and have a discriminator and observer to influence viability of the plan being executed. 


