# Snake [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AleksaC/dldidact/blob/master/LICENSE)

Snake game [gym](https://github.com/openai/gym) environment with RL agents for solving it.

## About
This was done over a year before it was published here as a public
repository and it was of my first endeavors in reinforcement learning. 
The task was one of the warmups in 
[OpenAI's Requests for Research 2.0](https://openai.com/blog/requests-for-research-2/).

## Trying it out
### Install the environment
To install the environment open the terminal and enter the following
commands:
1. `git clone https://github.com/AleksaC/snake.git`
2. `cd gym-snake`
3. `python -m pip install .`

Note:
To be able to use the environment you need to `import gym_snake` before 
making the environment 
### Agents
Unfortunately I didn't have time to try out various types of agents when I
was working on this so there's only implementation of vanilla 
policy gradient algorithm. It was built with tf.keras so you need to have
tensorflow installed to be able to run it.

## Contact
If you want to reach out to me to make an inquiry about this project 
or for something else you can contact me through my
[personal website](https://www.aleksac.me) or via social media linked there.
If you want to stay up to date with my latest projects you should follow me 
on twitter:
<a target="_blank" href="http://twitter.com/aleksa_c_"><img alt='Twitter followers' src="https://img.shields.io/twitter/follow/aleksa_c_.svg?style=social"></a>
