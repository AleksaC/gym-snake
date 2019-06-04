import gym
import gym_snake
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class Agent:
    def __init__(self, inp_shape):
        self.inp_shape = inp_shape
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=inp_shape, kernel_initializer='he_normal'))
        self.model.add(Dense(3, activation='softmax'))

        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, 3), name="action_onehot_placeholder")
        discount_reward_placeholder = K.placeholder(shape=(None,), name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam(lr=1e-4)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(
            inputs=[self.model.input,
                    action_onehot_placeholder,
                    discount_reward_placeholder],
            outputs=[],
            updates=updates
        )

    def action(self, state):
        action_prob = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(3), p=action_prob), action_prob

    def discounted_reward(self, R, discount_rate=0.99):
        disc_R = np.zeros_like(R, dtype=np.float32)
        running_add = 0

        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            disc_R[t] = running_add

        disc_R -= disc_R.mean()
        disc_R /= disc_R.std()

        return disc_R

    def fit(self, S, A, R):
        action_onehot = to_categorical(A, num_classes=3)
        self.train_fn([S, action_onehot, self.discounted_reward(R)])

    def run_episode(self, env, verbose=False, render=False):
        S = []
        A = []
        R = []

        done = False
        s = env.reset()

        reward_total = 0

        while not done:
            if render:
                env.render()
            a, a_prob = self.action(s.reshape((1,) + self.inp_shape))
            if verbose:
                print(f"ACTION PROB: {a_prob} Action: {env.ACTIONS[a]}")
            s2, r, done, _ = env.step(a)

            reward_total += r

            S.append(s.reshape(self.inp_shape))
            A.append(a)
            R.append(r)

            s = s2

            if done:
                self.fit(np.array(S), np.array(A), np.array(R))

        if verbose:
            print(f"TOTAL REWARD: {reward_total} SNAKE LENGTH: {env.snake_size}")
            print("=================================================================")

        return env.snake_size, reward_total


if __name__ == "__main__":
    env = gym.make('snake-v0')
    agent = Agent((484,))

    max_length = 0
    episode = 1

    try:
        while True:
            print(f"EPISODE {episode}:")
            print("==================")
            length, reward = agent.run_episode(env, verbose=True, render=True)
            if length > max_length: max_length = length
            episode += 1
    except KeyboardInterrupt:
        print(f"\nBest score: {max_length}")
        agent.model.save("model.h5")
