import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v0")
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
EPSILON = 1e-8
DISCOUNT = 0.99
MAX_EPISODES = 10000

class ActorCriticNetwork:

    _critic_learning_rate = 0.005
    _actor_learning_rate = 0.005

    def __init__(self, sess, input_size, output_size, hidden_dims):
        self._sess = sess
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_dims = hidden_dims

        self._BulidBaseNetwork()
        self._BulidCriticNetwork()
        self._BulidActorNetwork()
        self._sess.run(tf.global_variables_initializer())

    def _BulidBaseNetwork(self):
        self._obs = tf.placeholder(tf.float32, shape=[None, self._input_size], name="obs_input")
        self._action = tf.placeholder(tf.float32, shape=[None, self._output_size], name="action_input")
        self._reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward_input")
        self._advantage_value = tf.placeholder(tf.float32, shape=[None, 1], name="advantage_input")

        net = self._obs
        for i, h_dim in enumerate(self._hidden_dims):
            net = tf.contrib.layers.fully_connected(net, h_dim, activation_fn=None, scope=f"fc-{i}")
            net = tf.nn.relu(net)
        
        self._net = net

    # [Critic]
    def _BulidCriticNetwork(self):
        self._value = tf.contrib.layers.fully_connected(self._net, 1, activation_fn=None, scope="critic_output")        

        self._critic_loss = tf.reduce_mean(tf.squared_difference(self._value, self._reward))
        self._critic_train = tf.train.AdamOptimizer(self._critic_learning_rate).minimize(self._critic_loss)

    def CriticPredict(self, obs):
        p = self._sess.run([self._value], feed_dict={self._obs: obs})
        return p

    def CriticTrain(self, obs, rew):
        _, loss = self._sess.run([self._critic_train, self._critic_loss], feed_dict={self._obs : obs, self._reward : rew})
        print("Critic Loss :", loss);

    # [Actor]
    def _BulidActorNetwork(self):
        self._logit = tf.contrib.layers.fully_connected(self._net, self._output_size, activation_fn=tf.nn.softmax, scope="actor_output")
        log_p = -self._action * tf.log(tf.clip_by_value(self._logit, EPSILON, 1.))
        log_lik = log_p * self._advantage_value
        self._actor_loss = tf.reduce_mean(tf.reduce_sum(log_lik, axis=1))
        self._actor_train = tf.train.AdamOptimizer(self._actor_learning_rate).minimize(self._actor_loss)
    
    def GetAction(self, obs):
        obs = np.reshape(obs, [-1, self._input_size])
        action = self._sess.run([self._logit], feed_dict={self._obs : obs})[0]
        action = np.random.choice(np.arange(self._output_size), p=action[0])
        return action

    def ActorTrain(self, obs, act, a_rew):
        _, loss = self._sess.run([self._actor_train, self._actor_loss], feed_dict={self._obs : obs, self._action : act, self._advantage_value : a_rew})
        print("Actor Loss :", loss);

def main():

    with tf.Session() as sess:
        network = ActorCriticNetwork(sess, INPUT_SIZE, OUTPUT_SIZE, [32, 32])
        
        for episode in range(MAX_EPISODES):
            obs = env.reset()
            done = False
            step = 0

            obs_list = []
            act_list = []
            rew_list = []

            while True:
                if episode > 500:
                    env.render()

                action = network.GetAction(obs)
                obs_list.append(obs)
                act_list.append(OneHot(action))

                obs, rew, done, _ = env.step(action)
                rew_list.append(rew)
                step += 1

                if done:   
                    obs_list = np.vstack(obs_list)
                    act_list = np.vstack(act_list)
                    rew_list = DiscountRewards(rew_list)
                    predict = network.CriticPredict(obs_list)
                    a_rew = NormalizeRewards(rew_list, predict)

                    network.CriticTrain(obs_list, rew_list)
                    network.ActorTrain(obs_list, act_list, a_rew)
                    print("Episode : ", episode, " Step : ", step)
                    print("============================")
                    break;

def OneHot(value):
    zero = np.zeros(OUTPUT_SIZE, dtype = np.int)
    zero[value] = 1
    return  zero

def DiscountRewards(reward_memory):
    v_memory = np.vstack(reward_memory)
    discounted = np.zeros_like(v_memory, dtype=np.float32)
    add_value = 0
    length = len(reward_memory)

    for i in reversed(range(length)):
        if v_memory[i] < 0:
            add_value = 0
        add_value = v_memory[i] + (DISCOUNT * add_value)
        discounted[i] = add_value

    return discounted

def NormalizeRewards(rewards, v_rewards):
    a_reward = np.vstack(rewards) - np.vstack(v_rewards)
    a_reward -= np.mean(a_reward)
    a_reward /= (np.std(a_reward) + EPSILON)
    return a_reward

if __name__ == "__main__":
    main()
