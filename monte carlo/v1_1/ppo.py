import tensorflow as tf
import numpy as np

class PolicyNet(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # Embedding 层
        self.embedding0 = tf.keras.layers.Embedding(25, 16)
        self.embedding1 = tf.keras.layers.Embedding(10, 16)
        self.embedding2 = tf.keras.layers.Embedding(10, 16)
        self.embedding3 = tf.keras.layers.Embedding(10, 16)
        self.embedding4 = tf.keras.layers.Embedding(10, 16)
        self.embedding5 = tf.keras.layers.Embedding(10, 16)
        self.embedding6 = tf.keras.layers.Embedding(10, 16)
        self.embedding7 = tf.keras.layers.Embedding(10, 16)
        self.embedding8 = tf.keras.layers.Embedding(10, 16)
        self.embedding9 = tf.keras.layers.Embedding(10, 16)
        self.embedding10 = tf.keras.layers.Embedding(10, 16)
        self.embedding11 = tf.keras.layers.Embedding(10, 16)
        self.embedding12 = tf.keras.layers.Embedding(10, 16)
        self.embedding13 = tf.keras.layers.Embedding(10, 16)
        self.embedding14 = tf.keras.layers.Embedding(10, 16)
        self.embedding15 = tf.keras.layers.Embedding(10, 16)
        self.embedding16 = tf.keras.layers.Embedding(10, 16)

        # 全连接层和 Dropout 层
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax', name='output')

    def call(self, x):
        # Embedding 层
        # x0 = self.embedding0(x[:, 0])
        # x1 = self.embedding1(x[:, 1])
        # x2 = self.embedding2(x[:, 2])
        # x3 = self.embedding3(x[:, 3])
        # x4 = self.embedding4(x[:, 4])
        # x5 = self.embedding5(x[:, 5])
        # x6 = self.embedding6(x[:, 6])
        # x7 = self.embedding7(x[:, 7])
        # x8 = self.embedding8(x[:, 8])
        # x9 = self.embedding9(x[:, 9])
        # x10 = self.embedding10(x[:, 10])
        # x11 = self.embedding11(x[:, 11])
        # x12 = self.embedding12(x[:, 12])
        # x13 = self.embedding13(x[:, 13])
        # x14 = self.embedding14(x[:, 14])
        # x15 = self.embedding15(x[:, 15])
        # x16 = self.embedding16(x[:, 16])

        # # 将所有 embedding 层的输出拼接
        # x_embedding = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16], axis=1)
        
        # 全连接层 + Dropout
        x_dense1 = self.dense1(x)
        
        # 输出层
        x_output = self.output_layer(x_dense1)

        return x_output


class ValueNet(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu', input_dim=state_dim)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)

class PPO:
    def __init__(self, actor_net, critic_net, actor_lr, critic_lr, clip_epsilon):
        self.actor = actor_net
        self.critic = critic_net
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=critic_lr)
        self.clip_epsilon = clip_epsilon

    def take_action(self, state):
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            prob = self.actor(state)
            action_dist = tf.random.categorical(tf.math.log(prob), 1)
            action = action_dist.numpy()[0][0]
            return action,prob.numpy().reshape(-1)
    def train_one_batch(self, states, actions, action_probs_behavior, returns):

        # 1. compute advantage
        states_value = tf.squeeze(self.critic(states)) # shape: [batch_size]
        advantages = tf.stop_gradient(returns - states_value) # shape: [batch_size]

        # 2. actor learn
        with tf.GradientTape() as tape_actor:

            actions_prob_predict = self.actor(states)
            action_probs_target = tf.gather(actions_prob_predict, actions, axis=1, batch_dims=1) # shape: [batch_size]
            ratio = tf.exp(tf.math.log(action_probs_target) - tf.math.log(action_probs_behavior))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio*advantages, clipped_ratio*advantages))

        actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 3 critic learn
        with tf.GradientTape() as tape_actor:
            states_value_preds = tf.squeeze(self.critic(states)) # shape: [batch_size]
            critic_loss = tf.reduce_mean(tf.square(states_value_preds - returns))
        critic_grads = tape_actor.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return actor_loss.numpy().item(), critic_loss.numpy().item()