import tensorflow as tf
import numpy as np
tf.random_seed(1)
class PolicyNet(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # Embedding 层
        self.embedding0 = tf.keras.layers.Embedding(25, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding1 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding2 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding3 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding4 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding5 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding6 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding7 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding8 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding9 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding10 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding11 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding12 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding13 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding14 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding15 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())
        self.embedding16 = tf.keras.layers.Embedding(10, 16, embeddings_initializer=tf.keras.initializers.GlorotUniform())

        # 全连接层和 Dropout 层
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax', name='output', kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, x):
        # Embedding 层
        x0 = self.embedding0(x[:, 0])
        x1 = self.embedding1(x[:, 1])
        x2 = self.embedding2(x[:, 2])
        x3 = self.embedding3(x[:, 3])
        x4 = self.embedding4(x[:, 4])
        x5 = self.embedding5(x[:, 5])
        x6 = self.embedding6(x[:, 6])
        x7 = self.embedding7(x[:, 7])
        x8 = self.embedding8(x[:, 8])
        x9 = self.embedding9(x[:, 9])
        x10 = self.embedding10(x[:, 10])
        x11 = self.embedding11(x[:, 11])
        x12 = self.embedding12(x[:, 12])
        x13 = self.embedding13(x[:, 13])
        x14 = self.embedding14(x[:, 14])
        x15 = self.embedding15(x[:, 15])
        x16 = self.embedding16(x[:, 16])

        # 将所有 embedding 层的输出拼接
        x_embedding = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16], axis=1)
        
        # 全连接层 + Dropout
        x_dense1 = self.dense1(x_embedding)
        
        x_dense2 = self.dense2(x_dense1)
        # 输出层
        x_output = self.output_layer(x_dense2)

        return x_output


class ValueNet(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), input_dim=state_dim)
        self.fc2 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, minibatch_size, device):
        self.minibatch_size = minibatch_size
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic = ValueNet(state_dim, hidden_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数

    def take_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state)
        action_dist = tf.random.categorical(tf.math.log(probs), 1)
        action = action_dist.numpy()[0][0]
        return action

    def offline_train(self, states, actions, returns):
        # 创建数据集并批量化
        dataset = tf.data.Dataset.from_tensor_slices((states.astype(np.float32), actions.astype(np.int32).reshape(-1), returns.astype(np.float32).reshape(-1)))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.minibatch_size)

        print("load_data done.")

        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_returns in dataset:

                # 在每个批次中将数据转换为张量
                # batch_states = tf.convert_to_tensor(batch_states)
                # batch_actions = tf.reshape(tf.convert_to_tensor(batch_actions), (-1, 1))
                # batch_returns = tf.reshape(tf.convert_to_tensor(batch_returns), (-1, 1))

                # 计算 td_target 和 advantage
                td_target = batch_returns
                advantage = td_target - self.critic(batch_states)
                advantage = tf.stop_gradient(advantage)

                # 计算 old_log_probs
                old_log_probs = tf.math.log(self.actor(batch_states))
                old_log_probs = tf.gather(old_log_probs, batch_actions, axis=1, batch_dims=1)

                # 使用 with 语句管理计算图的创建，避免每个 batch 创建新图
                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    # 计算 log_probs
                    log_probs = tf.math.log(self.actor(batch_states))
                    
                    # 从 log_probs 中选择动作的概率
                    log_probs = tf.gather(log_probs, batch_actions, axis=1, batch_dims=1)

                    # 计算比率 ratio
                    ratio = tf.exp(log_probs - old_log_probs)

                    # 计算目标的优势函数
                    surr1 = ratio * advantage
                    surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantage
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                    # 计算 critic 损失
                    critic_loss = tf.reduce_mean(tf.square(self.critic(batch_states) - td_target))

                    print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}")

                # 计算梯度
                actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

                # 应用梯度更新
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
