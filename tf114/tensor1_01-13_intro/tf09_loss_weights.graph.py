import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_random_seed(518)

#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]

w = tf.compat.v1.placeholder(tf.float32)

#2. 모델
hypothesis = x * w

#3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

wght_hist = []
loss_hist = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_wght = i 
        curr_loss = sess.run(loss, feed_dict = {w: curr_wght})
        
        wght_hist.append(curr_wght)
        loss_hist.append(curr_loss)

print("🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹[WGHT_history]🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print(wght_hist)
print("🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹[LOSS_history]🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
print(loss_hist)

plt.plot(wght_hist, loss_hist)
plt.xlabel("WGHT")
plt.ylabel("LOSS")
plt.grid()
plt.show()