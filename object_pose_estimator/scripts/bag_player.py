# import rosbag
# import rospy
# import argparse
# from std_msgs.msg import Bool

# is_playing = False

# def toggle_callback(msg):
#     global is_playing
#     is_playing = msg.data
#     rospy.logwarn("Toggle play state -> %s" % ("Playing" if is_playing else "Paused"))

# def bag_play(bag_file):
#     global is_playing
#     rospy.init_node('bag_player', anonymous=True)
#     rospy.loginfo("Bag file: %s" % bag_file)

#     rospy.Subscriber("/toggle_play", Bool, toggle_callback)

#     pub_dict = {}
#     with rosbag.Bag(bag_file, 'r') as bag:
#         start_time = None
#         play_start = rospy.Time.now()

#         for topic, msg, t in bag.read_messages():
#             if start_time is None:
#                 start_time = t
#                 play_start = rospy.Time.now()

#             elapsed = (t - start_time).to_sec()

#             # ⏸ 在这里检查暂停状态（全局阻塞）
#             while not is_playing and not rospy.is_shutdown():
#                 print("Paused...", end='\r')
#                 rospy.sleep(0.1)

#             # 等待到对应时间戳
#             while (rospy.Time.now() - play_start).to_sec() < elapsed and not rospy.is_shutdown():
#                 rospy.sleep(0.001)

#             # 动态创建 publisher
#             if topic not in pub_dict:
#                 pub_dict[topic] = rospy.Publisher(topic, type(msg), queue_size=10)
#                 rospy.loginfo("Publishing topic: %s" % topic)

#             pub_dict[topic].publish(msg)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--bag", required=True, help="Path to bag file")
#     args = parser.parse_args()
#     bag_play(args.bag)


#!/usr/bin/env python
import rosbag
import rospy
import argparse
from std_msgs.msg import Bool

is_playing = False  # 初始暂停；True=播放，False=暂停

def toggle_callback(msg):
    global is_playing
    is_playing = bool(msg.data)
    rospy.loginfo("Command -> %s" % ("Playing" if is_playing else "Paused"))

def bag_play(bag_file):
    global is_playing
    rospy.init_node('bag_player', anonymous=True)
    rospy.loginfo("Bag file: %s" % bag_file)

    rospy.Subscriber("/toggle_play", Bool, toggle_callback)

    pub_dict = {}
    with rosbag.Bag(bag_file, 'r') as bag:
        start_time = None          # bag 内的第一个时间戳
        playback_time = 0.0        # 虚拟播放时钟（秒），只在播放时累计
        wall_last = rospy.Time.now()  # 上一次采样墙钟

        for topic, msg, t in bag.read_messages():
            if start_time is None:
                start_time = t
                playback_time = 0.0
                wall_last = rospy.Time.now()
                rospy.loginfo("Playback starts at bag t=%.6f" % start_time.to_sec())

            # 当前帧相对 bag 起点的目标时间（秒）
            target = (t - start_time).to_sec()

            # 用虚拟时钟对齐到 target；暂停时 playback_time 不变
            while not rospy.is_shutdown() and playback_time < target:
                now = rospy.Time.now()

                if is_playing:
                    dt = (now - wall_last).to_sec()
                    wall_last = now
                    playback_time += dt  # 如需倍速：这里乘以 rate
                    rospy.sleep(0.001)   # 防忙等
                else:
                    wall_last = now      # 保持墙钟同步，避免恢复时跳帧
                    rospy.sleep(0.05)
                    continue

            # 发布前再次确认：如果此刻被按了暂停，就等恢复
            while not rospy.is_shutdown() and not is_playing:
                wall_last = rospy.Time.now()
                rospy.sleep(0.05)
                # 注意：暂停期间 wall_last 更新，playback_time 不会偷偷前进

            # 动态创建 publisher
            if topic not in pub_dict:
                pub_dict[topic] = rospy.Publisher(topic, type(msg), queue_size=10)
                rospy.loginfo("Publishing topic: %s" % topic)

            pub_dict[topic].publish(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to bag file")
    args = parser.parse_args()
    bag_play(args.bag)
