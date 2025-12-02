#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt32, Float64, Bool, Int16

import matplotlib.pyplot as plt


class Controller(object):
    def __init__(self):
        # publish motor commands
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        # subscribe to detected line index
        self.color_sub = rospy.Subscriber(
            "line_idx", UInt32, self.camera_callback, queue_size=1
        ) 

        self.color_v_sub = rospy.Subscriber("color_v", Twist, self.color_v_callback, queue_size=1)
        self.line_follow_sub = rospy.Subscriber("line_follow", Int16, self.line_follow_callback, queue_size=1)


        self.error_array = []
        self.error = 0
        self.last_error = 0
        self.last_t = rospy.get_time()
        self.target_location = 320

        self.derivative = 0
        self.integral = 0

        # speed = 0.26
        # self.kp = 0.0075
        # self.ki = 0.0001
        # self.kd = 0.1

        # speed = 0.2
        # self.kp = 0.005
        # self.ki = 0.0001
        # self.kd = 0.1

        # speed = 0.1
        self.kp = 0.005
        self.ki = 0.0001
        self.kd = 0.1

        self.color_v_x = 0
        self.color_v_z = 0
        self.line_follow = 1

    def camera_callback(self, msg):
        """Callback for line index."""
        self.error = self.target_location-msg.data
        self.error_array.append(self.error)
        pass

    def color_v_callback(self, msg):
        self.color_v_x = msg.linear.x
        self.color_v_z = msg.angular.z

    def line_follow_callback(self, msg):
        self.line_follow = msg.data


    def follow_the_line(self):
        """
        TODO: complete the function to follow the line
        """
        rate = rospy.Rate(10)
        current_t = rospy.get_time()
        dt = current_t - self.last_t if current_t > self.last_t else 1e-6
        self.derivative = (self.error-self.last_error) / dt
        self.integral += dt * (self.error + self.last_error)/2

        twist = Twist()
        # twist.linear.x = 0.20

        # Bang-Bang Controller
        # while not rospy.is_shutdown():
        #     if self.error > 0: 
        #         twist.angular.z = 0.2
        #         rospy.loginfo("moved left")
        #     elif self.error < 0:
        #         twist.angular.z = -0.2
        #         rospy.loginfo("moved right")
        #     else:
        #         twist.angular.z = 0

        #     self.cmd_pub.publish(twist)
        #     rate.sleep()

        # P Controller
        # PI Controller

        # PID Controller
        while not rospy.is_shutdown():
            twist.linear.x = self.color_v_x


            if self.line_follow == 1:
                rospy.loginfo("line following")

                control = self.kp * self.error + self.ki * self.integral + self.kd * self.derivative
                twist.angular.z = control
            elif self.line_follow == 0: 
                rospy.loginfo("driving straight")
                twist.angular.z = 0
            elif self.line_follow == 2: 
                rospy.loginfo("delivering")
                twist.linear.x = 0
                twist.angular.z = self.color_v_z
                
            self.cmd_pub.publish(twist)


            self.last_t = current_t
            self.last_error = self.error
            rate.sleep()


        # plt.plot(self.error_array)
        # plt.axhline(y=0, color='r')
        # plt.show()

if __name__ == "__main__":
    rospy.init_node("lab3")
    controller = Controller()
    controller.follow_the_line()