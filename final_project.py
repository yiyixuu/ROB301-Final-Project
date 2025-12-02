#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt32, Float64MultiArray, Float64, Bool, Int16
import numpy as np
import colorsys
import matplotlib.pyplot as plt


class BayesLoc:
    def __init__(self, p0, colour_codes, colour_map):
        self.colour_sub = rospy.Subscriber("mean_img_rgb", Float64MultiArray, self.colour_callback)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.color_v_pub = rospy.Publisher("color_v", Twist, queue_size=1)
        self.line_follow_pub = rospy.Publisher("line_follow", Int16, queue_size=1)

        self.rate = 10

        self.num_states = len(p0)
        self.colour_codes = np.array(colour_codes)
        self.colour_map = colour_map
        self.probability = p0
        self.state_prediction = np.zeros(self.num_states)

        self.current_state_index = None

        self.cur_colour = None

        self.v_x = 0.05
        self.v_z = 0

        self.transition_models = self._init_transition_models()
        
        self.last_color_idx = None

        self.belief_history = []

    # BayesLoc
    def colour_callback(self, msg):
        """
        callback function that receives the most recent colour measurement from the camera.
        """
        self.cur_colour = np.array(msg.data)  # [r, g, b]
        # print(self.cur_colour)

    def wait_for_colour(self):
        """Loop until a colour is received."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.cur_colour is None:
            rate.sleep()

    def rgb_to_hsv(self, rgb):
        """Convert RGB [0-255] to HSV (Hue in degrees)."""
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return np.array([h * 360, s, v])
    
    def hsv_distance(self, hsv1, hsv2):
        """Compute circular HSV distance with hue wrapping."""
        dh = min(abs(hsv1[0] - hsv2[0]), 360 - abs(hsv1[0] - hsv2[0])) / 180.0
        ds = abs(hsv1[1] - hsv2[1])
        dv = abs(hsv1[2] - hsv2[2])
        return math.sqrt((1.0 * dh)**2 + (0.5 * ds)**2 + (0.2 * dv)**2)
    
    def classify_colour(self, tolerance=0.25):
        """Return color index (0-4) or None if not recognized."""
        if self.cur_colour is None:
            self.wait_for_colour()

        hsv_cur = self.rgb_to_hsv(self.cur_colour)
        hsv_refs = [self.rgb_to_hsv(code) for code in self.colour_codes]

        distances = [self.hsv_distance(hsv_cur, ref) for ref in hsv_refs]
        min_idx = int(np.argmin(distances))
        if distances[min_idx] < tolerance:
            return min_idx
        return None

    def _init_transition_models(self):

        N = self.num_states
        transition_models = {}

        T_neg1 = np.zeros((N, N))
        for i in range(N):
            T_neg1[i, (i - 1) % N] = 0.85 
            T_neg1[i, i]           = 0.10
            T_neg1[i, (i + 1) % N] = 0.05 
        transition_models[-1] = T_neg1

        T_0 = np.zeros((N, N))
        for i in range(N):
            T_0[i, (i - 1) % N] = 0.05
            T_0[i, i]           = 0.90
            T_0[i, (i + 1) % N] = 0.05
        transition_models[0] = T_0

        T_pos1 = np.zeros((N, N))
        for i in range(N):
            T_pos1[i, (i - 1) % N] = 0.05
            T_pos1[i, i]           = 0.10
            T_pos1[i, (i + 1) % N] = 0.85
        transition_models[1] = T_pos1

        return transition_models

    def state_model(self, u):
        """
        State model: p(x_{k+1} | x_k, u)

        """
        if u not in self.transition_models:
            u = 0
        return self.transition_models[u]

    def measurement_model(self, x=None):
        """
        Measurement model p(z_k | x_k = colour) - given the pixel intensity,
        what's the probability that of each possible colour z_k being observed?
        """
        if self.cur_colour is None:
            return np.ones(len(self.colour_codes)) / float(len(self.colour_codes))

        hsv_cur = self.rgb_to_hsv(self.cur_colour)
    
        dists = []
        for ref in self.colour_codes:
            hsv_ref = self.rgb_to_hsv(ref)
            dists.append(self.hsv_distance(hsv_cur, hsv_ref))

        dists = np.array(dists)

        eps = 1e-6
        inv = 1.0 / (dists + eps)
        prob = inv / np.sum(inv)

        return prob

    def state_predict(self, u=1):
        rospy.loginfo("predicting state")
        T = self.state_model(u) 
        self.state_prediction = T.T.dot(self.probability)

        total = np.sum(self.state_prediction)
        if total > 0:
            self.state_prediction /= total
        else:
            self.state_prediction = np.ones(self.num_states) / float(self.num_states)

    def state_update(self):
        rospy.loginfo("updating state")
        colour_likelihoods = self.measurement_model()

        state_likelihoods = np.zeros(self.num_states)
        for i, colour_idx in enumerate(self.colour_map):
            state_likelihoods[i] = colour_likelihoods[colour_idx]

        unnormalized = self.state_prediction * state_likelihoods

        total = np.sum(unnormalized)
        if total > 0:
            self.probability = unnormalized / total
        else:
            self.probability = self.state_prediction.copy()

        ml_idx = int(np.argmax(self.probability))
        self.current_state_index = ml_idx
        rospy.loginfo("Most likely state: index %d, colour %d, belief=%.3f",
                      ml_idx, self.colour_map[ml_idx], self.probability[ml_idx])

def is_color(col1, col2, tol):
        return max(abs(c1-c2) for c1, c2 in zip(col1, col2)) <= tol

if __name__ == "__main__":

    # This is the known map of offices by colour
    # 0: red, 1: green, 2: blue/purple, 3: yellow, 4: line
    # current map starting at cell #2 and ending at cell #12
    colour_map = [3, 1, 2, 0, 0, 1, 2, 0, 3, 1, 2]

    # TODO calibrate these RGB values to recognize when you see a colour
    # NOTE: you may find it easier to compare colour readings using a different
    # colour system, such as HSV (hue, saturation, value). To convert RGB to
    # HSV, use:
    # h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    colour_codes = [
        [250, 140, 67],  # red
        [171, 178, 168],  # green
        [194, 124, 164],  # blue/purple
        [180, 158, 137],  # yellow
        [149, 135, 138],  # line
    ]

    p0 = np.ones_like(colour_map) / len(colour_map)

    localizer = BayesLoc(p0, colour_codes, colour_map)

    rospy.init_node("final_project")
    rospy.sleep(0.5)
    rate = rospy.Rate(localizer.rate)
    tolerance = 10

    confirmed_color = None
    color_buffer = []
    CONFIRM_FRAMES = 11

    localizing_iters_left = 15

    stops = [0, 4, 5] # indices not colors

    twist = Twist()
    while not rospy.is_shutdown():
        raw_color_idx = localizer.classify_colour(tolerance=0.25)

        color_buffer.append(raw_color_idx)
        if len(color_buffer) > CONFIRM_FRAMES:
            color_buffer.pop(0)

        if len(color_buffer) == CONFIRM_FRAMES and len(set(color_buffer)) == 1:
            stable_color = color_buffer[-1]

            if stable_color in [0, 1, 2, 3]:

                confirmed_color = stable_color

            elif stable_color == 4:
    
                confirmed_color = None
                color_buffer.clear()

            else:
                pass

        if confirmed_color in [0, 1, 2, 3]:
            u=1

        else:
            u=0
        localizer.state_predict(u=u)

        if confirmed_color in [0, 1, 2, 3]:
            if confirmed_color != localizer.last_color_idx:
                localizer.state_update()
                localizing_iters_left -= 1
                localizer.belief_history.append(localizer.probability.copy())
        localizer.last_color_idx = confirmed_color

        if raw_color_idx in [0, 1, 2, 3]:
            color_names = ["red", "green", "blue", "yellow"]
            rospy.loginfo(f"Detected {color_names[raw_color_idx]}")
            twist.linear.x = localizer.v_x
            twist.angular.z = 0
            localizer.color_v_pub.publish(twist)
            localizer.line_follow_pub.publish(0)
        else:
            rospy.loginfo("LINE detected — following line...")
            twist.linear.x = localizer.v_x
            twist.angular.z = 0
            localizer.color_v_pub.publish(twist)
            localizer.line_follow_pub.publish(1)

        if localizing_iters_left <= 0:
            rospy.loginfo("finished loc")
            if localizer.current_state_index in stops:

                stops.remove(localizer.current_state_index)
                start = rospy.Time.now()
                twist.linear.x = 0
                twist.angular.z = math.pi/2
                while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < (math.pi/2/twist.angular.z):
                    rospy.loginfo("turning left")
                    localizer.color_v_pub.publish(twist)
                    localizer.line_follow_pub.publish(2)
                    rate.sleep()
                rate.sleep()
                start = rospy.Time.now()
                twist.linear.x = 0
                twist.angular.z = -math.pi/2
                while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < (math.pi/2/abs(twist.angular.z)):
                    rospy.loginfo("turning back")
                    localizer.color_v_pub.publish(twist)
                    localizer.line_follow_pub.publish(2)
                    rate.sleep()

        rate.sleep()

    rospy.loginfo("finished!")
    rospy.loginfo(localizer.probability)


    belief_matrix = np.array(localizer.belief_history).T   

    plt.figure(figsize=(10, 5))
    plt.imshow(belief_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Belief Probability")
    plt.xlabel("Time step")
    plt.ylabel("State index")
    plt.title("Belief Evolution Over Time")
    plt.show()