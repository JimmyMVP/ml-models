import rospy
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import os
import json
import numpy as np


ENV = os.environ
SUMMARY_DIR = ENV["SUMMARY_DIR"]


topic = 'output_visualization'
publisher = rospy.Publisher(topic, MarkerArray)

rospy.init_node('register')

markerArray = MarkerArray()
count = 0

MARKERS_MAX = 3000
trans = [0.1,0.1,0.1]

rate = rospy.Rate(0.5) # 10hz


#Load the data 

grids = np.load(open(SUMMARY_DIR + "outputs/sidney", "rb"))


while not rospy.is_shutdown():
    
    for grid in grids:
        for index in np.ndindex(grid.shape):
            if(grid[index] <= 0.3):
                continue
            # ... here I get the data I want to plot into a vector called trans
            marker = Marker()
            marker.id = count
            count+=1
            marker.header.frame_id = "/map"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1
            marker.pose.orientation.w = 1.0
            trans[0] = index[0] * 0.2 - grid.shape[0]//2 * 0.2
            trans[1]=  index[1] * 0.2 - grid.shape[1]//2 * 0.2
            trans[2]=  index[2] * 0.2 - grid.shape[2]//2 * 0.2
            marker.pose.position.x = trans[0]
            marker.pose.position.y = trans[1]
            marker.pose.position.z = trans[2]
            marker.pose.orientation.w = 1
        
            marker.color.g = min(grid[index], 1)
            marker.color.b = 1 - marker.color.g
        
            marker.lifetime.secs = 2
            marker.lifetime.nsecs = 2
            markerArray.markers.append(marker)
        
        print "Created output grid"

        publisher.publish(markerArray)
        markerArray = MarkerArray()
        rate.sleep()
