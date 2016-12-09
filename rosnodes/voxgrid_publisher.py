import rospy
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np


topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, MarkerArray)

rospy.init_node('register')

markerArray = MarkerArray()
count = 0

MARKERS_MAX = 3000
trans = [0.1,0.1,0.1]

rate = rospy.Rate(1) # 10hz


grid = np.ones((32,32,32))


while not rospy.is_shutdown():
    
    for index in np.ndindex(grid.shape):
        if(grid[index] == 0):
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
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        trans[0] = index[0] * 0.2
        trans[1]=  index[1] * 0.2
        trans[2]=  index[2] * 0.2
        marker.pose.position.x = trans[0]
        marker.pose.position.y = trans[1]
        marker.pose.position.z = trans[2]
        marker.pose.orientation.w = 1
    
        marker.color.g = 1.0;
    
        marker.lifetime.secs = 60
        marker.lifetime.nsecs = 60
        markerArray.markers.append(marker)
        # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        # Publish the MarkerArray
    print "Created grid"
    publisher.publish(markerArray)
    markerArray = MarkerArray()
    rate.sleep()
