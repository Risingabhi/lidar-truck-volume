## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# First import the library
import pyrealsense2 as rs

# REQUIRED 
#python version =3.6
#numpy = 1.16 EDITED On 19/3/23


# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

# Declare RealSense pipeline, encapsulating the actual device and sensors

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # rs.config.enable_device_from_file(config,"/content/20220205_171337.bag")

pipe = rs.pipeline()
config = rs.config()

rs.config.enable_device_from_file(config,"empty_truck.bag")
# Enable depth stream
config.enable_stream(rs.stream.depth, rs.format.z16, 30)

# Start streaming with chosen configuration
pipe.start(config)

# We'll use the colorizer to generate texture for our PLY
# (alternatively, texture can be obtained from color or infrared stream)
colorizer = rs.colorizer()

try:
    # Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames()
    colorized = colorizer.process(frames)

    # Create save_to_ply object
    ply = rs.save_to_ply("empty_truck.ply")

    # Set options to the desired values
    # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving to 1.ply...")
    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(colorized)
    print("Done")
finally:
    pipe.stop()
