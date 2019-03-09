import numpy as np



client_greeting=">Client connected"
server_greeting=">Server connected"
chunksize =15
port = 8765
ip_local = "localhost"
ip = '192.168.1.3'
#client_connection = 'wss://'+ip + ":"+ str(port)
client_connection ='wss://localhost:8765'
client_connection_local = 'wss://localhost:8765'
midiout =10
midiin =4
clock=False


#Chesscam config:

cam_white_areas =(5,5) #Small number = small white areas
colorBoundaries= [
            [np.array([10, 10, 10]), np.array([255, 56, 50])], # red
            [np.array([0, 70, 5]), np.array([50, 200, 50])],   # green
            [np.array([4, 31, 86]), np.array([50, 88, 220])]    # blue
        ]





# colorBoundaries= [
#             [np.array([100, 15, 17]), np.array([200, 56, 50])], # red
#             [np.array([0, 70, 5]), np.array([50, 200, 50])],   # green
#             [np.array([4, 31, 86]), np.array([50, 88, 220])]    # blue
#         ]
