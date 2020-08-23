import zmq

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:9998")

subscriber.setsockopt(zmq.SUBSCRIBE, "FaceLandmarks".encode())

while True:
    topic = subscriber.recv()
    print(topic)
    topic = topic.decode('utf-8')
    print(topic)