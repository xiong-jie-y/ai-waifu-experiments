import zmq
import time
from random import randrange

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5555")

message_id = 0
while True:
  zipcode     = randrange(10001, 10010)
  temprature  = randrange(0, 215) - 80
  relhumidity = randrange(0, 50) + 10

  update = "%05d %d %d %d" % (zipcode, temprature, relhumidity, message_id)
  message_id += 1
  print(update)
  time.sleep(1.0)
  data = [zipcode, temprature, relhumidity, message_id]
  new_data = []
  for dtm in data:
      new_data.append(str(dtm).encode('utf-8'))
  publisher.send_multipart(new_data)