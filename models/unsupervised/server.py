import socket
import threading

class Reader(threading.Thread):
    def __init__(self, client):
        threading.Thread.__init__(self)
        self.client = client

    def run(self):
        while True:
            data = self.client.recv(1024)
            if data:
                string = bytes.decode(data, 'utf-8')
                print(string, end='')
            else:
                break
        print("close", self.client.getpeername())

class Listener(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        self.sock.bind(('localhost', port))
        self.sock.listen(0)
    def run(self):
        print('listen start')
        while True:
            client, cltadd = self.sock.accept()
if __name__ == '__main__':
    l = Listener(9011)
    l.start()