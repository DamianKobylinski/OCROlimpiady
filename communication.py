import socket
import struct


# TODO thread method
class Communication:
    def __init__(self):
        self.srcSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dstSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def Host(self, host, port):
        self.srcSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srcSock.bind((host, port))
        self.srcSock.listen(1)
        self.dstSock, self.dstAddr = self.srcSock.accept()

    def Connect(self, host, port):
        self.dstSock.connect((host, port))

    def Send(self, header, data):
        self.dstSock.sendall(header+data)

    def RecvFirstChunk(self, headerSize, data):
        while len(data) < headerSize:
            packet = self.dstSock.recv(4*1024)
            if not packet:
                break
            data += packet
        self.ReadHeader(headerSize, data)
        return data

    # Implement your own protocol handling
    def ReadHeader(self, headerSize, data):
        self.msgSize = struct.unpack("!I", data[:headerSize])[0]

    def HandleHeaderParams(self):
        return self.msgSize == 0xffffffff
    # ---

    def RecvSecondChunk(self, headerSize, data):
        data = data[headerSize:]
        while len(data) < self.msgSize:
            data += self.dstSock.recv(4*1024)
        return data[:self.msgSize], data[self.msgSize:]

    def Close(self):
        self.srcSock.close()
        self.dstSock.close()
