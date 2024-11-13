import socket
import time
import random

def generate_random_fen():
    pieces = 'rnbqkpRNBQKP'
    fen = ''
    for _ in range(8):
        empty = 0
        for _ in range(8):
            if random.random() < 0.5:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += random.choice(pieces)
        if empty > 0:
            fen += str(empty)
        if _ < 7:
            fen += '/'
    return fen + ' w KQkq - 0 1'

def main():
    host = '127.0.0.1'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}")
    
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    try:
        while True:
            fen = generate_random_fen()
            print(f"Sending FEN: {fen}")
            conn.sendall(fen.encode())
            time.sleep(5)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()
