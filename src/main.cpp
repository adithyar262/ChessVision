#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <csignal>
#include <unistd.h> 
#include "shader.hpp"
#include "texture.hpp"
#include "renderer.hpp"

std::string currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";  // Initial position
std::string currentScore = "0.0";  // Initial score
std::mutex fenMutex;

#pragma comment(lib, "ws2_32.lib")

struct TextureCoord {
    float u, v;
};

std::unordered_map<char, TextureCoord> pieceToTexture = {
    {'R', {0.0f, 0.0f}},        // White Rook
    {'N', {0.125f, 0.0f}},      // White Knight
    {'B', {0.25f, 0.0f}},       // White Bishop
    {'Q', {0.375f, 0.0f}},      // White Queen
    {'K', {0.5f, 0.0f}},        // White King
    {'P', {0.0f, 0.125f}},      // White Pawn
    {'r', {0.0f, 0.875f}},      // Black Rook
    {'n', {0.125f, 0.875f}},    // Black Knight
    {'b', {0.25f, 0.875f}},     // Black Bishop
    {'q', {0.375f, 0.875f}},    // Black Queen
    {'k', {0.5f, 0.875f}},      // Black King
    {'p', {0.0f, 0.75f}},       // Black Pawn
    {' ', {0.5f, 0.5f}}         // Empty square
};

// void receiveFen() {
//     int sock = socket(AF_INET, SOCK_STREAM, 0);
//     if (sock < 0) {
//         std::cerr << "Socket creation failed" << std::endl;
//         return;
//     }

//     struct sockaddr_in serverAddr;
//     serverAddr.sin_family = AF_INET;
//     serverAddr.sin_port = htons(12345);
//     inet_pton(AF_INET, "127.0.0.1", &serverAddr.sin_addr);

//     if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
//         std::cerr << "Connection failed" << std::endl;
//         close(sock);
//         return;
//     }

//     char buffer[128];
//     while (true) {
//         int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
//         if (bytesReceived > 0) {
//             buffer[bytesReceived] = '\0';
//             std::lock_guard<std::mutex> lock(fenMutex);
//             currentFen = buffer;
//         } else if (bytesReceived == 0) {
//             std::cout << "Server disconnected" << std::endl;
//             break;
//         } else {
//             std::cerr << "recv failed" << std::endl;
//             break;
//         }
//     }

//     close(sock);
// }

int server_fd = -1; // Global variable to store the server socket

void signalHandlerForMain(int signum) {
    if (signum == SIGINT) {
        std::cout << "\nSIGINT received. Cleaning up and exiting..." << std::endl;
        if (server_fd != -1) {
            close(server_fd); // Close the server socket
            std::cout << "Socket closed." << std::endl;
        }
        exit(0); // Exit gracefully
    }
}

void setupSignalHandler() {
    struct sigaction sa = {};
    sa.sa_handler = signalHandlerForMain; // Set the handler function
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGINT, &sa, nullptr) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
}

void receiveFen() {
    setupSignalHandler(); // Set up the signal handler at the start of the function

    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        std::cerr << "Setsockopt failed" << std::endl;
        return;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(12345); // Use a fixed port or dynamic port as needed

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return;
    }

    if (listen(server_fd, 1) < 0) {
        std::cerr << "Listen failed" << std::endl;
        return;
    }

    std::cout << "Listening on port 12345..." << std::endl;

    int new_socket;
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        std::cerr << "Accept failed" << std::endl;
        return;
    }

    char buffer[128];
    while (true) {
        int bytesReceived = recv(new_socket, buffer, sizeof(buffer), 0);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::string receivedData = buffer;
            size_t delimiterPos = receivedData.find('|');
            if (delimiterPos != std::string::npos) {
                std::string fen = receivedData.substr(0, delimiterPos);
                std::string scoreStr = receivedData.substr(delimiterPos + 1);
                std::cout << "FEN Received: " << fen << std::endl;
                std::cout << "Score: " << scoreStr << std::endl;
                
                float score = std::stof(scoreStr);
                
                std::lock_guard<std::mutex> lock(fenMutex);
                currentFen = fen;
                currentScore = score;
            }
        } else if (bytesReceived == 0) {
            std::cout << "Client disconnected" << std::endl;
            break;
        } else {
            std::cerr << "recv failed" << std::endl;
            break;
        }
    }

    close(new_socket);
}


int main() {
    // int delay = 0;
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    struct sigaction sa = {};
    sa.sa_handler = signalHandlerForMain;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA Chessboard", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    Shader shader("../shaders/vertex.glsl", "../shaders/fragment.glsl");
    Texture texture("../textures/board.png");
    Texture overlayTexture("../textures/pieces.png");
    Renderer renderer;

    std::string fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R";

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::vector<float> piecePositions(128, -1.0f);
    GLint piecePositionsLoc = glGetUniformLocation(shader.ID, "piecePositions");

    std::thread fenThread(receiveFen);

    while (!glfwWindowShouldClose(window)) {
        // delay += 1;
        // if(delay == 1000)
        //     renderer.UpdateVertices(1.0);

        glClear(GL_COLOR_BUFFER_BIT);

        {
            std::lock_guard<std::mutex> lock(fenMutex);
            // std::cout<<"Current FEN : "<<currentFen<<std::endl;
            renderer.updateBoard(currentFen);
            float score = std::stof(currentScore);
            renderer.UpdateVertices(score);
        }

        shader.use();
        glActiveTexture(GL_TEXTURE0);
        texture.bind();
        glUniform1i(glGetUniformLocation(shader.ID, "boardTexture"), 0);

        glActiveTexture(GL_TEXTURE1);
        overlayTexture.bind();
        glUniform1i(glGetUniformLocation(shader.ID, "overlayTexture"), 1);

        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                char piece = renderer.board[row][col];
                int index = row * 8 + col;
                if (piece != ' ' && pieceToTexture.find(piece) != pieceToTexture.end()) {
                    piecePositions[index * 2] = pieceToTexture[piece].u;
                    piecePositions[index * 2 + 1] = pieceToTexture[piece].v;
                } else {
                    piecePositions[index * 2] = -1.0f;
                    piecePositions[index * 2 + 1] = -1.0f;
                }
            }
        }

        glUniform2fv(piecePositionsLoc, 64, piecePositions.data());

        renderer.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    fenThread.join();
    glfwTerminate();
    return 0;
}
