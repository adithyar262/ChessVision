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
#include <unistd.h>
#include "shader.hpp"
#include "texture.hpp"
#include "renderer.hpp"

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

std::string currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";  // Initial position
std::mutex fenMutex;

void receiveFen() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(12345);
    inet_pton(AF_INET, "127.0.0.1", &serverAddr.sin_addr);

    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        close(sock);
        return;
    }

    char buffer[128];
    while (true) {
        int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::lock_guard<std::mutex> lock(fenMutex);
            currentFen = buffer;
        } else if (bytesReceived == 0) {
            std::cout << "Server disconnected" << std::endl;
            break;
        } else {
            std::cerr << "recv failed" << std::endl;
            break;
        }
    }

    close(sock);
}


int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

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
        glClear(GL_COLOR_BUFFER_BIT);
        // renderer.updateBoard(fen);

        {
            std::lock_guard<std::mutex> lock(fenMutex);
            renderer.updateBoard(currentFen);
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
