#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "cuda_kernel.h"

#include "main.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    std::string vertexCode = readShaderFile(vertexPath);
    std::string fragmentCode = readShaderFile(fragmentPath);

    GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexCode.c_str());
    GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentCode.c_str());

    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);

    checkCompileErrors(ID, "PROGRAM");

    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use() { 
    glUseProgram(ID); 
}

std::string Shader::readShaderFile(const char* filePath) {
    std::ifstream shaderFile;
    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        shaderFile.open(filePath);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        return shaderStream.str();
    }
    catch (std::ifstream::failure& e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        return "";
    }
}

GLuint Shader::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    checkCompileErrors(shader, type == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT");
    return shader;
}

void Shader::checkCompileErrors(GLuint shader, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    }
}

Texture::Texture(const char* path) {
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_2D, ID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_set_flip_vertically_on_load(true);
    int width, height, nrChannels;
    unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);

    std::cout<<"Path - "<<path<<" Width - "<<width<< " Height - "<<height<<" nrChannels - "<<nrChannels<<std::endl;
    
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cerr << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
}

void Texture::bind() {
    glBindTexture(GL_TEXTURE_2D, ID);
}

std::vector<std::vector<char>> parseFEN(const std::string& fen) {
    std::vector<std::vector<char>> board(8, std::vector<char>(8, ' '));
    int row = 0, col = 0;  // Start from the bottom row
    
    for (char c : fen) {
        if (c == '/') {
            // row++;
            col = 0;
        } else if (c >= '1' && c <= '8') {
            col += c - '0';
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            if (row >= 0 && row < 8 && col >= 0 && col < 8) {
                board[row][col] = c;
            }
            col++;
        }
        if (col >= 8) {
            row++;
            col = 0;
        }
        if (row >= 8) break;
    }
    return board;
}


void Renderer::updateBoard(const std::string& fen) {
    board = parseFEN(fen);
    // Debug print
    std::cout << "Current board state:" << std::endl;
    for (int row = 7; row >= 0; --row) {
        for (int col = 0; col < 8; ++col) {
            std::cout << (board[row][col] == ' ' ? '.' : board[row][col]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


Renderer::Renderer() {
    const float step = 1.0f / 8.0f;
    const int gridSize = 8;
    const int verticesPerSquare = 6;
    const int componentsPerVertex = 5;
    const int totalVertices = gridSize * gridSize * verticesPerSquare;

    float v1[totalVertices * componentsPerVertex];

    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            float x1 = -1.0f + (col * 2 * step);
            float x2 = x1 + 2 * step;
            float y1 = -1.0f + (row * 2 * step);
            float y2 = y1 + 2 * step;

            int baseIndex = (row * gridSize + col) * verticesPerSquare * componentsPerVertex;

            // First triangle
            v1[baseIndex     ] = x1; v1[baseIndex + 1] = y2; v1[baseIndex + 2] = 0.0f; v1[baseIndex + 3] = col * step; v1[baseIndex + 4] = (row + 1) * step;
            v1[baseIndex + 5 ] = x2; v1[baseIndex + 6] = y2; v1[baseIndex + 7] = 0.0f; v1[baseIndex + 8] = (col + 1) * step; v1[baseIndex + 9] = (row + 1) * step;
            v1[baseIndex + 10] = x2; v1[baseIndex + 11] = y1; v1[baseIndex + 12] = 0.0f; v1[baseIndex + 13] = (col + 1) * step; v1[baseIndex + 14] = row * step;

            // Second triangle
            v1[baseIndex + 15] = x1; v1[baseIndex + 16] = y2; v1[baseIndex + 17] = 0.0f; v1[baseIndex + 18] = col * step; v1[baseIndex + 19] = (row + 1) * step;
            v1[baseIndex + 20] = x2; v1[baseIndex + 21] = y1; v1[baseIndex + 22] = 0.0f; v1[baseIndex + 23] = (col + 1) * step; v1[baseIndex + 24] = row * step;
            v1[baseIndex + 25] = x1; v1[baseIndex + 26] = y1; v1[baseIndex + 27] = 0.0f; v1[baseIndex + 28] = col * step; v1[baseIndex + 29] = row * step;
        }
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v1), v1, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Initialize the board with empty squares
    board = std::vector<std::vector<char>>(8, std::vector<char>(8, ' '));
}

void Renderer::draw() {
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6*64);
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
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

    /*
    Openng position: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    Middle game position: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
    End game position: "4k3/8/8/8/8/8/4P3/4K3"
    */

    renderer.updateBoard(fen);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::vector<float> piecePositions(128, -1.0f);
    GLint piecePositionsLoc = glGetUniformLocation(shader.ID, "piecePositions");

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        
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

    glfwTerminate();
    return 0;
}
