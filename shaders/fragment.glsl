#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D boardTexture;
uniform sampler2D overlayTexture; // New texture

void main() {
    vec4 baseColor = texture(boardTexture, TexCoord);
    vec4 overlayColor = texture(overlayTexture, TexCoord);
    FragColor = mix(baseColor, overlayColor, overlayColor.a); // Blend based on alpha
    // FragColor = texture(overlayTexture, TexCoord);
}
