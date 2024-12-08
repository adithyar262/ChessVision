#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D boardTexture;
uniform sampler2D overlayTexture;
uniform vec2 piecePositions[64];

void main()
{
    vec4 boardColor = texture(boardTexture, TexCoord);
    vec2 pieceTexCoord = vec2(-1.0, -1.0);
    
    int squareIndex = int(TexCoord.x * 8) + int(TexCoord.y * 8) * 8;
    if (squareIndex >= 0 && squareIndex < 64) {
            pieceTexCoord = piecePositions[squareIndex];
        if (pieceTexCoord.x >= 0 && pieceTexCoord.y >= 0) {
            vec4 pieceColor = texture(overlayTexture, pieceTexCoord + fract(TexCoord * 8.0) / 8.0);
            FragColor = mix(boardColor, pieceColor, pieceColor.a);
        } else {
            FragColor = boardColor;
        }
    }
    else
    {
        FragColor = vec4(TexCoord.x, TexCoord.x, TexCoord.x, 1.0);  
    }
    
}
