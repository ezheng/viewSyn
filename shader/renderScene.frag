#version 420

uniform sampler2D textures0;
in vec4 projTextureCoord;
//-------------------------------------------------------
layout(location = 0) out vec4 fragColor;

void main()
{
	fragColor = textureProj(textures0, projTextureCoord);
}