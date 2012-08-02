#version 420

uniform sampler2D textures0;
//uniform sampler2D textures1;
in vec4 projTextureCoord;

layout(location = 0) out vec4 fragColor;

void main()
{
	fragColor = textureProj(textures0, projTextureCoord);
	//fragColor = vec4(1.0f,1.0f,1.0f,1.0f);
}