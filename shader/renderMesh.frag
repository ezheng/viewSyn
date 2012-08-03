#version 420
layout(location = 0) out vec4 fragColor;

in vec4 projTexCoord;
in vec4 projTexCoord1;

uniform sampler2D textures0;
uniform sampler2D textures1;

void main()
{
	//fragColor = (textureProj(textures0, projTexCoord) + textureProj(textures1, projTexCoord1))/2.0f;

	fragColor =  textureProj(textures0, projTexCoord) + 0.001 * textureProj(textures1, projTexCoord1);

	//fragColor = 0.001 * textureProj(textures0, projTexCoord) + textureProj(textures1, projTexCoord1);

}

